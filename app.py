import os
import re
import json
import uuid
import zipfile
import shutil
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    jsonify,
    session,
)
from authlib.integrations.flask_client import OAuth
import pandas as pd
import soundfile as sf
import librosa
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# =========================================================
# APP CONFIG
# =========================================================

PER_PAGE = 10

CSV_COLUMNS = [
    "speaker_id",
    "topic",
    "subtopic",
    "paragraph_id",
    "sentence_id",
    "transcript",
    "duration",
    "audio_path",
    "save_dir",
    "start",
    "end",
    "gender",
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-this")
app.permanent_session_lifetime = timedelta(days=7)

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
DEV_BYPASS_LOGIN = os.getenv("DEV_BYPASS_LOGIN", "").strip().lower() in {"1", "true", "yes", "on"}
DEV_BYPASS_NAME = os.getenv("DEV_BYPASS_NAME", "Local Dev User").strip() or "Local Dev User"
DEV_BYPASS_EMAIL = os.getenv("DEV_BYPASS_EMAIL", "local-dev@example.com").strip() or "local-dev@example.com"
FLASK_DEBUG_MODE = os.getenv("FLASK_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}

oauth = OAuth(app)
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        client_kwargs={"scope": "openid email profile"},
    )


# =========================================================
# AUTH HELPERS
# =========================================================

def get_dev_user() -> dict:
    return {
        "sub": "local-dev-user",
        "email": DEV_BYPASS_EMAIL,
        "name": DEV_BYPASS_NAME,
        "picture": "",
    }


def get_current_user() -> dict | None:
    user = session.get("user")
    if not user and DEV_BYPASS_LOGIN:
        user = get_dev_user()
        session["user"] = user
    return user if isinstance(user, dict) else None


def get_user_id() -> str:
    user = get_current_user()
    if not user or not user.get("sub"):
        raise RuntimeError("User is not logged in.")
    return sanitize_name(str(user["sub"]))


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not get_current_user():
            if request.path in {"/process-ajax", "/verify"} or request.is_json:
                return jsonify({
                    "ok": False,
                    "message": "Unauthorized. Please sign in again.",
                    "login_url": url_for("login"),
                }), 401
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)
    return wrapped_view


# =========================================================
# USER / WORKSPACE HELPERS
# =========================================================

def get_base_dir() -> Path:
    user_id = get_user_id()
    return Path.cwd() / "storage" / "users" / user_id / "workspace"


def get_dirs() -> dict[str, Path]:
    base_dir = get_base_dir()
    raw_dir = base_dir / "raw_audio"
    processed_dir = base_dir / "processed_audio"
    preview_dir = base_dir / "preview_audio"
    chunks_dir = base_dir / "chunks"
    csv_dir = base_dir / "csv"
    export_dir = base_dir / "exports"
    state_dir = base_dir / "session_state"

    return {
        "BASE_DIR": base_dir,
        "RAW_DIR": raw_dir,
        "PROCESSED_DIR": processed_dir,
        "PREVIEW_DIR": preview_dir,
        "CHUNKS_DIR": chunks_dir,
        "CSV_DIR": csv_dir,
        "CSV_PATH": csv_dir / "dataset.csv",
        "EXPORT_DIR": export_dir,
        "STATE_DIR": state_dir,
        "STATE_PATH": state_dir / "state.json",
    }


def ensure_base_dirs():
    dirs = get_dirs()
    for key, path in dirs.items():
        if key.endswith("_DIR") or key == "BASE_DIR":
            path.mkdir(parents=True, exist_ok=True)


# =========================================================
# GENERAL HELPERS
# =========================================================

def sanitize_name(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "audio"


def format_time_range(start: float, end: float) -> str:
    return f"{start:.1f}-{end:.1f}s"


def int_duration(start: float, end: float) -> int:
    return int(max(0, end - start))


def sec_to_time(sec: float) -> str:
    sec = int(sec)
    m = sec // 60
    s = sec % 60
    return f"{m}:{s:02d}"


def round_seconds(value: float) -> float:
    return round(float(value), 3)


def clamp_page(page: int, total: int) -> int:
    if total <= 0:
        return 1
    return max(1, min(page, total))


def build_pagination(page: int, total_pages: int) -> list[int | str]:
    if total_pages <= 1:
        return [1]

    pages: list[int | str] = []
    window = {1, 2, total_pages - 1, total_pages, page - 1, page, page + 1}
    valid = sorted(p for p in window if 1 <= p <= total_pages)

    last = 0
    for p in valid:
        if p - last > 1:
            pages.append("...")
        pages.append(p)
        last = p
    return pages


def ui_to_csv_speaker(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"agents", "agent"}:
        return "agents"
    if v in {"customers", "customer"}:
        return "customers"
    return "customers"


def csv_to_ui_speaker(value: str) -> str:
    return "Agents" if ui_to_csv_speaker(value) == "agents" else "Customers"


def ui_to_csv_gender(value: str) -> str:
    v = (value or "").strip().lower()
    if v == "male":
        return "male"
    if v == "female":
        return "female"
    return "female"


def csv_to_ui_gender(value: str) -> str:
    return "Male" if ui_to_csv_gender(value) == "male" else "Female"


# =========================================================
# CSV / STATE HELPERS
# =========================================================

def ensure_csv():
    ensure_base_dirs()
    csv_path = get_dirs()["CSV_PATH"]
    if not csv_path.exists():
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(csv_path, index=False, encoding="utf-8-sig")


def load_csv() -> pd.DataFrame:
    ensure_csv()
    csv_path = get_dirs()["CSV_PATH"]

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.DataFrame(columns=CSV_COLUMNS)

    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df[CSV_COLUMNS].copy()


def save_csv(df: pd.DataFrame):
    ensure_base_dirs()
    csv_path = get_dirs()["CSV_PATH"]

    out = df.copy()
    for col in CSV_COLUMNS:
        if col not in out.columns:
            out[col] = ""

    out["sentence_id"] = pd.to_numeric(out["sentence_id"], errors="coerce").fillna(0).astype(int)
    out["duration"] = pd.to_numeric(out["duration"], errors="coerce").round(3)
    out["start"] = pd.to_numeric(out["start"], errors="coerce").round(3)
    out["end"] = pd.to_numeric(out["end"], errors="coerce").round(3)

    out["save_dir"] = out["save_dir"].fillna("").astype(str)
    out["audio_path"] = out["audio_path"].fillna("").astype(str)
    out["topic"] = out["topic"].fillna("").astype(str)
    out["subtopic"] = out["subtopic"].fillna("").astype(str)
    out["transcript"] = out["transcript"].fillna("").astype(str)
    out["speaker_id"] = out["speaker_id"].fillna("").astype(str)
    out["gender"] = out["gender"].fillna("").astype(str)

    out = out.sort_values(by=["save_dir", "sentence_id"], kind="stable").reset_index(drop=True)
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")


def add_or_replace_csv_rows(rows_to_add: list[dict]):
    df = load_csv()
    if rows_to_add:
        target_save_dir = str(rows_to_add[0]["save_dir"])
        df = df[df["save_dir"].astype(str) != target_save_dir]
        df = pd.concat([df, pd.DataFrame(rows_to_add)], ignore_index=True)
    save_csv(df)


def default_state(status: str = "Ready.") -> dict:
    return {
        "rows": [],
        "raw_audio_path": "",
        "status": status,
        "topic": "",
        "subtopic": "",
        "file_name": "",
        "gemini_api_key": "",
    }


def load_state() -> dict:
    ensure_base_dirs()
    state_path = get_dirs()["STATE_PATH"]

    if not state_path.exists():
        return default_state()

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return default_state()

    if not isinstance(state, dict):
        return default_state()

    state.setdefault("rows", [])
    state.setdefault("raw_audio_path", "")
    state.setdefault("status", "Ready.")
    state.setdefault("topic", "")
    state.setdefault("subtopic", "")
    state.setdefault("file_name", "")
    state.setdefault("gemini_api_key", "")
    return state


def save_state(state: dict):
    ensure_base_dirs()
    state_path = get_dirs()["STATE_PATH"]

    merged = default_state()
    if isinstance(state, dict):
        merged.update(state)

    state_path.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")


def reset_storage():
    dirs = get_dirs()
    base_dir = dirs["BASE_DIR"]

    old_state = load_state()
    saved_key = old_state.get("gemini_api_key", "")

    if base_dir.exists():
        shutil.rmtree(base_dir, ignore_errors=True)

    ensure_base_dirs()

    new_state = default_state("History cleared. Storage reset complete.")
    new_state["gemini_api_key"] = saved_key
    save_state(new_state)


# =========================================================
# AUDIO / GEMINI HELPERS
# =========================================================

def extract_json_block(text: str) -> dict:
    text = (text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    brace = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(1))
        except Exception:
            pass

    raise ValueError(f"Invalid JSON from Gemini. Raw response: {text[:1000]}")


def preprocess_audio(raw_path: Path) -> Path:
    ensure_base_dirs()
    processed_dir = get_dirs()["PROCESSED_DIR"]

    y, sr = librosa.load(str(raw_path), sr=None, mono=True)

    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000, res_type="kaiser_best")
        sr = 16000

    if len(y) > 0:
        y = librosa.util.normalize(y)
        y, _ = librosa.effects.trim(y, top_db=30)

    clean_path = processed_dir / f"{sanitize_name(raw_path.name)}_clean.wav"
    sf.write(str(clean_path), y, sr)
    return clean_path


def export_audio_slice(source_audio_path: Path, start: float, end: float, out_path: Path):
    data, sr = sf.read(str(source_audio_path))
    s = max(0, int(start * sr))
    e = max(s, int(end * sr))

    if e <= s:
        raise ValueError("Invalid slice range")

    chunk = data[s:e]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), chunk, sr)


def save_preview_chunk(clean_audio_path: Path, start: float, end: float, save_dir: str, audio_name: str) -> Path:
    preview_dir = get_dirs()["PREVIEW_DIR"]
    preview_folder = preview_dir / save_dir
    preview_folder.mkdir(parents=True, exist_ok=True)
    out_path = preview_folder / audio_name
    export_audio_slice(clean_audio_path, start, end, out_path)
    return out_path


def gemini_transcribe(clean_audio_path: Path, user_api_key: str) -> dict:
    if not user_api_key:
        raise ValueError("Gemini API key is missing.")

    genai.configure(api_key=user_api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    with open(clean_audio_path, "rb") as f:
        audio_bytes = f.read()

    prompt = """
You are a Khmer speech transcription and speaker labeling system.

Return ONLY strict JSON.
Do not use markdown.
Do not add explanation.
Do not add any text before or after JSON.

Schema:
{
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "speaker_id": "agents",
      "text": "..."
    }
  ]
}

Rules:
- "speaker_id" must be only "agents" or "customers"
- Output segments in chronological order
- Use timestamps in seconds
- Do not overlap segments if possible
- Transcribe exactly what is spoken
- Use Khmer Unicode for Khmer speech
- Keep English words in English if code-switching happens
- Do not translate
- Do not summarize
- Do not paraphrase
- Do not include speaker names inside text
- Keep segments review-friendly, strictly under 15 seconds
"""

    response = model.generate_content([
        {"mime_type": "audio/wav", "data": audio_bytes},
        prompt,
    ])

    raw_text = response.text if hasattr(response, "text") and response.text else str(response)
    return extract_json_block(raw_text)


def validate_gemini_api_key(user_api_key: str):
    if not user_api_key:
        raise ValueError("Gemini API key is missing.")

    genai.configure(api_key=user_api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content("Reply with OK.")
    raw_text = response.text if hasattr(response, "text") and response.text else str(response)

    if not raw_text.strip():
        raise ValueError("Gemini returned an empty response while validating the API key.")


def validate_segments(payload: dict, audio_duration: float) -> list[dict]:
    segments = payload.get("segments", [])
    repaired = []

    for item in segments:
        try:
            start = float(item.get("start", 0))
            end = float(item.get("end", 0))
            speaker_id = ui_to_csv_speaker(item.get("speaker_id", "customers"))
            text = str(item.get("text", "")).strip()

            if start < 0:
                start = 0.0
            if end > audio_duration:
                end = audio_duration

            if end <= start or (end - start) < 0.3 or not text:
                continue

            repaired.append({"start": start, "end": end, "speaker_id": speaker_id, "text": text})
        except Exception:
            continue

    repaired.sort(key=lambda x: x["start"])

    fixed = []
    prev_end = None
    for seg in repaired:
        start = seg["start"]
        end = seg["end"]

        if prev_end is not None and start < prev_end and prev_end - start <= 0.3:
            start = prev_end

        if end <= start:
            continue

        fixed.append({"start": start, "end": end, "speaker_id": seg["speaker_id"], "text": seg["text"]})
        prev_end = end

    return fixed


def process_audio(upload_path: str, original_name: str, topic: str, subtopic: str) -> tuple[list[dict], str, str]:
    ensure_base_dirs()
    raw_dir = get_dirs()["RAW_DIR"]

    if not upload_path:
        return [], "", "Error: no audio file selected."

    source = Path(upload_path)
    if not source.exists():
        return [], "", f"Error: uploaded temp file not found: {source}"

    state = load_state()
    user_api_key = state.get("gemini_api_key", "").strip()
    if not user_api_key:
        return [], "", "Error: Please enter your Gemini API key."

    safe_stem = sanitize_name(original_name or source.name)
    suffix = Path(original_name).suffix.lower() if original_name else source.suffix.lower()
    raw_path = raw_dir / f"{safe_stem}{suffix}"

    with open(source, "rb") as fsrc, open(raw_path, "wb") as fdst:
        fdst.write(fsrc.read())

    clean_path = preprocess_audio(raw_path)
    info = sf.info(str(clean_path))
    audio_duration = info.frames / info.samplerate

    gemini_payload = gemini_transcribe(clean_path, user_api_key)
    segments = validate_segments(gemini_payload, audio_duration)

    if not segments:
        return [], str(raw_path), "Error: no valid segments returned from Gemini."

    rows = []
    for i, seg in enumerate(segments, start=1):
        temp_audio_name = f"{safe_stem}_temp_{i:02d}.wav"
        preview_path = save_preview_chunk(
            clean_audio_path=clean_path,
            start=seg["start"],
            end=seg["end"],
            save_dir=safe_stem,
            audio_name=temp_audio_name,
        )

        rows.append(
            {
                "topic": (topic or "").strip(),
                "subtopic": (subtopic or "").strip(),
                "paragraph_id": 1,
                "source_sentence_id": i,
                "audio_name": temp_audio_name,
                "save_dir": safe_stem,
                "start": seg["start"],
                "end": seg["end"],
                "duration": int_duration(seg["start"], seg["end"]),
                "speaker_id": seg["speaker_id"],
                "gemini_speaker_id": seg["speaker_id"],
                "transcript": seg["text"],
                "gemini_text": seg["text"],
                "gender": "female",
                "verified": False,
                "raw_audio_path": str(raw_path),
                "clean_audio_path": str(clean_path),
                "preview_audio_path": str(preview_path),
                "finalized": False,
            }
        )

    return rows, str(raw_path), f"Ready for review. {len(rows)} chunks created."


def finalize_audio(rows: list[dict]) -> tuple[list[dict], str]:
    ensure_base_dirs()
    chunks_dir = get_dirs()["CHUNKS_DIR"]

    if not rows:
        return rows, "Error: no audio loaded."

    verified_rows = [dict(r) for r in rows if r.get("verified", False)]
    if not verified_rows:
        return rows, "Error: no verified chunks to finalize."

    base_name = verified_rows[0]["save_dir"]
    clean_audio_path = Path(verified_rows[0]["clean_audio_path"])
    final_chunk_folder = chunks_dir / base_name
    final_chunk_folder.mkdir(parents=True, exist_ok=True)

    verified_rows.sort(key=lambda x: float(x["start"]))
    final_csv_rows = []

    for old_file in final_chunk_folder.glob("*.wav"):
        try:
            old_file.unlink()
        except Exception:
            pass

    for new_idx, row in enumerate(verified_rows, start=1):
        final_audio_name = f"{base_name}_{new_idx:02d}.wav"
        final_chunk_path = final_chunk_folder / final_audio_name

        export_audio_slice(
            source_audio_path=clean_audio_path,
            start=float(row["start"]),
            end=float(row["end"]),
            out_path=final_chunk_path,
        )

        start_sec = float(row["start"])
        end_sec = float(row["end"])
        duration_sec = round_seconds(max(0, end_sec - start_sec))

        final_csv_rows.append(
            {
                "speaker_id": row["speaker_id"],
                "topic": row["topic"],
                "subtopic": row["subtopic"],
                "paragraph_id": 1,
                "sentence_id": new_idx,
                "transcript": row["transcript"],
                "duration": duration_sec,
                "audio_path": final_audio_name,
                "save_dir": base_name,
                "start": round_seconds(start_sec),
                "end": round_seconds(end_sec),
                "gender": row["gender"],
            }
        )

    add_or_replace_csv_rows(final_csv_rows)

    for r in rows:
        r["finalized"] = False

    verified_sorted = sorted([r for r in rows if r.get("verified", False)], key=lambda x: float(x["start"]))
    for new_idx, verified in enumerate(verified_sorted, start=1):
        verified["finalized"] = True
        verified["final_sentence_id"] = new_idx
        verified["final_audio_name"] = f"{base_name}_{new_idx:02d}.wav"

    return rows, f"Finalize complete: {len(final_csv_rows)} segments saved."


def export_dataset_zip() -> tuple[Path, str]:
    ensure_csv()
    dirs = get_dirs()
    csv_path = dirs["CSV_PATH"]
    export_dir = dirs["EXPORT_DIR"]
    chunks_dir = dirs["CHUNKS_DIR"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    zip_path = export_dir / f"dataset_export_{timestamp}_{unique_id}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if csv_path.exists():
            zf.write(csv_path, arcname="dataset.csv")

        if chunks_dir.exists():
            for file_path in chunks_dir.rglob("*"):
                if file_path.is_file():
                    arcname = Path("chunks") / file_path.relative_to(chunks_dir)
                    zf.write(file_path, arcname=str(arcname))

    return zip_path, f"Export ready: {zip_path.name}"


def safe_media_path(path_str: str) -> Path | None:
    if not path_str:
        return None

    p = Path(path_str)
    if not p.exists() or not p.is_file():
        return None

    base_dir = get_dirs()["BASE_DIR"].resolve()
    rp = p.resolve()

    if str(rp).startswith(str(base_dir)):
        return rp
    return None


# =========================================================
# AUTH ROUTES
# =========================================================

@app.get("/login")
def login():
    if get_current_user():
        return redirect(url_for("index"))

    if DEV_BYPASS_LOGIN:
        session.permanent = True
        session["user"] = get_dev_user()
        return redirect(url_for("index"))

    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return (
            "Google OAuth is not configured. "
            "Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env",
            500,
        )

    redirect_uri = url_for("auth_callback", _external=True)
    return oauth.google.authorize_redirect(
        redirect_uri,
        prompt="select_account"
    )


@app.get("/auth/callback")
def auth_callback():
    token = oauth.google.authorize_access_token()
    userinfo = token.get("userinfo")

    if not userinfo:
        userinfo = oauth.google.parse_id_token(token)

    if not userinfo or not userinfo.get("sub"):
        return "Failed to get Google user info.", 400

    session.permanent = True
    session["user"] = {
        "sub": str(userinfo["sub"]),
        "email": userinfo.get("email", ""),
        "name": userinfo.get("name", "Google User"),
        "picture": userinfo.get("picture", ""),
    }

    return redirect(url_for("index"))


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# =========================================================
# APP ROUTES
# =========================================================

@app.get("/")
@login_required
def index():
    ensure_base_dirs()

    page = request.args.get("page", default=1, type=int)
    state = load_state()
    rows = state.get("rows", [])

    total_items = len(rows)
    total_pages = max(1, (total_items + PER_PAGE - 1) // PER_PAGE)
    page = clamp_page(page, total_pages)

    start = (page - 1) * PER_PAGE
    end = start + PER_PAGE
    visible_rows = rows[start:end]

    page_items = build_pagination(page, total_pages)

    return render_template(
        "index.html",
        rows=visible_rows,
        page=page,
        total_pages=total_pages,
        page_items=page_items,
        total_items=total_items,
        per_page=PER_PAGE,
        status=state.get("status", "Ready."),
        raw_audio_path=state.get("raw_audio_path", ""),
        topic=state.get("topic", ""),
        subtopic=state.get("subtopic", ""),
        file_name=state.get("file_name", ""),
        current_user=get_current_user(),
        format_time_range=format_time_range,
        csv_to_ui_speaker=csv_to_ui_speaker,
        csv_to_ui_gender=csv_to_ui_gender,
    )


@app.post("/process")
@login_required
def process_route():
    ensure_base_dirs()
    raw_dir = get_dirs()["RAW_DIR"]

    page = request.form.get("page", default=1, type=int)
    topic = request.form.get("topic", "").strip()
    subtopic = request.form.get("subtopic", "").strip()
    gemini_key = request.form.get("gemini_api_key", "").strip()
    uploaded = request.files.get("audio_file")

    state = load_state()

    if gemini_key:
        state["gemini_api_key"] = gemini_key

    if not uploaded or not uploaded.filename:
        state["status"] = "Error: no audio file selected."
        state["topic"] = topic
        state["subtopic"] = subtopic
        save_state(state)
        return redirect(url_for("index", page=page))

    temp_upload = raw_dir / f"_upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{sanitize_name(uploaded.filename)}"
    uploaded.save(temp_upload)

    try:
        rows, raw_audio_path, status = process_audio(str(temp_upload), uploaded.filename, topic, subtopic)
        state["rows"] = rows
        state["raw_audio_path"] = raw_audio_path
        state["status"] = status
        state["topic"] = topic
        state["subtopic"] = subtopic
        state["file_name"] = uploaded.filename
        save_state(state)
    except Exception as e:
        state["status"] = f"Error: {e}"
        state["topic"] = topic
        state["subtopic"] = subtopic
        state["file_name"] = uploaded.filename if uploaded else ""
        save_state(state)
    finally:
        try:
            temp_upload.unlink(missing_ok=True)
        except Exception:
            pass

    return redirect(url_for("index", page=1))


@app.post("/process-ajax")
@login_required
def process_ajax_route():
    ensure_base_dirs()
    raw_dir = get_dirs()["RAW_DIR"]

    topic = request.form.get("topic", "").strip()
    subtopic = request.form.get("subtopic", "").strip()
    gemini_key = request.form.get("gemini_api_key", "").strip()
    uploaded = request.files.get("audio_file")

    state = load_state()

    if gemini_key:
        state["gemini_api_key"] = gemini_key

    if not uploaded or not uploaded.filename:
        msg = "Error: no audio file selected."
        state["status"] = msg
        state["topic"] = topic
        state["subtopic"] = subtopic
        save_state(state)
        return jsonify({"ok": False, "message": msg}), 400

    temp_upload = raw_dir / f"_upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{sanitize_name(uploaded.filename)}"
    uploaded.save(temp_upload)

    try:
        rows, raw_audio_path, status = process_audio(str(temp_upload), uploaded.filename, topic, subtopic)
        state["rows"] = rows
        state["raw_audio_path"] = raw_audio_path
        state["status"] = status
        state["topic"] = topic
        state["subtopic"] = subtopic
        state["file_name"] = uploaded.filename
        save_state(state)
        return jsonify({"ok": True, "message": status})
    except Exception as e:
        msg = f"Error: {e}"
        state["status"] = msg
        state["topic"] = topic
        state["subtopic"] = subtopic
        state["file_name"] = uploaded.filename if uploaded else ""
        save_state(state)
        return jsonify({"ok": False, "message": msg}), 500
    finally:
        try:
            temp_upload.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/save-gemini-key")
@login_required
def save_gemini_key_route():
    gemini_key = request.form.get("gemini_api_key", "").strip()
    state = load_state()

    if not gemini_key:
        msg = "Error: please enter your Gemini API key."
        state["status"] = msg
        save_state(state)
        return jsonify({"ok": False, "message": msg}), 400

    try:
        validate_gemini_api_key(gemini_key)
        state["gemini_api_key"] = gemini_key
        state["status"] = "Gemini API key added successfully."
        save_state(state)
        return jsonify({"ok": True, "message": state["status"]})
    except Exception as e:
        msg = f"Error: unable to validate Gemini API key. {e}"
        state["status"] = msg
        save_state(state)
        return jsonify({"ok": False, "message": msg}), 400


@app.post("/verify")
@login_required
def verify_route():
    payload = request.get_json(silent=True) or {}
    index = int(payload.get("index", -1))
    transcript = (payload.get("transcript") or "").strip()
    speaker_ui = payload.get("speaker", "Customers")
    gender_ui = payload.get("gender", "Female")
    action = payload.get("action", "verify")

    state = load_state()
    rows = state.get("rows", [])

    if index < 0 or index >= len(rows):
        return jsonify({"ok": False, "message": "Invalid row index."}), 400

    rows[index]["transcript"] = transcript
    rows[index]["speaker_id"] = ui_to_csv_speaker(speaker_ui)
    rows[index]["gender"] = ui_to_csv_gender(gender_ui)

    if action == "verify":
        if not transcript:
            return jsonify({"ok": False, "message": "Transcript is empty."}), 400
        rows[index]["verified"] = True
        msg = f"Verified segment {index + 1}"
    else:
        rows[index]["verified"] = False
        msg = f"Unverified segment {index + 1}"

    state["rows"] = rows
    state["status"] = msg
    save_state(state)

    return jsonify({"ok": True, "message": msg, "verified": rows[index]["verified"]})


@app.post("/export")
@login_required
def export_route():
    state = load_state()
    rows = state.get("rows", [])

    try:
        rows, finalize_msg = finalize_audio(rows)
        state["rows"] = rows
        state["status"] = finalize_msg
        save_state(state)

        zip_path, export_msg = export_dataset_zip()
        state["status"] = export_msg
        save_state(state)

        response = send_file(zip_path, as_attachment=True)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    except Exception as e:
        state["status"] = f"Error: {e}"
        save_state(state)
        return redirect(url_for("index"))


@app.post("/clear-history")
@login_required
def clear_history_route():
    try:
        reset_storage()
    except Exception as e:
        state = load_state()
        saved_key = state.get("gemini_api_key", "")
        state = default_state(f"Error while clearing history: {e}")
        state["gemini_api_key"] = saved_key
        save_state(state)
    return redirect(url_for("index", page=1))


@app.get("/media")
@login_required
def media_route():
    path_str = request.args.get("path", "")
    safe_path = safe_media_path(path_str)
    if not safe_path:
        return "Not found", 404
    return send_file(safe_path)


@app.get("/debug-user")
@login_required
def debug_user_route():
    return {
        "user": get_current_user(),
        "base_dir": str(get_dirs()["BASE_DIR"]),
    }


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=FLASK_DEBUG_MODE, use_reloader=False)
