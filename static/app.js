(function () {
  const body = document.body;
  const verifyUrl = body?.dataset?.verifyUrl;
  const processUrl = body?.dataset?.processUrl;
  const saveKeyUrl = body?.dataset?.saveKeyUrl;
  const statusEl = document.querySelector(".status");

  function setStatus(text) {
    if (statusEl) statusEl.textContent = text;
  }

  function bindProcessForm() {
    const form = document.getElementById("process-form");
    const btn = document.getElementById("process-btn");
    if (!form || !btn || !processUrl) return;

    form.addEventListener("submit", function (event) {
      event.preventDefault();

      const fileInput = form.querySelector('input[name="audio_file"]');
      if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        setStatus("Error: no audio file selected.");
        return;
      }

      btn.disabled = true;
      setStatus("Uploading... 0%");

      const data = new FormData(form);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", processUrl, true);

      let transcribeTimer = null;
      let transcribePercent = 70;

      xhr.upload.onprogress = function (e) {
        if (!e.lengthComputable) return;
        const pct = Math.min(70, Math.max(1, Math.round((e.loaded / e.total) * 70)));
        setStatus(`Uploading... ${pct}%`);
      };

      xhr.upload.onload = function () {
        setStatus("Transcribing... 70%");
        transcribeTimer = window.setInterval(function () {
          transcribePercent = Math.min(95, transcribePercent + 1);
          setStatus(`Transcribing... ${transcribePercent}%`);
        }, 400);
      };

      xhr.onload = function () {
        if (transcribeTimer) window.clearInterval(transcribeTimer);

        let payload = null;
        try {
          payload = JSON.parse(xhr.responseText || "{}");
        } catch (e) {
          payload = { ok: false, message: "Unexpected server response." };
        }

        if (xhr.status >= 200 && xhr.status < 300 && payload.ok) {
          setStatus("Done... 100%");
          window.location.href = "/?page=1";
          return;
        }

        setStatus(payload.message || "Error while processing audio.");
        btn.disabled = false;
      };

      xhr.onerror = function () {
        if (transcribeTimer) window.clearInterval(transcribeTimer);
        setStatus("Network error while uploading audio.");
        btn.disabled = false;
      };

      xhr.send(data);
    });
  }

  function bindAddKeyButton() {
    const form = document.getElementById("process-form");
    const btn = document.getElementById("add-key-btn");
    if (!form || !btn || !saveKeyUrl) return;

    btn.addEventListener("click", async function () {
      const keyInput = form.querySelector('input[name="gemini_api_key"]');
      const geminiKey = keyInput ? keyInput.value.trim() : "";

      if (!geminiKey) {
        setStatus("Error: please enter your Gemini API key.");
        return;
      }

      btn.disabled = true;
      const oldText = btn.textContent;
      btn.textContent = "Checking...";
      setStatus("Checking Gemini API key...");

      try {
        const data = new FormData();
        data.append("gemini_api_key", geminiKey);

        const res = await fetch(saveKeyUrl, {
          method: "POST",
          body: data,
        });

        const payload = await res.json().catch(function () {
          return { ok: false, message: "Unexpected server response." };
        });

        setStatus(payload.message || "Unable to save Gemini API key.");
      } catch (e) {
        setStatus("Network error while saving Gemini API key.");
      } finally {
        btn.disabled = false;
        btn.textContent = oldText;
      }
    });
  }

  function bindClearHistoryModal() {
    const modal = document.getElementById("clear-history-modal");
    const openBtn = document.getElementById("clear-history-open");
    const cancelBtn = document.getElementById("clear-history-cancel");
    const confirmBtn = document.getElementById("clear-history-confirm");
    const form = document.getElementById("clear-history-form");
    if (!modal || !openBtn || !cancelBtn || !confirmBtn || !form) return;

    function closeModal() {
      modal.classList.remove("is-open");
      modal.setAttribute("aria-hidden", "true");
    }

    openBtn.addEventListener("click", function () {
      modal.classList.add("is-open");
      modal.setAttribute("aria-hidden", "false");
    });

    cancelBtn.addEventListener("click", closeModal);

    modal.addEventListener("click", function (event) {
      if (event.target === modal) closeModal();
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && modal.classList.contains("is-open")) {
        closeModal();
      }
    });

    confirmBtn.addEventListener("click", function () {
      confirmBtn.disabled = true;
      form.submit();
    });
  }

  bindProcessForm();
  bindAddKeyButton();
  bindClearHistoryModal();
  if (!verifyUrl) return;

  async function onVerifyClick(event) {
    const btn = event.target.closest(".verify-btn");
    if (!btn) return;

    const card = btn.closest(".chunk-card");
    if (!card) return;

    const index = Number(card.dataset.index);
    const transcriptEl = card.querySelector(".transcript");
    const speakerEl = card.querySelector(".speaker");
    const genderEl = card.querySelector(".gender");

    const payload = {
      index,
      transcript: transcriptEl ? transcriptEl.value : "",
      speaker: speakerEl ? speakerEl.value : "Customers",
      gender: genderEl ? genderEl.value : "Female",
      action: btn.dataset.action || "verify",
    };

    btn.disabled = true;
    const oldText = btn.textContent;
    btn.textContent = "...";

    try {
      const res = await fetch(verifyUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok || !data.ok) {
        alert(data.message || "Verify failed.");
        btn.textContent = oldText;
        return;
      }

      if (data.verified) {
        btn.dataset.action = "unverify";
        btn.textContent = "Verified";
        btn.classList.remove("is-unverified");
        btn.classList.add("is-verified");
      } else {
        btn.dataset.action = "verify";
        btn.textContent = "Verify";
        btn.classList.remove("is-verified");
        btn.classList.add("is-unverified");
      }

      setStatus(data.message);
    } catch (e) {
      alert("Network error while saving verification state.");
      btn.textContent = oldText;
    } finally {
      btn.disabled = false;
    }
  }

  document.addEventListener("click", onVerifyClick);
})();
