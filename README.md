# Khmer STT Tool v4

Khmer STT Tool v4 is a Flask-based web app for Khmer speech transcription and review.  
It lets you upload audio, automatically split/transcribe it with Gemini, verify each chunk manually, and export the final dataset as a ZIP file.

## Features

- Upload audio files from the browser
- Automatic preprocessing to clean audio
- Gemini-based Khmer transcription and speaker labeling
- Chunk-by-chunk review interface
- Edit transcript, speaker, and gender before export
- Export verified dataset as ZIP
- Clear history and fully reset project storage

---

## Project Structure

```bash
V4/
├── app.py
├── requirements.txt
├── .env
├── static/
│   ├── app.js
│   └── styles.css
├── templates/
│   └── index.html
└── khmer_stt_data/
    ├── raw_audio/
    ├── processed_audio/
    ├── preview_audio/
    ├── chunks/
    ├── csv/
    ├── exports/


Getting Started

Follow these steps to run the project locally.

1. Clone the Repository
git clone https://github.com/kimyongsien/khmer_stt_annotation_toolss
cd your-repo-name

2. Create Virtual Environment
Windows
python -m venv .venv
.venv\Scripts\activate
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate


3. Install Dependencies
pip install -r requirements.txt

If requirements.txt is missing, install manually:

pip install flask pandas soundfile librosa google-generativeai python-dotenv


4. Setup Environment Variable (IMPORTANT)

Create a file named .env in the root directory:

GEMINI_API_KEY=your_api_key_here

5. Run the Application
python app.py

6. Open in Browser

Go to:
http://127.0.0.1:7860