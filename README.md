
# Meem AI — Chat over Maryam's LinkedIn posts

## Quick Start (Local)
1. Install Python 3.10+.
2. In terminal:
   ```bash
   pip install -r requirements.txt
   export OPENAI_API_KEY=sk-...   # put your key
   export MEEM_TOKEN=your-secret  # optional to keep the link private
   streamlit run app.py
   ```
3. Open the local URL shown by Streamlit.

## Deploy — Streamlit Community Cloud (Free)
1. Push these files to a new GitHub repo (e.g., `meem-ai`).
2. Go to https://streamlit.io/cloud → "New app" → connect your repo.
3. Set:
   - **Main file path**: `app.py`
   - **Python version**: 3.10 or newer
   - **Secrets**:
     - `OPENAI_API_KEY = sk-...`
     - `MEEM_TOKEN = your-secret` (optional, for private-by-link)
4. Click **Deploy**. Share the app URL with people you choose.

## Deploy — Hugging Face Spaces
1. Create a Space → Type: **Streamlit**.
2. Upload all files (drag & drop ZIP contents).
3. In **Secrets** (Settings → Variables & secrets), add:
   - `OPENAI_API_KEY`
   - `MEEM_TOKEN` (optional)
4. Wait for the build to finish. Share the Space URL.

## Notes
- The app only answers from `posts.csv`. To update posts, replace that file and redeploy.
- You can also let users upload a CSV at runtime from the app's expander.
