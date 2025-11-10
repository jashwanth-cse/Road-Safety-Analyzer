# Road Safety Intervention GPT - Prototype (Streamlit)

This repo contains a Streamlit prototype that uses a provided interventions CSV as the canonical database.
It retrieves relevant DB rows and constructs a prompt to an LLM (OpenAI) to generate grounded recommendations.

## Files
- data/interventions.csv : your uploaded CSV (copied)
- data/interventions.jsonl : generated from CSV by ingest script
- src/ingest.py : converts CSV -> JSONL
- src/retrieval.py : simple retrieval functions (keyword-based)
- src/app_streamlit.py : Streamlit app (UI + LLM orchestration)
- requirements.txt : Python dependencies

## Quick start (local)
1. Set up a virtualenv and install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Add your OpenAI API key to environment:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
3. Generate JSONL from CSV:
   ```bash
   python src/ingest.py
   ```
4. Run the app:
   ```bash
   streamlit run src/app_streamlit.py
   ```

## Notes
- The app currently uses keyword retrieval. For better grounding, you should run embedding indexing and semantic retrieval (optional).
- Keep temperature at 0.0 to reduce hallucinations.
- Ensure `data/official_sources` contains downloaded NHAI/MoRTH/WHO PDFs if you want to include official snippets in prompts.

