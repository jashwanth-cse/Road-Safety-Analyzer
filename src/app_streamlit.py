import streamlit as st
import os
import json
import re
import ast
import requests
from pathlib import Path
from retrieval import simple_keyword_retrieval
from dotenv import load_dotenv

# ============================
# 🔧 Load environment variables
# ============================
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Road Safety Intervention GPT", layout="wide")

st.title("🚦 Road Safety Intervention GPT — Groq OSS 120B")
st.caption("Powered by Groq's OpenAI-Compatible API — Built for National Road Safety Hackathon")

# Sidebar
st.sidebar.header("Settings")
if GROQ_API_KEY:
    st.sidebar.success("✅ GROQ_API_KEY loaded")
else:
    st.sidebar.error("❌ GROQ_API_KEY missing. Add it to your `.env` file.")

top_k = st.sidebar.slider("Top DB matches to include", 1, 6, 3)
temp = st.sidebar.slider("Model temperature", 0.0, 1.0, 0.2, step=0.1)
max_tokens = st.sidebar.number_input("Max tokens for model output", value=700, step=50)

DATA_PATH = ROOT / "data" / "interventions.jsonl"

st.markdown("""
### 🧠 About
This tool retrieves top-matching interventions from the database and uses Groq’s **GPT-OSS 120B model**
to recommend the most suitable **road safety interventions** with explanations and clause references.
""")

# ============================
# 🧩 Input area
# ============================
st.subheader("Describe the road safety problem")
user_input = st.text_area(
    "Enter a detailed description:",
    height=160,
    value="Busy urban intersection near a school. No zebra crossing. Cars frequently run red lights. Poor night lighting."
)

# ============================
# 🚀 Action button
# ============================
if st.button("Retrieve & Recommend"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a problem description.")
    else:
        with st.spinner("🔍 Retrieving top DB matches..."):
            top = simple_keyword_retrieval(user_input, str(DATA_PATH), topk=top_k)

        st.subheader("📋 Top Database Matches Used")
        for r in top:
            st.markdown(f"""
            **ID {r['id']} — {r['category']}**  
            Clause: `{r.get('clause','-')}`  
            Data: {r.get('data','')[:400]}...
            """)

        # ============================
        # 🧠 Build structured prompt
        # ============================
        db_snippets = []
        for r in top:
            db_snippets.append(
                f"ID {r['id']} | Category: {r.get('category')} | Clause: {r.get('clause')} | Data: {r.get('data')}"
            )
        db_text = "\n".join(db_snippets)

        prompt = f"""
You are a precise and concise road safety expert.
You must output ONLY valid JSON (no markdown, no commentary).

### Context
The following database entries describe road safety interventions.
Use ONLY these to recommend solutions for the given problem.

### User Problem
{user_input}

### Database Entries
{db_text}

### Task
1. Select the top 2–3 interventions that best solve the user's problem.
2. For each intervention include:
   - "title"
   - "description"
   - "why"
   - "support" (list of ID and clause references)
3. Include:
   - "rationale" (short paragraph)
   - "assumptions" (list)
   - "references" (list of clause references)

### Output Format
Return ONLY this valid JSON (no text outside):

{{
  "recommended_interventions": [
    {{
      "title": "string",
      "description": "string",
      "why": "string",
      "support": ["ID #", "Clause #"]
    }}
  ],
  "rationale": "string",
  "assumptions": ["string"],
  "references": ["string"]
}}
IMPORTANT:
- Return ONLY valid JSON. 
- Do not include ```json or ``` or any explanations.
- Do not write text outside the JSON object.
- Do not include comments or markdown.

"""

        st.subheader("🧾 Prompt Sent to Model")
        st.code(prompt, language="text")

        # ============================
        # 🧠 Groq API Call
        # ============================
        if not GROQ_API_KEY:
            st.warning("⚠️ GROQ_API_KEY not found. Please add it to `.env`.")
        else:
            try:
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
                api_url = "https://api.groq.com/openai/v1/chat/completions"

                payload = {
                    "model": "openai/gpt-oss-120b",
                    "messages": [
                        {"role": "system", "content": "You are a precise road safety expert that outputs strict JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": float(temp),
                    "max_tokens": int(max_tokens),
                    "top_p": 1.0
                }

                resp = requests.post(api_url, headers=headers, json=payload)
                resp.raise_for_status()
                resp_json = resp.json()
                text = resp_json["choices"][0]["message"]["content"].strip()

                # ============================
                # 🧩 Clean & Parse JSON
                # ============================
                st.subheader("🧮 Model Output (Raw)")
                cleaned = re.sub(r"```(json)?", "", text)
                cleaned = cleaned.replace("```", "").strip()
                st.code(cleaned, language="json")

                def try_parse_json(txt):
                    try:
                        return json.loads(txt)
                    except:
                        pass
                    try:
                        return ast.literal_eval(txt)
                    except:
                        pass
                    m = re.search(r"\{.*\}", txt, re.DOTALL)
                    if m:
                        try:
                            return json.loads(m.group(0))
                        except:
                            try:
                                return ast.literal_eval(m.group(0))
                            except:
                                return None
                    return None

                out = try_parse_json(cleaned)

                # ============================
                # ✅ Display parsed results
                # ============================
                if out:
                    st.success("✅ Successfully parsed model output")
                    st.subheader("📌 Recommended Interventions")
                    for idx, it in enumerate(out.get("recommended_interventions", []), start=1):
                        st.markdown(f"### {idx}. {it.get('title','No title')}")
                        st.write(it.get("description", ""))
                        st.write("**Why it helps:**", it.get("why", ""))
                        st.write("**Support:**", ", ".join(it.get("support", [])))

                    st.markdown("---")
                    st.markdown(f"**Rationale:** {out.get('rationale','')}")
                    st.markdown(f"**Assumptions:** {', '.join(out.get('assumptions', []))}")
                    st.markdown(f"**References:** {', '.join(out.get('references', []))}")
                else:
                    st.error("⚠️ Could not parse model output as JSON. Please review the raw output above.")

            except Exception as e:
                st.error(f"API Error: {e}")

st.markdown("---")
st.caption("Prototype built for National Road Safety Hackathon — uses Groq OSS 120B model for recommendations.")
