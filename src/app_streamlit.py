import streamlit as st
import os
import json
import textwrap
from pathlib import Path
from retrieval import simple_keyword_retrieval

# ✅ Load .env (explicit path)
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=ROOT / ".env")

# ✅ Try importing OpenAI safely
try:
    import openai
except ImportError:
    openai = None

# ✅ Load key from .env or environment
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY
    st.sidebar.write("🔑 API Key Loaded: ✅")
else:
    st.sidebar.write("🔑 API Key Loaded: ❌ (check your .env)")
    st.warning("⚠️ OPENAI_API_KEY not set. Please create a .env file in project root with your API key.")


DATA_PATH = ROOT / "data" / "interventions.jsonl"

st.set_page_config(page_title="Road Safety Intervention GPT", layout="wide")
st.title("Road Safety Intervention GPT — Streamlit Prototype")

st.markdown("""
This prototype uses your uploaded interventions database (interventions.csv) as the canonical source.
It retrieves top-matching DB rows and sends them to an LLM (OpenAI) to produce grounded recommendations.
""")

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model (OpenAI)", ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"], index=0)
    top_k = st.slider("Top DB matches to include", 1, 6, 3)
    temp = st.slider("LLM temperature", 0.0, 1.0, 0.0, step=0.1)
    max_tokens = st.number_input("Max tokens for LLM", value=700, step=50)

st.subheader("Describe the road safety problem")
user_input = st.text_area("Problem description", height=160, value="Busy urban intersection near a school. No zebra crossing. Cars frequently run red lights. Poor night lighting.")

if st.button("Retrieve & Recommend"):
    if not user_input.strip():
        st.warning("Please enter a problem description.")
    else:
        with st.spinner("Retrieving top DB matches..."):
            top = simple_keyword_retrieval(user_input, str(DATA_PATH), topk=top_k)
        st.subheader("Top database matches used")
        for r in top:
            st.markdown(f"**ID {r['id']} — {r['category']}**  \nClause: {r.get('clause','-')}  \nData: {r.get('data','')[:400]}")
            # Build prompt
            db_snippets = []
            for r in top:
                db_snippets.append(
                    f"ID {r['id']} | Category: {r.get('category')} | Clause: {r.get('clause')} | Data: {r.get('data')}"
                )
            db_text = "\n".join(db_snippets)

            prompt = f"""
    You are a concise road safety expert. Only use the database rows provided below to recommend interventions.

    User problem:
    {user_input}

    Database rows:
    {db_text}

    Task:
    1) Select the top 2–3 recommended interventions from the database rows that best address the user's problem.
    2) For each recommended intervention include: title, short description, why it helps, and support (list of IDs/clause text).
    3) Provide an overall rationale and explicit assumptions.
    4) ONLY cite IDs/clause text present in the database rows above; do NOT invent other sources.

    Return a JSON object with keys:
    recommended_interventions (list), rationale (string), references (list), assumptions (list).
    """

            st.code(prompt, language="text")
        # Call OpenAI if available
        if openai is None:
            st.info("OpenAI SDK not available in this environment. Paste the prompt into your account to test.")
        else:
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                st.warning("OPENAI_API_KEY not set on server. Set it in environment to enable LLM calls.")
            else:
                openai.api_key = key
                try:
                    resp = openai.ChatCompletion.create(
                        model=model,
                        messages=[{"role":"system","content":"You are a precise road safety expert."},
                                  {"role":"user","content":prompt}],
                        temperature=float(temp),
                        max_tokens=int(max_tokens)
                    )
                    text = resp["choices"][0]["message"]["content"]
                    st.subheader("LLM Output (raw)")
                    st.code(text, language="json")
                    # try to parse JSON
                    try:
                        out = json.loads(text)
                        st.subheader("Parsed Recommendations")
                        for idx, it in enumerate(out.get("recommended_interventions", []), start=1):
                            st.markdown(f"**{idx}. {it.get('title','No title')}**")
                            st.write(it.get("description",""))
                            st.write("Why:", it.get("why",""))
                            st.write("Support:", it.get("support",""))
                    except Exception as e:
                        st.error("Could not parse LLM output as JSON. See raw output.")
                except Exception as e:
                    st.error(f"OpenAI API error: {e}")

st.markdown("---")
st.caption("Prototype built for National Road Safety Hackathon.")

