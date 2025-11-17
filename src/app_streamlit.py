import streamlit as st
import os
import json
import re
import ast
import requests
import time
from pathlib import Path
from dotenv import load_dotenv
from retrieval import simple_keyword_retrieval


# basic setup
root_dir = Path(__file__).resolve().parents[1]
load_dotenv(root_dir / ".env")
api_token = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Road Safety Intervention Tool", layout="wide")
st.title("Road Safety Intervention Tool")
st.write("Model-assisted selection of safety interventions based on a reference dataset.")

# sidebar inputs
st.sidebar.header("Options")
if api_token:
    st.sidebar.success("API token loaded")
else:
    st.sidebar.error("API token not found")

fetch_limit = st.sidebar.slider("Dataset items to match", 1, 10, 3)
model_temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.1)
token_cap = st.sidebar.number_input("Token limit", value=600, step=50)

data_file = root_dir / "data" / "interventions.jsonl"


# ------------------------------
# JSON patching helper
# ------------------------------
def try_fix_json(output_txt):
    if not output_txt or not isinstance(output_txt, str):
        return None

    txt = output_txt.strip().replace("```json", "").replace("```", "").strip()
    blocks = re.findall(r"\{[\s\S]*", txt)
    if not blocks:
        return None

    chunk = blocks[0].strip()

    # normal attempt
    try:
        return json.loads(chunk)
    except:
        pass

    # the rest is best-effort fixing
    if chunk.count('"') % 2 != 0:
        chunk += '"'

    missing = chunk.count("{") - chunk.count("}")
    if missing > 0:
        chunk += "}" * missing

    missing_br = chunk.count("[") - chunk.count("]")
    if missing_br > 0:
        chunk += "]" * missing_br

    chunk = re.sub(r",\s*([}\]])", r"\1", chunk)

    try:
        return json.loads(chunk)
    except:
        pass

    try:
        return ast.literal_eval(chunk)
    except:
        return None


# ------------------------------
# Model call wrapper
# ------------------------------
def call_model(msgs, max_tokens, temp):
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": msgs,
        "temperature": float(temp),
        "max_tokens": int(max_tokens),
        "top_p": 1.0
    }

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=20
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ------------------------------
# Preparing dataset summaries
# ------------------------------
def compress_items(entries):
    rows = []
    for e in entries:
        main_text = e.get("data", "")
        small = main_text[:350] + "..." if len(main_text) > 350 else main_text
        row = f"ID {e['id']} | Category: {e['category']} | Clause: {e['clause']} | Data: {small}"
        rows.append(row)
    return "\n".join(rows)


# ------------------------------
# Prompt builder
# ------------------------------
def shape_prompt(user_case, data_text):
    return f"""
You must select road safety interventions from a provided dataset. 
Only use entries found inside the dataset. Avoid inventing interventions or clauses. 
Choose up to three options that match the user case. Write everything completely.

User case:
{user_case}

Dataset:
{data_text}

Return strictly this JSON structure:
{{
  "recommended_interventions": [
    {{
      "title": "",
      "description": "",
      "why": "",
      "support": ["ID #", "Clause #"],
      "user_friendly_explanation": ""
    }}
  ],
  "rationale": "",
  "assumptions": [],
  "references": []
}}
"""


# ------------------------------
# Render function
# ------------------------------
def display_interventions(data_obj):
    if not data_obj or "recommended_interventions" not in data_obj:
        return None

    items = data_obj["recommended_interventions"]
    if not items:
        return None

    lines = []
    lines.append("## Recommended Interventions\n")

    for idx, it in enumerate(items, start=1):
        lines.append(f"### {idx}. {it.get('title', '')}")
        lines.append(f"- Description: {it.get('description', '')}")
        lines.append(f"- Reason: {it.get('why', '')}")
        lines.append(f"- Explanation: {it.get('user_friendly_explanation', '')}")
        lines.append("")
    return "\n".join(lines)


# ------------------------------
# Guaranteed fallback builder
# ------------------------------
def build_backup(entries):
    if not entries:
        return {
            "recommended_interventions": [
                {
                    "title": "Basic Safety Measures",
                    "description": "General signage, markings, and hazard awareness steps suitable for common road conditions.",
                    "why": "Basic improvements can still help even without a precise dataset match.",
                    "support": [],
                    "user_friendly_explanation": "Introduce standard safety markings and signs appropriate for the area."
                }
            ],
            "rationale": "Fallback used due to insufficient data.",
            "assumptions": [],
            "references": []
        }

    e = entries[0]
    return {
        "recommended_interventions": [
            {
                "title": f"Intervention based on dataset entry {e['id']}",
                "description": e.get("data", "")[:250],
                "why": "Closest dataset-linked measure identified as fallback option.",
                "support": [f"ID {e['id']}", f"Clause {e['clause']}"],
                "user_friendly_explanation": "A safety approach derived from the closest known dataset entry."
            }
        ],
        "rationale": "Generated because model output was incomplete.",
        "assumptions": [],
        "references": []
    }


# ------------------------------
# Interface
# ------------------------------
st.subheader("Describe the road safety issue")
case_text = st.text_area("Enter description:", height=150)

if st.button("Generate"):
    if not case_text.strip():
        st.warning("Please write some details.")
        st.stop()

    with st.spinner("Looking up dataset..."):
        found_items = simple_keyword_retrieval(case_text, str(data_file), topk=fetch_limit)

    st.subheader("Dataset Matches")
    if not found_items:
        st.write("No matching dataset entries were located.")
    else:
        for f in found_items:
            st.markdown(f"**ID {f['id']} – {f['category']}** | Clause {f['clause']}")
            st.write(f["data"][:200] + "...")
            st.markdown("---")

    dataset_text = compress_items(found_items)
    prompt = shape_prompt(case_text, dataset_text)

    start = time.time()
    raw_model_out = call_model(
        [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        token_cap,
        model_temp
    )
    end = time.time()

    st.subheader("Raw Output")
    st.code(raw_model_out)

    parsed = try_fix_json(raw_model_out)

    st.subheader("Final Recommendations")

    if parsed and parsed.get("recommended_interventions"):
        final_txt = display_interventions(parsed)
        st.markdown(final_txt)
    else:
        backup_obj = build_backup(found_items)
        st.markdown(display_interventions(backup_obj))

    st.info(f"Time taken: {end - start:.2f} seconds")
