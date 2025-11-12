import json
import re
from pathlib import Path
from difflib import SequenceMatcher

def load_jsonl(path):
    """Load JSONL file into a list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def normalize(text):
    """Normalize text: lowercase, remove punctuation, trim spaces."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def simple_keyword_retrieval(query, jsonl_path, topk=5):
    """Retrieve top-k relevant entries from database using simple keyword matching."""
    q = normalize(query)
    q_tokens = q.split()
    docs = load_jsonl(jsonl_path)
    results = []

    common_phrases = [
        "pedestrian", "speed", "curve", "intersection", "school", "bus stop",
        "pothole", "lighting", "guardrail", "cycle", "zebra", "crossing"
    ]

    for r in docs:
        score = 0
        text_fields = {
            "problem": 3.0,
            "data": 2.5,
            "category": 1.5,
            "type": 1.2,
            "raw": 1.0,
        }

        # Normalize all text fields
        normalized = {k: normalize(v) for k, v in r.items() if isinstance(v, str)}

        # Token matches
        for field, weight in text_fields.items():
            ftxt = normalized.get(field, "")
            for tok in q_tokens:
                if tok in ftxt:
                    score += weight

        # Phrase boost
        for phrase in common_phrases:
            if phrase in q and any(phrase in normalized.get(f, "") for f in ["problem", "data"]):
                score += 5

        # Clause number match bonus
        if normalized.get("clause", "") in q:
            score += 3

        # Fallback fuzzy similarity for low-score matches
        sim = SequenceMatcher(None, q, normalized.get("problem", "")).ratio()
        score += sim * 2

        results.append((score, r))

    results.sort(key=lambda x: x[0], reverse=True)
    filtered = [r for s, r in results if s > 0]
    if not filtered:
        filtered = [r for s, r in results[:topk]]
    return filtered[:topk]

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "busy intersection near school with no crossing and poor lighting"
    path = Path(__file__).resolve().parents[1] / "data" / "interventions.jsonl"
    results = simple_keyword_retrieval(query, path, topk=5)
    for r in results:
        print(r["id"], r["category"], r.get("clause", "-"))
