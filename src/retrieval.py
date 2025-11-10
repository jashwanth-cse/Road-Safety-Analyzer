import json, os, re
from pathlib import Path
from collections import Counter

def load_jsonl(p):
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for l in f:
            out.append(json.loads(l))
    return out

def simple_keyword_retrieval(query, jsonl_path, topk=5):
    q = query.lower()
    results = []
    docs = load_jsonl(jsonl_path)
    for r in docs:
        score = 0
        # exact keyword matches in 'problem' and 'data'
        for field in ("problem","data","category","type","raw"):
            txt = str(r.get(field,"")).lower()
            for tok in q.split():
                if tok in txt:
                    score += 2
        # phrase checks for common phrases
        for phrase in ["pedestrian","speed","curve","intersection","school","bus stop","pothole","lighting","guardrail","cycle"]:
            if phrase in q and phrase in (r.get("problem","")+" "+r.get("data","")).lower():
                score += 5
        # small bonus for category match
        if r.get("category","").lower() in q:
            score += 3
        results.append((score, r))
    results.sort(key=lambda x: x[0], reverse=True)
    # filter out zero-score
    filtered = [r for s,r in results if s>0]
    if not filtered:
        # fallback: top by score even if zero
        filtered = [r for s,r in results[:topk]]
    return filtered[:topk]

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv)>1 else "busy intersection near school with no crossing and poor lighting"
    path = Path(__file__).resolve().parents[1] / "data" / "interventions.jsonl"
    results = simple_keyword_retrieval(q, path, topk=5)
    for r in results:
        print(r["id"], r["category"], r["clause"])
