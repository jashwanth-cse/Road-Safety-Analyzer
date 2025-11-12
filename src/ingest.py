import pandas as pd
import json
from pathlib import Path
import math

def safe_str(x):
    """Safely convert to string without NaN or None."""
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()

def ingest(csv_path, out_path):
    # Read with UTF-8 fallback
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")

    records = []
    for i, row in df.iterrows():
        # safer ID handling
        try:
            sid = int(row.get("S. No.", i + 1))
        except Exception:
            sid = i + 1

        rec = {
            "id": sid,
            "problem": safe_str(row.get("problem", "")),
            "category": safe_str(row.get("category", "")),
            "type": safe_str(row.get("type", "")),
            "data": safe_str(row.get("data", "")),
            "code": safe_str(row.get("code", "")),
            "clause": safe_str(row.get("clause", "")),
            "raw": " ".join([safe_str(row.get(c, "")) for c in df.columns])
        }
        records.append(rec)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(records)} records to {out_path}")

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    ingest(
        ROOT / "data" / "interventions.csv",
        ROOT / "data" / "interventions.jsonl"
    )
