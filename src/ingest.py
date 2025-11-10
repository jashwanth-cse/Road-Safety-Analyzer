import pandas as pd
import json
from pathlib import Path

def ingest(csv_path, out_path):
    df = pd.read_csv(csv_path, encoding='latin-1')
    records = []
    for i, row in df.iterrows():
        rec = {
            "id": int(row.get("S. No.", i+1)),
            "problem": str(row.get("problem","")).strip(),
            "category": str(row.get("category","")).strip(),
            "type": str(row.get("type","")).strip(),
            "data": str(row.get("data","")).strip(),
            "code": str(row.get("code","")).strip(),
            "clause": str(row.get("clause","")).strip(),
            "raw": " ".join([str(row.get(c,"")) for c in df.columns])
        }
        records.append(rec)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(f"Wrote {len(records)} records to {out_path}")

if __name__ == "__main__":
    ingest(Path(__file__).resolve().parents[1] / "data" / "interventions.csv", Path(__file__).resolve().parents[1] / "data" / "interventions.jsonl")
