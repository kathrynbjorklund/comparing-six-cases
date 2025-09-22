12) Extract Impacts From Articles

import os, json, time, argparse, re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from openai import OpenAI

MODEL             = "gpt-5-nano"
INPUT_FILE        = "25_Sampled_GPT_Trial.csv"            # must contain 'full.text'
OUT_JSON_RECORDS  = "25_Sampled_GPT_Trial_Impacts_Records.json"  # flat, analysis-ready
OUT_JSON_NESTED   = "25_Sampled_GPT_Trial_Impacts_Nested.json"   # raw/nested per-article
OUT_CSV           = "25_Sampled_GPT_Trial_Impacts_Extracted_V4.csv"
CATS_JSON         = "Impact_Categories_V4.json"
BATCH_SIZE        = 5
SLEEP_SEC         = 0.2
# -------------------------------------------------------

# OG columns
ORIGINAL_COLS = [
    "feed_title","feed_link","feed_description","feed_language","url",
    "item_description","item_title","real.url","domain","publication.date",
    "full.text","Domain","NewsYN","RelevantYN"
]

# NEW columns
NEW_COLS = [
    "article_id","observed_impact_text","cause_of_impact_text",
    "location_of_impact_text","aggregated_location","time_of_impact_text",
    "category_of_impact"
]

# Allowed values for aggregated_location
AGG_LOC_ENUM = ["Municipality","Country","Continent","Globe"]

SYSTEM_DISCOVER = """You are a research scientist extracting *observed* impacts of emerging pests and pathogens (EPPs) from news articles.

Scope — extract impacts that the article attributes to EPPs (as described in the text).

Rules:
- The cause_of_impact_text is the underlying driver or mechanism described for that impact; it MUST NOT be identical to observed_impact_text. If no cause is stated, return "N/A".
- Use ONLY information present in the text. Do NOT infer beyond what the article states.
- Ignore prescriptive/normative statements (e.g., "should", "must").
- If a field is missing, return "N/A" exactly.
- Aggregated Location of Impact: choose the LEVEL CLOSEST to the location of impact as stated in the text (not the place name), using exactly one of: Municipality, Country, Continent, Globe. If unclear, use "N/A".

For each article return:
- article_id (integer; index provided)
- impacts: array of items, ONE per distinct observed impact with fields:
  * category_of_impact_free (short free-text label created from this sample; do not use external taxonomies)
  * observed_impact_text (verbatim observed impact as written in text)
  * cause_of_impact_text (verbatim cause of observed impact as written in text; "N/A" if absent)
  * location_of_impact_text (verbatim location of observed impact as written in text; "N/A" if absent)
  * aggregated_location (one of: Municipality/Country/Continent/Globe; or "N/A")
  * time_of_impact_text (verbatim time of observed impact  as written in text; "N/A" if absent)
"""

SYSTEM_APPLY_TEMPLATE = """You are a research scientist extracting *observed* impacts of emerging pests and pathogens (EPPs) from news articles.

Scope — extract impacts that the article attributes to EPPs (as described in the text).

Rules:
- The cause_of_impact_text is the underlying driver or mechanism described for that impact; it MUST NOT be identical to observed_impact_text. If no cause is stated, return "N/A".
- Use ONLY information in the text; do not infer beyond what is stated.
- Ignore prescriptive/normative statements (e.g., "should", "must").
- If a field is missing, return "N/A" exactly.
- Aggregated Location of Impact: choose the LEVEL CLOSEST to the location of impact as stated in the text (not the place name), using exactly one of: Municipality, Country, Continent, Globe. If unclear, use "N/A".

Use this fixed category set for category_of_impact (choose exactly one):
{category_list}

For each article return:
- article_id (integer; index provided)
- impacts: array of items, ONE per distinct observed impact with fields:
  * category_of_impact (choose one from the fixed list above)
  * observed_impact_text (verbatim observed impact as written in text)
  * cause_of_impact_text (verbatim cause of observed impact as written in text; "N/A" if absent)
  * location_of_impact_text (verbatim location of observed impact as written in text; "N/A" if absent)
  * aggregated_location (one of: Municipality/Country/Continent/Globe; "N/A" if absent)
  * time_of_impact_text (verbatim time of observed impact as written in text; "N/A" if absent)
"""

def normalize_string(x: Any) -> str:
    s = x if isinstance(x, str) else ""
    s = s.strip()
    return s if s else "N/A"

def _canon(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r'[\s"“”‘’\'`~.,;:!?()\[\]{}<>-]+', " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def flatten(all_rows: List[dict]) -> pd.DataFrame:
    rows = []
    for art in all_rows:
        aid = art.get("article_id")
        impacts = art.get("impacts", []) or []
        if impacts:
            for it in impacts:
                observed = normalize_string(it.get("observed_impact_text"))
                cause    = normalize_string(it.get("cause_of_impact_text"))
                if _canon(observed) and _canon(observed) == _canon(cause):
                    cause = "N/A"
                row = {
                    "article_id": aid,
                    "category_of_impact": (
                        normalize_string(it["category_of_impact"])
                        if "category_of_impact" in it
                        else normalize_string(it.get("category_of_impact_free"))
                    ),
                    "observed_impact_text": observed,
                    "cause_of_impact_text": cause,
                    "location_of_impact_text": normalize_string(it.get("location_of_impact_text")),
                    "aggregated_location": normalize_string(it.get("aggregated_location")),
                    "time_of_impact_text": normalize_string(it.get("time_of_impact_text")),
                }
                rows.append(row)
        else:
            rows.append({
                "article_id": aid,
                "category_of_impact": "N/A",
                "observed_impact_text": "N/A",
                "cause_of_impact_text": "N/A",
                "location_of_impact_text": "N/A",
                "aggregated_location": "N/A",
                "time_of_impact_text": "N/A",
            })
    return pd.DataFrame(rows)

def make_schema(batch_size: int, mode: str, enum_categories: Optional[List[str]]) -> dict:
    impact_props_common = {
        "observed_impact_text": {"type": "string"},
        "cause_of_impact_text": {"type": "string"},
        "location_of_impact_text": {"type": "string"},
        "aggregated_location": {"type": "string", "enum": AGG_LOC_ENUM + ["N/A"]},
        "time_of_impact_text": {"type": "string"},
    }
    if mode == "discover":
        impact_item = {
            "type": "object", "additionalProperties": False,
            "properties": {"category_of_impact_free": {"type": "string"}, **impact_props_common},
            "required": ["category_of_impact_free", *impact_props_common.keys()]
        }
    else:
        impact_item = {
            "type": "object", "additionalProperties": False,
            "properties": {"category_of_impact": {"type": "string", "enum": enum_categories}, **impact_props_common},
            "required": ["category_of_impact", *impact_props_common.keys()]
        }

    article_item = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "article_id": {"type": "integer"},
            "impacts": {"type": "array", "items": impact_item},
        },
        "required": ["article_id", "impacts"]
    }

    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "rows": {
                "type": "array", "minItems": batch_size, "maxItems": batch_size, "items": article_item
            }
        },
        "required": ["rows"]
    }

def build_messages(texts: List[str], start_index: int, system_prompt: str) -> list:
    numbered = "\n\n".join([f"Article {start_index + i}: {t}" for i, t in enumerate(texts, start=1)])
    user = (
        "Extract observed impacts for each article separately. "
        "Return a JSON object with key 'rows'; each element must correspond to the article at the same position "
        "and include 'article_id' and 'impacts' (array of items, one per impact).\n\n"
        + numbered
    )
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user}]

def call_batch(client: OpenAI, texts: List[str], start_index: int, mode: str, enum_categories: Optional[List[str]]) -> dict:
    schema = make_schema(len(texts), mode=mode, enum_categories=enum_categories)
    system = SYSTEM_DISCOVER if mode == "discover" else SYSTEM_APPLY_TEMPLATE.format(
        category_list=", ".join(f'"{c}"' for c in enum_categories)
    )
    msgs = build_messages(texts, start_index, system)
    resp = client.responses.create(
        model=MODEL,
        input=msgs,
        text={"format": {"type": "json_schema", "name": f"impact_{mode}", "schema": schema, "strict": True}}
    )
    data = json.loads(resp.output_text or "{}")
    rows = data.get("rows", [])
    # enforce exact count
    if len(rows) < len(texts):
        for _ in range(len(texts) - len(rows)):
            rows.append({"article_id": start_index + len(rows) + 1, "impacts": []})
    elif len(rows) > len(texts):
        rows = rows[:len(texts)]
    # fill missing article_id/arrays
    for i, r in enumerate(rows):
        if not isinstance(r.get("article_id"), int):
            r["article_id"] = start_index + i + 1
        if not isinstance(r.get("impacts"), list):
            r["impacts"] = []
    return {"rows": rows}

def dedupe_categories_from_discovery(all_rows: List[dict], min_len: int = 1) -> List[str]:
    cats = []
    for r in all_rows:
        for it in (r.get("impacts", []) or []):
            c = normalize_string(it.get("category_of_impact_free"))
            if c != "N/A" and len(c) >= min_len:
                cats.append(c)
    norm = {}
    for c in cats:
        k = c.lower()
        norm.setdefault(k, []).append(c)
    cleaned = sorted({min(v, key=len) for v in norm.values()})
    return cleaned

def _ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for c in columns:
        if c not in df.columns:
            df[c] = "N/A"
    return df[columns]

def run_mode(mode: str, input_file: str, sample_size: Optional[int], categories_path: Path,
             out_json_records: Path, out_json_nested: Path, out_csv: Path):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)

    df = pd.read_csv(input_file)
    if "full.text" not in df.columns:
        raise ValueError("Input CSV must contain a 'full.text' column.")
    texts_all = df["full.text"].astype(str).tolist()

    # Keep/align original columns (in your exact order), add if missing
    base_df = _ensure_columns(df.copy(), ORIGINAL_COLS)

    if mode == "discover":
        n = min(sample_size or 50, len(texts_all))
        texts = texts_all[:n]
        print(f"Discovering categories on first {n} articles...")
        all_rows = []
        for s in range(0, n, BATCH_SIZE):
            e = min(s + BATCH_SIZE, n)
            batch = texts[s:e]
            print(f"  batch {s+1}–{e}")
            data = call_batch(client, batch, start_index=s, mode="discover", enum_categories=None)
            all_rows.extend(data["rows"])
            time.sleep(SLEEP_SEC)

        # ---- write NESTED raw JSON (audit/provenance) ----
        with open(out_json_nested, "w", encoding="utf-8") as f:
            json.dump({"rows": all_rows}, f, ensure_ascii=False, indent=2)

        # ---- flatten to tabular (analysis) ----
        flat = flatten(all_rows)
        flat["__row_idx"] = flat["article_id"] - 1
        base_df = base_df.reset_index(drop=True)
        base_df["__row_idx"] = base_df.index
        merged = base_df.merge(flat, on="__row_idx", how="left").drop(columns=["__row_idx"])
        merged = _ensure_columns(merged, ORIGINAL_COLS + NEW_COLS)

        # ---- write RECORDS JSON + CSV (analysis-ready) ----
        merged.to_csv(out_csv, index=False)
        merged.to_json(out_json_records, orient="records", force_ascii=False, indent=2)

        # ---- discover & save categories for apply ----
        cats = dedupe_categories_from_discovery(all_rows)
        with open(categories_path, "w", encoding="utf-8") as f:
            json.dump(cats, f, ensure_ascii=False, indent=2)

        print(f"Wrote nested JSON → {out_json_nested}")
        print(f"Wrote CSV → {out_csv}")
        print(f"Wrote records JSON → {out_json_records}")
        print(f"Wrote discovered categories → {categories_path}")

    else:
        # APPLY or PIPELINE (expect categories to exist)
        if not categories_path.exists():
            raise FileNotFoundError(f"Missing categories file: {categories_path}")
        enum_categories = json.loads(Path(categories_path).read_text(encoding="utf-8"))
        print(f"Applying fixed categories to {len(texts_all)} articles...")
        all_rows = []
        for s in range(0, len(texts_all), BATCH_SIZE):
            e = min(s + BATCH_SIZE, len(texts_all))
            batch = texts_all[s:e]
            print(f"  batch {s+1}–{e}")
            data = call_batch(client, batch, start_index=s, mode="apply", enum_categories=enum_categories)
            all_rows.extend(data["rows"])
            time.sleep(SLEEP_SEC)

        # ---- write NESTED raw JSON (audit/provenance) ----
        with open(out_json_nested, "w", encoding="utf-8") as f:
            json.dump({"rows": all_rows}, f, ensure_ascii=False, indent=2)

        # ---- flatten + merge, then write records + csv ----
        flat = flatten(all_rows)
        flat["__row_idx"] = flat["article_id"] - 1
        base_df = base_df.reset_index(drop=True)
        base_df["__row_idx"] = base_df.index
        merged = base_df.merge(flat, on="__row_idx", how="left").drop(columns=["__row_idx"])
        merged = _ensure_columns(merged, ORIGINAL_COLS + NEW_COLS)

        merged.to_csv(out_csv, index=False)
        merged.to_json(out_json_records, orient="records", force_ascii=False, indent=2)

        print(f"Wrote nested JSON → {out_json_nested}")
        print(f"Wrote CSV → {out_csv}")
        print(f"Wrote records JSON → {out_json_records}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["discover","apply","pipeline"], default="discover")
    p.add_argument("--input", default=INPUT_FILE)
    p.add_argument("--out_json_records", default=OUT_JSON_RECORDS)
    p.add_argument("--out_json_nested", default=OUT_JSON_NESTED)
    p.add_argument("--out_csv", default=OUT_CSV)
    p.add_argument("--categories", default=CATS_JSON)
    p.add_argument("--sample_size", type=int, default=None)
    args = p.parse_args()

    mode = args.mode
    if mode == "pipeline":
        # 1) discover on sample → write both JSONs + CSV
        run_mode("discover", args.input, args.sample_size, Path(args.categories),
                 Path(args.out_json_records), Path(args.out_json_nested), Path(args.out_csv))
        # 2) apply to full set using discovered cats → write both JSONs + CSV with .apply suffixes
        run_mode("apply", args.input, None, Path(args.categories),
                 Path(args.out_json_records).with_suffix(".apply.json"),
                 Path(args.out_json_nested).with_suffix(".apply.json"),
                 Path(args.out_csv).with_suffix(".apply.csv"))
    else:
        run_mode(mode, args.input, args.sample_size, Path(args.categories),
                 Path(args.out_json_records), Path(args.out_json_nested), Path(args.out_csv))

if __name__ == "__main__":
    main()
