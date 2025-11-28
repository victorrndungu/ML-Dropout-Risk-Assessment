#!/usr/bin/env python3
"""
build_features.py

Reads anonymized .txt files from ANON_DIR,
extracts structured fields & keyword flags,
computes Sentence-BERT embeddings,
and writes features_dataset.csv (+ embeddings.npy optional).

Adjust ANON_DIR to your path if needed.
"""
import os, re, json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
BASE_DIR = os.path.dirname(__file__)
ANON_DIR = os.path.join(BASE_DIR, "usable")
AUG_DIR = os.path.join(BASE_DIR, "usable_aug")  # optional augmented texts
OUT_CSV = os.path.join(ANON_DIR, "features_dataset.csv")
OUT_EMB_NPY = os.path.join(ANON_DIR, "embeddings.npy")  # optional
EMB_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, good
EMB_DIM = None  # will be inferred
# ---------- Regex / heuristics ----------
RE_AGE = re.compile(r"\bAGE\s*[:\-]?\s*(\d{1,2})\b|\b(\d{1,2})-year-old\b", re.I)
RE_CLASS = re.compile(r"\bCLASS\s*[:\-]?\s*([A-Za-z0-9\-]+)\b", re.I)
RE_HEALTH = re.compile(r"\bHEALTH\s*STATUS\s*[:\-]?\s*([A-Za-z ,\-]+)\b", re.I)
RE_LAST_SCORE = re.compile(r"(LAST\s+EXAM[S]?\s+SCORE|LAST\s+SCORE|LAST\s+EXAMS?|scored|score)\s*[:\-]?\s*(\d{1,4})\s*(?:marks?|points?)?", re.I)
RE_MEALS_NUM = re.compile(r"\b(?:meals?|meals per day|meals/day)\s*[:\-]?\s*(\d)\b", re.I)
RE_MEALS_TEXT = re.compile(r"\b(one|two|three|four|2|1|3)\s+meals?\b|\beats?\s+only\s+(one|two|three|four)\s+meals?\b|\b(one|two|three|four)\s+or\s+(one|two|three|four)\s+meals?\b|\bcomes?\s+to\s+school\s+hungry\b|\boften\s+hungry\b|\bgoes?\s+hungry\b", re.I)
# father absent patterns
RE_FATHER_ABSENT = re.compile(r"\b(father).*?(absent|unknown|doesn.?t support|missing|not.*support|not.*around)\b", re.I)
# mother employed heuristics
RE_MOTHER_EMPLOY = re.compile(r"\bmoth(er|er is|er:)?.{0,40}?(works|hawks|sells|waiter|vendor|trader|cook|cleaner|gardener|hawker|casual)\b", re.I)
# housing indicators
RE_IRON_SHEET = re.compile(r"\b(iron\s*sheet|iron sheets|iron-sheet|corrugated|corrugated iron)\b", re.I)
RE_SINGLE_ROOM = re.compile(r"\b(single room|single-room|one room|single bed room)\b", re.I)
RE_SHARED_BED = re.compile(r"\b(share(?:s)? (?:a )?bed|shared bed|share the bed|share bed)\b", re.I)
RE_NO_ELECTRIC = re.compile(r"\b(no electricity|no power|no electric|no lights|no light)\b", re.I)
# economic stress: rent arrears, landlord locks
RE_RENT_ARREARS = re.compile(r"\b(rent.*(arrear|arrears|owing|behind)|landlord.*lock|locked (the )?house)\b", re.I)
# hunger / food insecurity
RE_SLEEP_HUNGRY = re.compile(r"\b(sleep (hungry|on empty stomach|without food)|sleep hungry|sometimes sleep hungry|sometimes sleep hungry)\b", re.I)
RE_TWO_MEALS = re.compile(r"\b(two meals|2 meals|two meals a day)\b", re.I)
# sibling count: look for lines under SIBLINGS or "SIBLINGS" block
RE_SIBLINGS_BLOCK = re.compile(r"\bSIBLINGS\b(.*?)(?:CASE STUDY|CASE:|AMBITION|HOBBY|$)", re.I|re.S)
RE_SIB_NAME = re.compile(r"\b([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,})\b")

# leftover PII checks (sanity)
RE_PHONE = re.compile(r"\b(?:\+?254|0)?\s*\d{2,3}[-\s]?\d{3}[-\s]?\d{3}\b")
RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# keywords to flag (non-exhaustive)
KEYWORD_FLAGS = {
    "rent_arrears_flag": RE_RENT_ARREARS,
    "hunger_flag": re.compile(r"\b(hungry|hunger|no food|sleep hungry|sometimes sleep hungry|starv)\b", re.I),
    "father_absent_flag": re.compile(r"\b(father (absent|not.*support|doesn.?t support|not around|missing))\b", re.I),
    "mother_hawker_flag": re.compile(r"\b(hawk|hawker|hawking|sells tea|sells chapati|hawks tea)\b", re.I),
    "landlord_lock_flag": re.compile(r"\b(landlord (locked|locks)|locked the house)\b", re.I),
    "no_school_fees_flag": re.compile(r"\b(can.t pay school|can not pay school|can't pay school|fees not paid|school fees)\b", re.I),
    "works_unstable_flag": re.compile(r"\b(unstable job|casual work|temporary job|piece job|odd jobs|day labour)\b", re.I),
    "single_parent_flag": re.compile(r"\b(mother is the breadwinner|father not.*support|father absent|single mother|single parent)\b", re.I),
}

# -------------- helpers --------------
def safe_read(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def extract_structured_from_text(text):
    out = {}
    # Age extraction - handle multiple capture groups
    m = RE_AGE.search(text)
    if m:
        # Try each capture group until we find a non-None value
        age = None
        for i in range(1, m.lastindex + 1):
            if m.group(i) is not None:
                age = int(m.group(i))
                break
        out["age"] = age
    else:
        out["age"] = None
    
    # Class extraction - handle multiple capture groups
    m = RE_CLASS.search(text)
    if m:
        # Try each capture group until we find a non-None value
        class_val = None
        for i in range(1, m.lastindex + 1):
            if m.group(i) is not None:
                class_val = m.group(i).strip()
                break
        out["class"] = class_val
    else:
        out["class"] = None
    
    # Health status
    m = RE_HEALTH.search(text)
    out["health_status"] = m.group(1).strip() if m else None
    
    # Exam score extraction - handle multiple capture groups and special cases
    m = RE_LAST_SCORE.search(text)
    if m:
        # Try each capture group until we find a non-None value
        score = None
        if m.lastindex:
            for i in range(1, m.lastindex + 1):
                if m.group(i) is not None:
                    try:
                        score = int(m.group(i))
                        break
                    except ValueError:
                        # Handle special cases like "missed exams" or "performance dropped"
                        if "missed" in m.group(i).lower() or "dropped" in m.group(i).lower():
                            score = 0  # Assign 0 for missed exams or dropped performance
                            break
        out["last_exam_score"] = score
    else:
        out["last_exam_score"] = None
    
    # Meals extraction - handle multiple capture groups and special cases
    m = RE_MEALS_NUM.search(text)
    if m:
        out["meals_per_day"] = int(m.group(1))
    else:
        m2 = RE_MEALS_TEXT.search(text)
        if m2:
            # Try each capture group until we find a non-None value
            txtnum = None
            if m2.lastindex:
                for i in range(1, m2.lastindex + 1):
                    if m2.group(i) is not None:
                        txtnum = m2.group(i)
                        break
            if txtnum:
                # Handle special cases like "comes to school hungry"
                if "hungry" in txtnum.lower():
                    out["meals_per_day"] = 0  # Critical: hungry = 0 meals
                else:
                    out["meals_per_day"] = {"one":1,"two":2,"three":3,"four":4}.get(txtnum.lower(), None)
            else:
                out["meals_per_day"] = None
        else:
            # CRITICAL: If no meals pattern found, check for hunger indicators
            hunger_indicators = ["hungry", "starving", "no food", "lack of food", "food shortage"]
            if any(indicator in text.lower() for indicator in hunger_indicators):
                out["meals_per_day"] = 0  # Critical: hunger detected
            else:
                out["meals_per_day"] = None
    # CRITICAL: Add comprehensive risk flag detection for missed indicators
    out["critical_flags"] = []
    
    # Check for critical academic indicators
    academic_crisis = ["missed", "dropped", "failed", "absent", "truant", "performance dropped", "grades dropped"]
    if any(indicator in text.lower() for indicator in academic_crisis):
        out["critical_flags"].append("academic_crisis")
        # If exam score is still None, set to 0 (critical)
        if out["last_exam_score"] is None:
            out["last_exam_score"] = 0
    
    # Check for critical food insecurity
    food_crisis = ["hungry", "starving", "no food", "lack of food", "food shortage", "malnourished"]
    if any(indicator in text.lower() for indicator in food_crisis):
        out["critical_flags"].append("food_crisis")
        # If meals is still None, set to 0 (critical)
        if out["meals_per_day"] is None:
            out["meals_per_day"] = 0
    
    # Check for critical family issues
    family_crisis = ["abandoned", "left", "no support", "single parent", "orphaned", "neglected"]
    if any(indicator in text.lower() for indicator in family_crisis):
        out["critical_flags"].append("family_crisis")
    
    # Check for critical housing issues
    housing_crisis = ["iron sheets", "no electricity", "no water", "poor housing", "overcrowded"]
    if any(indicator in text.lower() for indicator in housing_crisis):
        out["critical_flags"].append("housing_crisis")
    
    # Check for critical emotional distress
    emotional_crisis = ["cries", "withdrawn", "depressed", "anxious", "emotional distress"]
    if any(indicator in text.lower() for indicator in emotional_crisis):
        out["critical_flags"].append("emotional_crisis")
    
    # Check for critical economic stress
    economic_crisis = ["struggles", "cannot afford", "no money", "poverty", "financial difficulties"]
    if any(indicator in text.lower() for indicator in economic_crisis):
        out["critical_flags"].append("economic_crisis")
    
    # Convert list to string for storage
    out["critical_flags"] = ";".join(out["critical_flags"]) if out["critical_flags"] else ""
    
    # siblings count via siblings block or count capitalized names
    sib_block = RE_SIBLINGS_BLOCK.search(text)
    if sib_block:
        names = RE_SIB_NAME.findall(sib_block.group(1))
        out["siblings_count"] = len(names)
        out["siblings_list"] = ";".join(names) if names else ""
    else:
        # fallback: look for lines like "SIBLINGS" elsewhere or count occurrences of "sister"/"brother"
        s_cnt = len(re.findall(r"\b(sister|brother)\b", text, re.I))
        out["siblings_count"] = s_cnt
        out["siblings_list"] = ""
    return out

def detect_housing_flags(text):
    return {
        "iron_sheets_flag": bool(RE_IRON_SHEET.search(text)),
        "single_room_flag": bool(RE_SINGLE_ROOM.search(text)),
        "shared_bed_flag": bool(RE_SHARED_BED.search(text)),
        "no_electric_flag": bool(RE_NO_ELECTRIC.search(text)),
    }

def detect_keywords(text):
    d = {}
    for k, patt in KEYWORD_FLAGS.items():
        d[k] = bool(patt.search(text))
    # also add siblings_count from structured extraction
    return d

def detect_leftover_pii(text):
    return {
        "leftover_phone": bool(RE_PHONE.search(text)),
        "leftover_email": bool(RE_EMAIL.search(text)),
    }

# ---------------- main ----------------
def main():
    # collect .txt files from usable/ and usable_aug/ if exists
    files = []
    if os.path.isdir(ANON_DIR):
        files += [os.path.join(ANON_DIR, f) for f in os.listdir(ANON_DIR) if f.lower().endswith(".txt")]
    if os.path.isdir(AUG_DIR):
        files += [os.path.join(AUG_DIR, f) for f in os.listdir(AUG_DIR) if f.lower().endswith(".txt")]
    files = sorted(files)
    if len(files) == 0:
        print("No .txt files found in", ANON_DIR, "or", AUG_DIR); return

    # load embedding model
    print("Loading embedding model:", EMB_MODEL_NAME)
    model = SentenceTransformer(EMB_MODEL_NAME)
    global EMB_DIM
    EMB_DIM = model.get_sentence_embedding_dimension()
    print("Embedding dim:", EMB_DIM)

    rows = []
    embeddings = []

    for fpath in tqdm(files, desc="Profiles"):
        text = safe_read(fpath)
        # basic metadata
        uid = Path(fpath).stem
        row = {"uid": uid, "source_file": Path(fpath).name, "text_len": len(text)}
        # structured fields
        s = extract_structured_from_text(text)
        row.update(s)
        # housing
        row.update(detect_housing_flags(text))
        # keywords flags
        row.update(detect_keywords(text))
        # left over PII check
        row.update(detect_leftover_pii(text))
        # simple counts
        row["sentence_count"] = max(1, len(re.split(r"[.!?]\s+", text)))
        # compute embedding
        emb = model.encode(text)
        embeddings.append(emb)
        # embedding placeholders: will expand later
        rows.append(row)

    # convert to DataFrame
    df = pd.DataFrame(rows)
    emb_arr = np.vstack(embeddings)  # shape (n, dim)
    # expand embedding columns
    emb_cols = [f"emb_{i}" for i in range(emb_arr.shape[1])]
    emb_df = pd.DataFrame(emb_arr, columns=emb_cols)
    df_out = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)

    # write CSV and embeddings separately (embeddings also embedded in CSV)
    print("Writing CSV:", OUT_CSV)
    df_out.to_csv(OUT_CSV, index=False)
    print("Writing embeddings numpy:", OUT_EMB_NPY)
    np.save(OUT_EMB_NPY, emb_arr)
    print("Done. rows:", len(df_out))

if __name__ == "__main__":
    main()