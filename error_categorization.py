# error_categorization.py
import re
import nltk
import difflib

# Download punkt once if missing (quiet)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

from nltk.tokenize import word_tokenize

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")  # simple check for Arabic characters

IDIOM_PATTERNS = [
    "kick the bucket", "raining cats and dogs"  # extend with English idioms
    # you can add Arabic idioms in Arabic script as well
]

def is_arabic(text: str) -> bool:
    return bool(ARABIC_RE.search(text))

def simple_tokenize(text: str):
    """Use NLTK for Latin scripts; fallback to whitespace for Arabic."""
    try:
        if is_arabic(text):
            # Arabic tokenization: simple whitespace split (lightweight)
            return text.split()
        else:
            return word_tokenize(text)
    except Exception:
        return text.split()

def detect_errors(student_text: str, reference_text: str):
    """
    Basic (rule-based) error detectors. Returns a list of short labels.
    This is intentionally simple and meant as a first pass.
    """
    errors = set()
    s = (student_text or "").strip()
    r = (reference_text or "").strip()

    # If either empty
    if not s:
        errors.add("Semantic: Missing student translation")
        return sorted(errors)

    # Similarity ratio (string-level)
    ratio = difflib.SequenceMatcher(None, s, r).ratio()
    if ratio < 0.85:
        errors.add("Linguistic: Low lexical similarity")

    # Token-length checks for omission/addition
    stoks = simple_tokenize(s)
    rtoks = simple_tokenize(r)
    if len(stoks) < 0.7 * max(1, len(rtoks)):
        errors.add("Semantic: Possible omission")
    elif len(stoks) > 1.3 * max(1, len(rtoks)):
        errors.add("Semantic: Possible addition")

    # Naive grammar/word-choice detector: presence of many rare/long tokens
    long_tokens = [w for w in stoks if len(w) > 15]
    if len(long_tokens) >= 1:
        errors.add("Stylistic: Awkward/long tokens")

    # Idiom check (very simple)
    low_s = s.lower()
    low_r = r.lower()
    for p in IDIOM_PATTERNS:
        if p in low_r and p not in low_s:
            errors.add("Cultural: Idiom not rendered")

    # Example: detect numeric mismatch (numbers should be preserved)
    nums_s = re.findall(r"\d+", s)
    nums_r = re.findall(r"\d+", r)
    if nums_s != nums_r:
        errors.add("Linguistic: Numeric mismatch")

    return sorted(errors)

def categorize_errors_dataframe(df, student_col='Student_Translation', reference_col='Reference_Translation'):
    """
    Input: pandas DataFrame with student and reference columns.
    Output: DataFrame with new column 'Errors_Detected' (semicolon-separated labels).
    """
    import pandas as pd

    errors_out = []
    for _, row in df.iterrows():
        st = str(row.get(student_col, "") or "")
        ref = str(row.get(reference_col, "") or "")
        errs = detect_errors(st, ref)
        errors_out.append("; ".join(errs) if errs else "")
    df['Errors_Detected'] = errors_out
    return df
