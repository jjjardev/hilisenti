"""
Hiligaynon text preprocessing for HiliSenti.

Shared by training and inference. The published checkpoint was trained on
normalized text, so inference must apply the same pipeline; importing this
module is what keeps the two paths consistent.
"""

import re
import unicodedata

import pandas as pd

# The 46 Hiligaynon terms added to the XLM-RoBERTa-large vocabulary during
# training, in the exact order they were added. Recovered from the published
# tokenizer (added_tokens ids 250002-250047); the embedding matrix grew from
# 250002 to 250048 rows to match. Adding them in a different order assigns
# different ids and will not reproduce the released checkpoint.
HILIGAYNON_TOKENS = [
    "bacolod", "pamangkot", "nanamian", "probinsya", "manunudlo", "nagahatag",
    "mabudlay", "kinahanglan", "lektyur", "gihapon", "kaugalingon", "paghatag",
    "magtuon", "salakyan", "siudad", "ginsiling", "pagtudlo", "ginhatag", "luyag",
    "imbestigasyon", "masadya", "nagalaum", "kapulisan", "naagyan", "himuon",
    "tubtob", "kasadya", "mabuot", "albee", "eugenio", "pumuluyo", "ginatudlo",
    "makapasar", "ginpadala", "naghatag", "ginapaabot", "maghatag", "makahalam",
    "makahalawhaw", "makawiwili", "pagtuon", "nagligad", "kabudlay",
    "makabulig", "alkalde", "shively",
]

ABBREVIATIONS = {
    r"\bwla\b": "wala", r"\bwaay\b": "wala", r"\bway\b": "wala",
    r"\bndi\b": "indi", r"\bnd\b": "indi",
    r"\bgd\b": "gid", r"\bgud\b": "gid", r"\bmn\b": "man",
    r"\bnmn\b": "naman", r"\bna lng\b": "nalang", r"\bnlng\b": "nalang",
    r"\bbl\b": "bala", r"\btni\b": "tani", r"\btne\b": "tani",
    r"\bsya\b": "siya", r"\bxa\b": "siya", r"\bxia\b": "siya",
    r"\bnya\b": "niya", r"\bnyo\b": "ninyo", r"\bcmu\b": "sa imo",
    r"\bsakn\b": "sa akon", r"\bskn\b": "sa akon", r"\bkw\b": "ikaw",
    r"\bky\b": "kay", r"\bmng\b": "mga", r"\bdpt\b": "dapat",
    r"\bsbng\b": "subong", r"\bkrn\b": "karon", r"\bhlng\b": "halong",
    r"\bamo\b": "amo", r"\bamu\b": "amo",
    r"\bpro\b": "pero", r"\bpru\b": "pero",
    r"\bkg\b": "kag", r"\bkng\b": "kon", r"\bkun\b": "kon",
}


def normalize_hiligaynon(text):
    """Lowercase and normalize one Hiligaynon sentence.

    Order matters: abbreviation expansion runs before character-run reduction
    so that expanded words are not collapsed, and whitespace is normalized last.
    """
    if pd.isna(text):
        return ""
    text = unicodedata.normalize('NFKC', str(text))
    text = text.lower()
    text = re.sub(r'\b(ha){2,}[h]*\b', 'hahaha', text)
    text = re.sub(r'\b(he){2,}[h]*\b', 'hehehe', text)
    text = re.sub(r'\b([a-z]{3,})2\b', r'\1-\1', text)
    text = re.sub(r'(\w+)-\1', r'\1 \1', text)
    for pattern, replacement in ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'([a-z])\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_frame(df, column="sentence"):
    """Apply normalize_hiligaynon to one column of a pandas DataFrame."""
    df = df.copy()
    df[column] = df[column].apply(normalize_hiligaynon)
    return df
