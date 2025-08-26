from .logger import logger
import json
import re

_SG_VARIANTS = ["singapore", "singaporean", "s'pore"]

_SG_REGEX = re.compile(
    r"\b(?:" + "|".join(re.escape(v) for v in _SG_VARIANTS + ["sg"]) + r")\b",
    flags=re.IGNORECASE,
)

def mentions_singapore(text: str, fuzzy_threshold: int = 75) -> bool:
    if not text:
        return False
    if _SG_REGEX.search(text):
        return True
    normalized_variants = {normalize_text(v) for v in _SG_VARIANTS}
    words = set(re.findall(r"[A-Za-z']+", text))
    for word in words:
        word_norm = normalize_text(word)
        for var_norm in normalized_variants:
            if fuzz.ratio(word_norm, var_norm) >= fuzzy_threshold:
                return True
    return False


def normalize_text(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', s.lower())