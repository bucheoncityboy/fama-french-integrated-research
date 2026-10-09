import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def stock_code(value):
    text = str(value).strip()
    if not re.fullmatch(r"A?[0-9A-Z]{6}", text):
        raise ValueError("stock_code must be six alphanumeric characters, optionally prefixed by A")
    return text[1:] if len(text) == 7 else text


def month_start(series):
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def snapshot(root):
    """Snapshot every source file, excluding only git metadata."""
    root = Path(root).resolve()
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*"))
            if p.is_file() and ".git" not in p.relative_to(root).parts}


def integrity(before, after):
    return pd.DataFrame([{"path": p, "sha256_before": before.get(p),
                          "sha256_after": after.get(p), "unchanged": before.get(p) == after.get(p)}
                         for p in sorted(set(before) | set(after))])


def lag_market_values(panel):
    p = panel.sort_values(["code", "date"]).copy()
    p["date"] = month_start(p["date"])
    prior_date = p.groupby("code")["date"].shift()
    contiguous = prior_date.eq(p["date"] - pd.offsets.MonthBegin(1))
    p["lag_me"] = p.groupby("code")["me"].shift().where(contiguous)
    p["lag_bm"] = p.groupby("code")["bm"].shift().where(contiguous)
    p["formation_date"] = p["date"] - pd.Timedelta(days=1)
    return p
