import io
import re
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd


def parse_corp_codes(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = ET.fromstring(z.read("CORPCODE.xml"))
    rows = [{name: node.findtext(name) for name in ["corp_code", "corp_name", "stock_code", "modify_date"]}
            for node in root.findall("list")]
    return validate_map(pd.DataFrame(rows).loc[lambda d: d.stock_code.str.fullmatch(r"[0-9A-Z]{6}", na=False)])


def validate_map(d):
    d = d.copy()
    for col, length in [("corp_code", 8), ("stock_code", 6)]:
        pattern = r"\d{8}" if col == "corp_code" else r"[0-9A-Z]{6}"
        if col not in d or not d[col].astype(str).str.fullmatch(pattern).all():
            raise ValueError(f"invalid {col} string identifier")
        if d[col].duplicated().any():
            raise ValueError(f"ambiguous duplicate {col} mapping")
    return d.reset_index(drop=True)
