"""
config.py
Korean Fama-French 3-Factor Model v2 analysis configuration
"""

import os
import sys

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'charts'), exist_ok=True)

# ============================================================
# Parent project import path (fama-ff3-/regression_engine.py)
# ============================================================
PARENT_DIR = os.path.join(os.path.dirname(BASE_DIR), 'fama-ff3-')
if os.path.exists(PARENT_DIR) and PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# ============================================================
# 분석 파라미터
# ============================================================
START_DATE = '2000-01'
END_DATE = '2026-05'

# ============================================================
# FnGuide suffix patterns
# ============================================================
PREFERRED_SUFFIXES = ['5', '6', '7']

# ============================================================
# 금융업识别 패턴
# ============================================================
FINANCIAL_PATTERNS = ['은행', '증권', '보험', '카드', '캐피탈', '저축', '금융', '신탁', '손해보험', '생명보험']