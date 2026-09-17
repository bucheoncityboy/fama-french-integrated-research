# fama-french-korea-factor — 한국 시장 Fama-French 3-Factor 실증

## 📄 연구 보고서

- [**한국 시장 Fama-French 3-Factor 실증 연구보고서 (PDF)**](./한국%20시장%20Fama-French%203-Factor%20실증%20연구보고서.pdf)

> FnGuide 1,054종목·311개월 패널을 가공해 25개 Size×BM 포트폴리오를 만들고, 3-Factor 모델의 적합성과 팩터 프리미엄을 미국(CRSP)과 비교 검증한 실증 연구다.

## ⭐ 핵심 발견

- **HML(가치 팩터) 월 0.83%, t=3.89**, 한국 시장에서 가장 강한 프리미엄으로, 미국(0.41%)의 약 2배 수준
- **SMB 부호 반전: 월 -0.67%(t=-2.57)**, 미국(+0.29%)과 반대로, 소형주가 대형주를 이기지 못하는 구조를 보인다
- **GRS 검정 F=1.697 (p=0.023)**, 25개 포트폴리오에 대한 3-Factor 모델의 설명력과 한계를 통계적으로 보여준다
- **FnGuide 원데이터 정제부터 GRS 검정까지 전체 파이프라인을 재현**, 1,054개 종목, 2000-07~2026-05, 311개월

## 주요 시각화

**미국 vs 한국 팩터 프리미엄 비교**, SMB 부호 반전이 한눈에 드러난다.

![미국 vs 한국 팩터 프리미엄](output/charts/08_us_vs_korea_comparison.png)

**25개 Size×BM 포트폴리오 평균 초과수익률 히트맵**, 고BM으로 갈수록 수익률이 체계적으로 높아지는 가치 효과를 보여준다.

![25 포트폴리오 히트맵](output/charts/03_heatmap_25_portfolios.png)

**팩터 누적수익률 (2000-2026)**, HML만 꾸준히 우상향하고, SMB는 금융위기 이후 하락해 음의 누적수익률로 수렴한다.

![누적수익률](output/charts/01_cumulative_returns.png)

## 팩터 통계 (2000-07 ~ 2026-05)

| 팩터 | 월평균 | 표준편차 | t | Sharpe | 미국(CRSP) |
|------|-------|---------|---|--------|-----------|
| Mkt-RF | +0.71% | 6.64% | 1.90 | 0.108 | +0.42% |
| SMB | **-0.67%** | 4.63% | **-2.57** | -0.146 | **+0.29%** |
| HML | **+0.83%** | 3.74% | **3.89** | 0.221 | +0.41% |

25개 포트폴리오 3-Factor 회귀 결과는 다음과 같다.

| 지표 | 결과 |
|---|---|
| 평균 R² | 0.72 (범위 0.40~0.88) |
| 시장 베타 | 평균 0.99, 범위 0.88~1.08로 모두 1 부근에 분포 |
| SMB 베타 | 소형주 포트폴리오는 1.0 이상, 대형주는 0 부근 또는 음수 |
| HML 베타 | 저BM은 음수(성장주 성향), 고BM은 0.59~1.13으로 단조 증가 |

25개 포트폴리오가 의도한 규모·가치 노출을 제대로 포착했음을 확인할 수 있다.

## SMB 부호 반전, 왜 일어났나

미국에서 규모 효과(Small Minus Big)는 양(+)이지만 한국은 반대다. 분석 기간 내내 대형주가 소형주보다 나은 성과를 냈다.

- **시장 집중도**: 삼성전자·SK하이닉스·현대차 등 소수 대형 제조업체가 시가총액의 대부분을 차지하고, 외국인·연기금 자금이 이들로 몰린다.
- **KOSDAQ의 성장주 편중**: 소형주 상당수가 기술·바이오·게임 성장주다. 전통적인 가치형 소형주 프리미엄이 희석되거나 역전됐다.
- **Mkt-RF와의 음의 상관(-0.378)**: 시장이 오를 때면 늘 대형주가 앞서는 구조가 팩터 수익률에 그대로 묻어난다.

## 방법론

Fama-French(1993)의 원론을 최대한 따르면서 한국 시장에 맞게 다시 구축했다.

```
FnGuide raw 패널 → build_panel_data.py ETL → qa_check(결측·품질 QA)
→ 우선주(코드 5·6·7)·금융업(명칭) 필터 → 가치가중 25개 Size×BM 포트폴리오
→ 3-Factor 회귀 → GRS 검정 → 미국(CRSP) 비교
```

- **포트폴리오**: 매년 6월 말 시가총액 중위값으로 Small/Big, BM 30/70 백분위로 Low/Med/High를 나눠 6개 기초 포트폴리오를 만들고, 5×5 분류는 별도 25개 포트폴리오로 구성. 보유 기간 12개월, 시가총액 가중.
- **팩터**: SMB=(S/L+S/M+S/H)/3-(B/L+B/M+B/H)/3, HML=(S/H+B/H)/2-(S/L+B/L)/2. 무위험수익률은 한국은행 기준금리 기반.
- **표본**: KOSPI+KOSDAQ 1,054종목, 우선주·금융업 제외. 분석 기간은 IMF 외환위기 이후 한국 자본시장의 구조 변화(IT 버블, 글로벌 금융위기, COVID-19, 금리 인상기)를 모두 포함한다.

## 저장소 구조

```
build_panel_data.py       # FnGuide raw 데이터 → 패널 ETL
qa_check.py               # 결측치 보정·데이터 품질 QA
config.py                 # 경로·표본 기준 설정
scripts/
  build_25_portfolios.py  # 25개 Size×BM 포트폴리오 구성
  construct_6_portfolios.py
  calc_stock_returns.py   # 수정주가 기반 수익률 계산
  run_regression_korea.py # 3-Factor 회귀 + GRS 검정
  generate_charts.py      # 시각화 생성
output/charts/            # 분석 결과물 8종
data/                     # 처리된 패널 데이터(parquet)
tests/                    # 출력 검증 테스트
```

## 참고문헌

- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.
- Gibbons, M. R., Ross, S. A., & Shanken, J. (1989). A test of the efficiency of a given portfolio. *Econometrica*, 57(5), 1121-1152.
- 조영선·김상태 (2000), 김진영 (2002) 등 국내 선행연구와 결과를 대조.
