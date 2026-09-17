# fama-french-1993-replication — 미국 주식·채권 5요인 모형의 FF(1993) 재현

## 📄 연구 보고서

- [**Fama-French (1993) 재현 발표자료 — HY-FIN 리서치 세션 5조 (PDF)**](./Fama-French%20(1993)%20재현%20발표자료%20(HY-FIN%20리서치%20세션%205조).pdf)

> CRSP·Compustat 원데이터로 25개 Size×BE/ME 포트폴리오와 SMB/HML을 직접 구성해 Fama-French(1993)의 다요인 모형을 재현한 실증 연구다. 주식 3요인에 채권 2요인(TERM·DEF)을 더한 5요인 구조를 CAPM 및 SMB+HML 모형과 대조하고, GRS 공동검정과 residual bootstrap으로 모형의 설명력을 검증한다. 표·그림은 과제 가이드 기준에 맞춰 `appendix_output/`으로 내보낸다.

## ⭐ 핵심 발견

- **규모·가치 효과 동시 관측**: SMALL-HiBM 포트폴리오가 월 **1.06%**, BIG-LoBM이 월 **0.39%** 초과수익률을 내며, HiBM>LoBM·Small>Big 구조가 5×5 그리드 전체에서 일관되게 나타난다.
- **양(+)의 주식 요인 프리미엄**: Mkt-RF **0.42%/월**, SMB **0.29%/월**, HML **0.41%/월**로 세 요인 모두 양(+)의 평균을 가진다. HML t값 3.15로 가장 유의하다.
- **3요인 모형이 CAPM을 압도**: 평균 R²가 CAPM **0.7524** → FF3F **0.8992**로 +0.1468 상승한다. 시장요인만으로는 잡지 못하는 규모·가치 변동을 SMB/HML이 흡수한 결과다.
- **E/P·D/P 정렬에서도 우월성 반복**: 25포트폴리오 밖의 추가 정렬에서도 모든 구간에서 FF3F R²가 CAPM을 상회한다(예: EP<=0 **0.6593→0.8341**).
- **GRS 검정으로 모형 선택**: stock mean \|alpha\|가 FF3F에서 **0.1240**으로 가장 작고, 공동검정 bootstrap p 역시 세 모형 중 가장 낮다. 채권 2요인 모형도 5% 수준에서 기각되지 않는다.

## 모형 설계

### 왜 25개 Size×BE/ME 포트폴리오인가

Fama-French(1993)는 주식 수익률의 횡단면이 규모(Size)와 장부가/시가 비율(BE/ME) 두 특성으로 체계적으로 갈린다는 사실에서 출발한다. 매년 6월 NYSE 시가총액 중위값으로 Small/Big을, BE/ME 5분위로 다섯 단계를 나눠 25개 포트폴리오를 만들고, 그 초과수익률을 요인 모형으로 설명하는 것이 이 재현의 뼈대다.

### 채권 요인은 수익률 기반이 아니라 프록시다

TERM·DEF는 FF(1993)처럼 실제 채권 포트폴리오 수익률로 만든 값이 아니라 FRED 장단기 국채·회사채 수익률에서 구성한 프록시다. 채권 포트폴리오 역시 같은 수익률 변화에서 파생되므로 회귀의 R²가 기계적으로 높게 나올 수 있다. 이 한계는 `data_sources.md`에 정리했고, 주식 표(table 4~9c)는 원문과 동일하게 해석할 수 있다.

## 검증 설계

무엇을, 어떤 기준으로 검증하는지 한눈에 보여준다.

| 검증 | 내용 | 핵심 결과 |
|---|---|---|
| SMB/HML 베타 구조 | 25개 포트폴리오 계수 패턴이 의도한 노출을 따르는가 | 소형주 SMB 베타 1.0 이상, 고BM HML 베타 단조 증가 |
| 모형 R² 비교 | CAPM·SMB+HML·FF3F 평균 R² | 0.7524 / 0.3488 / **0.8992** |
| alpha 크기 | 모형별 mean \|alpha\| | CAPM 0.2680, FF3F **0.1240** |
| GRS 공동검정 | 32개 포트폴리오 alpha가 공동으로 0인가 | F=59.31, bootstrap p **0.010** |
| Bootstrap 강건성 | 제한모형 잔차 복원추출(B=999) | CAPM 0.651 → FF3F 0.010으로 차별화 |
| 추가 정렬 | E/P·D/P 12개 포트폴리오 | 전 구간 FF3F > CAPM |

## 방법론

원문의 절차를 최대한 따르되, 과제 가이드의 제출 기준에 맞게 두 갈래로 구현했다.

### 2.1 데이터 기반 포트폴리오·요인 구성

**원데이터 → 정제 → 25개 Size×BE/ME 포트폴리오 구성 → 요인 계산**으로 이어진다. 주식 측은 CRSP raw 데이터와 Compustat BE를 쓴다. `compustat_portfolio_builder.py`가 이 파이프라인의 중심이다.

- Compustat BE(pre-computed book equity)와 CRSP의 `PRC·RET·SHROUT·PERMNO·PERMCO`를 읽고, `gvkey → PERMCO → PERMNO` 연결로 두 데이터를 묶는다.
- 매년 6월 NYSE breakpoint로 25개 Size×BE/ME 포트폴리오와 6개 2×3 포트폴리오를 구성하고, 후자에서 SMB/HML을 직접 재계산한다.

한 가지 데이터 제약이 있다. FF(1993) 방법론은 매년 7월 재구성 시 **작년 12월 BE와 당해 6월 ME**를 짝지으므로, 1963-07 첫 포트폴리오에는 1962-12 BE가 필요하다. 그런데 `compustat_be.csv`는 1964년부터 시작해 1962년 BE가 존재하지 않는다. 이 문제는 `_hybridize_ken_french_data()`의 **하이브리드 방식**으로 푼다.

| 구간 | 기간 | 데이터 출처 | 설명 |
|---|---|---|---|
| **Seed (보존)** | 1963-07 ~ 1964-06 | Ken French Data Library 원본 | 1962년 BE 부재로 자체 구성이 불가능한 첫 12개월. French 원본 25/6 포트폴리오와 요인 값을 그대로 쓴다. |
| **Replacement (대체)** | 1964-07 ~ 1991-12 | 자체 구성(CRSP + Compustat BE) | 1964년 BE부터 사용 가능하므로 직접 구성한 값이 French 원본을 월별로 덮어쓴다. 자체 구성에 실패한 월은 French 값이 fallback으로 남는다. |

SMB/HML 요인도 같은 구조로 생성한다. `01_section2_factors.py`가 6개 2×3 포트폴리오로 SMB/HML을 재계산하고, `compustat_portfolio_builder.py`가 1964-07부터 그 값을 French 원본 위에 덮어씌운다. 결과적으로 **1963-07~1991-12 전 기간에 빈 구간 없는 연속 factor 시계열**이 확보된다.

### 2.2 요인·표본

- **포트폴리오**: 25개 Size×BE/ME, 가치가중, 매년 6월 재구성.
- **주식 요인**: Mkt-RF·SMB·HML(직접 계산), RF(미국 1개월 T-Bill).
- **채권 요인**: TERM(GS10−TB3MS)·DEF(BAA−AAA), decimal → % 환산.
- **채권 포트폴리오**: FRED 수익률 기반 7개 프록시(만기·신용등급별), decimal → % 환산.
- **표본**: CRSP·Compustat 1963-07~1991-12, 342개월(채권 프록시 340~341개월).

### 2.3 제출용 표·그림

`appendix_output/`가 과제 제출의 표준 출력이다. `build_submission.py`가 제출 스크립트를 순서대로 돌리고, `10_appendix_table_exports.py`가 제외 지시를 파일 수준에서 반영해 최종 표를 정리한다. Table 9c에는 F-분포 p-value와 함께 **restricted model(H0: alpha=0) 잔차를 시간 축에서 복원추출해 횡단면 상관을 보존하는 residual bootstrap(B=999)** 확률 수준을 더한다. 어떤 표를 남기고 지우는지는 `TABLE_COVERAGE_AUDIT.md`에 전수 기록돼 있다.

## 재생성 방법

### 사전 준비

```bash
pip install -r requirements.txt
```

`data/`에는 CRSP·Compustat·Ken French 원데이터가 필요하다. 상세 포맷과 FRED 시리즈는 `data_sources.md`를 따른다. (이 저장소는 라이선스상 원데이터를 포함하지 않는다.)

### 제출 산출물 생성

`build_submission.py`는 루트에 있고, 실제 분석 스크립트는 `src/`에서 실행된다. 경로(`config.BASE_DIR`)는 항상 루트를 가리키므로 산출물은 `appendix_output/`에 모인다.

```bash
python build_submission.py   # src/ 아래 스크립트를 순서대로 실행
python -m pytest -q          # tests/ (src/ 경로 자동 추가)
```

단일 스크립트만 다시 돌리려면:

```bash
python src/10_appendix_table_exports.py   # 제출용 표만 재생성
```

최근 검증 결과:

- `274 passed, 1 skipped`

## 저장소 구조

```
build_submission.py        # 제출 패키지 공식 빌드 진입점 (루트 유지)
appendix_output/           # 제출용 표·그림 (표준 출력, 루트 고정)
tests/                     # 파이프라인·수치 검증 pytest 스위트
src/
  config.py                    # 경로·분석 구간(1963-07~1991-12)·URL 설정 (BASE_DIR=루트)
  data_loader.py               # 요인·주식·채권 포트폴리오 공통 로더
  regression_engine.py         # OLS·공분산·GRS 공통 엔진
  compustat_portfolio_builder.py  # CRSP+Compustat → 25개 포트폴리오·SMB/HML (하이브리드)
  fred_bond_fetcher.py         # FRED 수익률 → TERM/DEF·채권 포트폴리오
  ken_french_parser.py         # Ken French 원본 파싱
  download_data.py             # 원데이터 다운로드
  01_section2_factors.py       # 주식 요인(SMB/HML) 재계산
  01b_section2_bond_factors.py # 채권 요인(TERM/DEF) 구성·병합
  02_section2_portfolios.py    # 25개 Size×BE/ME 포트폴리오 구성
  02b_section2_bond_portfolios.py # 7개 채권 포트폴리오 프록시 구성
  03_section3_statistics.py    # 요인·포트폴리오 기초통계·상관행렬
  04_section4_regressions.py   # CAPM·SMB+HML·FF3F 회귀
  04b_section4_five_factor.py  # 5요인(주식 3+채권 2) 회귀
  05_section5_grs_test.py      # GRS(1989) 공동 alpha 검정
  05b_section5_intercepts.py   # 개별 alpha 분석(160행)
  06_section6_conclusions.py   # 결론 수치 집계
  07_section0_descriptive_stats.py # 표본 기술통계 표
  08_section8a_rmo_regressions.py  # RMO 보조 회귀(주식 외 수익원)
  09_section11_ep_dp_portfolios.py # E/P·D/P 정렬 CAPM vs FF3F
  10_appendix_table_exports.py # 제출용 표 최종 편집·bootstrap
  11_submission_visualizations.py  # 제출용 Figure 1~5
```

## 참고문헌

- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.
- Gibbons, M. R., Ross, S. A., & Shanken, J. (1989). A test of the efficiency of a given portfolio. *Econometrica*, 57(5), 1121-1152.
