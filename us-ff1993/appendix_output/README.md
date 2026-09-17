# Appendix Output — Fama-French (1993) 재현 제출 산출물

## 이 디렉터리는 무엇인가

과제 가이드(Fama-French 1993 재현 및 정리)의 제출 기준에 맞춘 표·그림이 모인 표준 출력 디렉터리다. `python build_submission.py`가 이 디렉터리를 만들고, `10_appendix_table_exports.py`가 최종 형태를 정리한다. 포함/제외의 근거는 `TABLE_COVERAGE_AUDIT.md`에 전수 기록돼 있다.

## 포함된 산출물

### 표

| 표 | 파일 | 내용 |
|---|---|---|
| Table 1 panel 1 | `table1_panel1_firm_count_market_cap.csv` | 25개 포트폴리오 평균 종목 수 / 시가총액 |
| Table 1 panel 2 | `table1_panel2_cap_share_firm_count.csv` | 시총 비중 / 평균 종목 수 |
| Table 1 panel 3 | `table1_panel3_ep_dp_reference.md` | E/P·D/P 참조 스냅샷 (가이드 원값) |
| Table 2 panel 1 | `table2_panel1_factor_summary.csv`, `table2_panel1_correlation_matrix.csv` | 요인 평균·표준편차·t값·자기상관 / 요인 상관행렬 |
| Table 2 panels 2-3 | `table2_panel2_stock_mean_std.csv`, `table2_panel3_stock_tstats.csv` | 25개 포트폴리오 평균·표준편차 / t값 |
| Table 4 | `table4_panel1_b_t_b.csv`, `table4_panel2_r2_se.csv` | CAPM 베타·t값 / R²·잔차 표준오차 |
| Table 5 | `table5_panel1_s_t_s.csv`, `table5_panel2_h_t_h.csv`, `table5_panel3_r2_se.csv` | SMB·HML 계수 / R²·잔차 표준오차 |
| Table 6 | `table6_panel1_b_t_b.csv` 외 3종 | 3요인 베타·t값 / R²·잔차 표준오차 |
| Table 9a | `table9a_stock_alphas.csv` | 모형 (ii)·(iii)·(iv) 개별 alpha |
| Table 9c | `table9c_joint_tests.csv` | GRS F-검정 + residual bootstrap(B=999) |
| Table 11 | `table11_ep_dp_long.csv` | E/P·D/P 정렬 CAPM vs FF3F (long format) |

### 그림

| 그림 | 파일 | 내용 |
|---|---|---|
| Figure 1 | `submission_fig1_stock_mean_heatmap.png` | 25개 포트폴리오 평균 초과수익률 히트맵 |
| Figure 2 | `submission_fig2_factor_premiums.png` | 5개 요인 평균 프리미엄 |
| Figure 3 | `submission_fig3_model_r2.png` | CAPM / SMB+HML / FF3F 평균 R² |
| Figure 4 | `submission_fig4_ep_dp_r2.png` | E/P·D/P CAPM vs FF3F R² |
| Figure 5 | `submission_fig5_alpha_tests.png` | 5A mean \|alpha\|, 5B GRS F-stat |

## 제외된 산출물

- Table 3·4·5·6의 채권 블록, Table 7b, Table 8b, Table 9b, Table 10은 과제 지시에 따라 파일 수준에서 제외했다.
- 연구 과정의 중간 표(`table1_market.csv` 등)·그림(`fig1_*` 등)은 `build_submission.py`가 제거한다.

## 데이터 주의

- 주식 표는 CRSP·Compustat으로 직접 구성한 값이다.
- 채권 요인(TERM·DEF)과 채권 포트폴리오는 FRED 수익률 기반 프록시라, 채권 회귀의 R²는 기계적으로 높게 나올 수 있다. 해석 시 `data_sources.md`의 한계 고지를 함께 봐야 한다.

## 산출물 재생성

```bash
python build_submission.py   # 제출 산출물 전체 재생성
python -m pytest -q          # 파이프라인·수치 검증
```
