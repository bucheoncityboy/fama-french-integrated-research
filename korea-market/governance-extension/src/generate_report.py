from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def table(d):
    # No optional tabulate dependency.
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"]*len(cols)) + " |"]
    for row in d.itertuples(index=False, name=None):
        lines.append("| " + " | ".join("NA" if pd.isna(x) else str(x).replace("|", "/") for x in row) + " |")
    return "\n".join(lines)


def benchmark_figure(output):
    output = Path(output)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for mode, label, color in [("legacy_compatible", "Existing code", "#52677e"), ("spec_aligned", "Fiscal t-1 / current June ME", "#007d75")]:
        d = pd.read_csv(output/f"ff3_factors_{mode}.csv", parse_dates=["date"])
        for ax, factor in zip(axes, ["HML", "SMB"]):
            ax.plot(d.date, d[factor].rolling(12, min_periods=12).mean()*100, label=label, color=color, linewidth=1.6)
            ax.axhline(0, color="#bec6ca", linewidth=.7)
            ax.set_ylabel(f"{factor}: monthly mean (%)")
            ax.grid(axis="y", color="#e5e9eb", linewidth=.5)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    fig.suptitle("Korea FF3: sensitivity to annual formation inputs", x=.1, ha="left", fontweight="bold")
    fig.text(.1, .03, "12-month rolling arithmetic means · 2000-07–2026-05 · 311 months · lag ME weights\nSource: pinned original FnGuide-derived panel. Benchmark reconstruction; governance not estimated.", fontsize=8, color="#52677e")
    fig.tight_layout(rect=[0, .09, 1, .94])
    (output/"figures").mkdir(exist_ok=True)
    fig.savefig(output/"figures/ff3_formation_sensitivity.png", dpi=160, facecolor="white")
    plt.close(fig)


def generate(output, status):
    output = Path(output)
    summary = pd.read_csv(output/"ff3_summary.csv")
    numbers = summary[["specification", "factor", "mean_monthly_pct", "t_ols", "t_hac3", "n_months"]].copy()
    for c in ["mean_monthly_pct", "t_ols", "t_hac3"]:
        numbers[c] = numbers[c].map(lambda x: f"{x:.4f}")
    numbers.columns = ["기준", "팩터", "월평균(%)", "t(OLS)", "t(HAC3)", "월수"]
    stage = "공시 데이터의 실행 가능성 검증" if not status["primary_alpha_estimated"] else "거버넌스 조건부 가치 프리미엄 탐색"
    g_statement = "거버넌스 정보의 추가 설명력은 아직 추정하지 않았다" if not status["primary_alpha_estimated"] else "공통 표본의 거버넌스 알파를 제한된 탐색 분석으로 추정했다"
    g_table = table(pd.read_csv(output/"factor_alpha.csv"))
    availability = ("현재 실행에는 DART 인증키와 공식 법인코드 매핑, 검증 가능한 과거 G 관측치가 없다. 무인증 API 점검은 JSON 데이터 대신 HTML을 반환했으며 인증된 요약 API의 정상 작동을 확인한 것으로 간주하지 않는다. 공식 포털은 2026.10.08 20:00~10.11 18:00 일부 서비스 제한을 안내한다. 키 부재와 정기점검은 구별하며, 점검 종료만으로 연구가 실행 가능해진다고 가정하지 않는다."
                    if status["research_route"] == "fallback_c" else "현재 입력의 공시 관측값·날짜를 캐시된 원응답 해시와 대조했다. 관측별 검수와 제외 결과는 source_audit.csv를 확인한다. 재무공시 근거의 원자료 검수, 배당·상폐·업종·수익성 통제는 별도 검토 조건이며 CSV 형식 통과를 연구 전체의 검증 완료로 해석하지 않는다.")
    conclusion = ("현재 연구에서 G 소팅 수익률과 알파는 모두 미추정이다." if not status["primary_alpha_estimated"] else "아래 수치는 제한된 입력·표본에서 산출한 탐색 결과이며 인과 효과나 독립 알파의 외부 검증으로 주장하지 않는다.")
    note = f"""# 가치 프리미엄의 두 번째 필터: 거버넌스 정보의 증분을 검증할 수 있는가?

**{stage} · 기준일 {status['run_date'][:10]} · 독립 개인 연구**

## 요약

기존 한국 FF3의 가치 프리미엄을 재현했으며, {g_statement}. 기존 코드 기준 HML 월평균 0.8264%, SMB -0.6736%가 확인됐다. 한편 6월 장부자본과 BM 구성 시점을 문서 정의에 맞추면 HML은 1.1441%, SMB는 -0.8222%로 달라졌다. 따라서 새 공시 신호의 효과를 검정하기 전에 기초 팩터의 구성 기준과 실제 공개일을 고정해야 한다.

## 1. 전통적 가치 팩터만으로 충분한가?

높은 BM에 속한 기업의 평균수익률이 높다는 관찰은 낮은 가격의 원인을 식별하지 못한다. 본 연구의 질문은 같은 가치주 집단 안에서 공시 기반 거버넌스 정보가 후행 수익률을 추가로 구분하는지, 그 차이가 MKT·SMB·HML을 통제한 뒤에도 남는지다. 미래에셋증권의 PBR×G, VFS 및 공시 도구 연구를 참고하되, 상용 MSCI 점수와 보고서 성과를 복제하지 않는다.

## 2. 기초 팩터 재현과 구성 시점의 민감도

원본 저장소 커밋 `{status['source_commit']}`의 1,054종목 패널을 읽기 전용으로 사용했다. 표본월은 2000.07~2026.05의 311개월이며 원본 전체 패널은 1999.12부터 시작한다. 원본 코드가 참조하는 과거 경로와 누락 CSV를 별도 구현으로 대체했다.

{table(numbers)}

`legacy_compatible`은 원본 6셀 코드의 구성 기준을 보존해 README의 반올림 수치를 재현한다. `spec_aligned`는 전년 회계연도 장부자본/당해 6월 시총을 사용하며, 직전 관측이 한 달 전이 아닌 경우 가중치를 제외한다. 원본 6월 BE 패널은 t-2 값을 갖고, 6월 panel.bm은 이전 구성연도의 BM을 담고 있다. 표의 차이는 구성 규약 민감도이며 새로운 투자전략의 초과성과가 아니다. `spec_aligned`에서도 실제 재무공시 공개일은 별도 확인이 필요하다.

![FF3 구성 시점 민감도](figures/ff3_formation_sensitivity.png)

단위: 소수 수익률 0.01=1%; ME 천원·BE 억원. 배당 포함 여부는 원자료 메타데이터로 확인되지 않았고, 무위험수익률은 원본 기준금리 기반 대용치다. 가격 수익률에서 100% 초과 변동을 제외한 원본 전처리, 시장구분 추정 및 상폐 수익률 검증 미완료를 함께 기록한다.

## 3. 공시 데이터로 거버넌스를 측정할 수 있는가?

주 변수는 `독립(사외)이사 수 / 전체 이사 수`다. 공식 OpenDART 응답의 `otcmp_drctr_co`와 `drctr_co`로 직접 산출할 수 있어 선택했다. 비중은 감독 기능의 제한적인 대리지표이며, 이사의 실제 독립성·주주권 보호나 종합 ESG 등급을 뜻하지 않는다. 그룹은 'HighG/LowG'로 표현하고 좋은/나쁜 기업으로 단정하지 않는다.

공시 접수번호를 공시목록의 실제 공개일과 일치시킨 뒤, 공시일 다음 날부터 처음 도달하는 월말을 사용 가능일로 둔다. 월말 공시는 다음 달 말까지 기다린다. 이후 수익률만 결합하며 사업연도 말이나 접수번호의 앞자리만으로 공개일을 확정하지 않는다. 정정공시는 해당 정정 공개일부터 사용하고, 과거 기간 정정이 더 최근 결산기의 이사회 정보를 덮어쓰지 않게 처리한다.

{availability}

커버리지와 결측은 [data_coverage.csv](data_coverage.csv), 원본 감사는 [input_audit.csv](input_audit.csv), API 근거와 미확보 사유는 [source_audit.csv](source_audit.csv)에 기록했다. 미수집 G를 0점으로 대체하지 않는다.

## 4. 사전 지정 검정과 현재 상태

주 분석은 전월 말 BM 중위값으로 가치군을 나누고, 각 BM군 내부 G 중위값으로 4셀을 구성한다. 주 비교는 `HighBM_HighG − HighBM_LowG`이며 기본은 직전 월 시총가중이다. 동점은 임의로 종목코드 순서로 쪼개지 않는다. 셀당 5종목, 공통 연속 24개월을 최소 문턱으로 두며, 구성 종목의 수익률이 누락되면 그 셀 수익률을 추정하지 않는다.

FF3 OLS와 Newey–West HAC(3개월, 소표본 보정, t 분포)를 비교한다. 롱온리에는 RF를 한 번 차감하고 자기금융 스프레드에는 차감하지 않는다. EW/VW, 30/70% G 기준, 상위 5종목 제외를 사전 지정 민감도로 둔다. 업종·수익성·검증된 시장구분·거래비용은 자료 없이는 계산하지 않는다. 30/70%는 각 분리 기준의 민감도이며 양 끝 30%만 고르는 전략은 아니다.

현재 연구 상태는 **{status['status']} / {status['research_route']}**다. {conclusion} 합성 데이터 테스트 결과를 실증 성과로 사용하지 않는다.

{g_table}

[factor_alpha.csv](factor_alpha.csv)와 [robustness.csv](robustness.csv)에 결과 또는 NA와 사유를 기록했다.

## 5. 결론과 다음 검증

이번 실행에서 확인한 사실은 기존 FF3 프리미엄의 재현성과 구성 시점에 따른 민감도다. G가 독립 팩터인지, 가치주의 특성인지는 아직 판단할 수 없다. 후속 실행에는 회사×공시일별 이사회 관측, 법인코드 매핑, 재무공시 공개일 근거가 필요하다. 확보 후 같은 표본월·같은 가치군 안에서 검정하고, 효과가 없거나 표본이 부족하면 그 결과를 그대로 보고한다.

자료 수집→공시일 대조→단위·결측 검수→후행 수익률 결합→회귀와 민감도→근거 연결 보고서를 하나의 재실행 과정으로 구성했다. 실제 조사분석 업무에서 재현값과 미확인 사항을 함께 관리하는 데 초점을 뒀다.

## 자료와 재현성

- [기존 FF3 저장소](https://github.com/bucheoncityboy/fama-french-integrated-research) 및 위 고정 커밋.
- [OpenDART 독립(사외)이사 개발가이드](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS002&apiId=2020012), [공식 포털](https://opendart.fss.or.kr/).
- [미래에셋 연구 요약](../../../references/mirae_research_notes.md): 사용자 제공 요약을 출발점으로 사용했으며 이번 실행에서 PDF 전체를 독립 재검수하지 않았다.
- 실행 상태 [RESULT_STATUS.json](RESULT_STATUS.json), 원본 파일 전후 해시 [source_integrity.csv](source_integrity.csv). 기존 연구 파일은 수정하지 않았다.

본 연구는 미래에셋증권과 제휴하거나 해당 회사가 발간한 보고서가 아니다.
"""
    (output/"RESEARCH_NOTE.md").write_text(note)
    benchmark_figure(output)
