# 한국 FF3 x 공시 기반 거버넌스 실증

**EMPIRICAL_LIMITED / 탐색적, 독립 OOS 없음.** 실제 OpenDART 과거 공시를 확보하고 공개일에 맞춰 BM x G 수익률과 조건부 FF3 알파를 계산했다. 전수 주 가설은 기간 부족으로 판단 불가이며, 별도 보조 표본에서는 유의한 추가 알파의 증거가 없다.

| 구분 | 표본/결과 | 해석 |
|---|---|---|
| 실제 취득 | 연간 G-PIT 4,540관측, 결합 782기업·38,742종목-월, 일부 자료 존재 59개월 | 주로 2020-2025사업연도; 2019년 2개 예외 |
| 전수 주 분석 | 4셀 개별 유효 23개월, 최장 연속 3개월(2022.12-2023.02) | 최소 24개월 미달, H1/FF3 알파 NA |
| 고정 150단계 보조 | 2024.04-2026.03 24개월, 기간 중 결합 138기업 | 가용성 감사 뒤 별도 등록한 탐색적 표본 |
| 보조 평균 스프레드 | 월 +1.3897%, HAC p=0.1535 | 비유의 |
| 보조 FF3 알파 | 월 +0.5145%, HAC3 t=0.8809, p=0.3888, 95% CI [-0.7039, 1.7329]% | 비유의, 벤치마크·표본 조건부 |
| 편향 감사 | PASS 4, UNVERIFIED 10 / 14항목 | 생존·상폐, 선택, 업종·규제·규모 혼동 잔존 |

G는 사외이사 수 / 전체 이사 수다. 종합 ESG 등급, COE 또는 정책 인과효과를 측정하지 않았다. 가격수익률·비용 전이며, 롱숏 공매·조달·유동성은 미검증이다. LowBM 보조 셀의 결측월은 누적성과/MDD로 이어 붙이지 않는다.

[연구노트](output/RESEARCH_NOTE.md), [2쪽 요약](output/EXECUTIVE_BRIEF.md), [연구노트 PDF](output/pdf/FF3_Governance_Research_Note.pdf), [요약 PDF](output/pdf/FF3_Governance_Executive_Brief.pdf), [결과 상태](output/RESULT_STATUS.json), [테스트](TEST_RESULTS.md), [편향](output/bias_audit.csv), [전체 보조검정](output/multiple_testing.csv).

기존 미국/한국 FF3 파일은 기준 커밋 `81f3bd921ffcb4ea84c19123d798fbcb830ee9e3`에서 보존했다. 기존 방식 HML +0.8264%·SMB -0.6736%/월(311개월)과 명세 정렬 HML +1.1441%·SMB -0.8222%를 함께 저장했다. 구현 차이와 벤치마크 PIT의 조건부 한계는 보고서에 설명한다. 120개 원본 파일의 전후 SHA256은 [source_integrity.csv](output/source_integrity.csv)에 있다.

## 공개 결과 수치 감사

Python 3.12에서 확장 디렉터리를 작업 경로로 사용한다. 원본 stock-level 입력이나 키 없이 아래 명령이 실행된다.

```sh
cd korea-market/governance-extension
python -m pip install -r requirements.txt
python -m src.cli verify-public
python -m src.cli test
python -m compileall -q src
```

공개 월별 수익률·FF3로 평균, 회귀, HAC t/p/CI, 누적 진단 경로와 결측 처리를 다시 계산한다. `output/public_numeric_audit.json`이 PASS를 기록한다. 실제 성과 추정에 합성 데이터는 포함되지 않았다.

## 실제 원천 재현

원래 패널 6개 parquet, 재배포 권한, OpenDART 인증이 있어야 한다. 인증은 환경변수 `DART_API_KEY` 또는 `OPENDART_API_KEY`에서 읽는다. `python -m src.execute_acquisition`은 키가 없으면 숨김 입력을 사용하며 파일에 저장하지 않는다. 키를 소스·명령 인자·README에 넣지 않는다.

```sh
python -m src.cli baseline
python -m src.cli mapping
python -m src.cli smoke-test
python -m src.execute_acquisition
python -m src.cli build-pit
python -m src.cli analyze
python -m src.cli audit
python -m pip install -r requirements-report.txt
python -m src.cli report
python -m src.cli test
```

`execute_acquisition`은 50 -> 150 -> 전수 수집/PIT 체크포인트를 순서대로 검증한다. 실제 실행 시 각 단계 뒤 확보한 150단계 패널을 고정했다. 새 환경은 `data/private/stage150_joint_panel.parquet`도 해당 시점에 저장하도록 같은 실행기를 사용한다. 각 수집은 immutable raw body와 요청 인자별 캐시를 사용한다. API 과거 수정이나 원본 가격 변경으로 원천이 달라지면 새 실행이 기존 해시와 같다고 가정하지 않는다. 사전등록/창 동결 해시가 다르면 중단한다.

`data/raw/` 및 `data/private/`는 로컬 전용이다. 원천 API 응답·FnGuide stock-level 파생 파일·키는 게시하지 않는다. 공개 파일은 집계 수익률, 편향 표, 공식 접수번호 및 해시 메타데이터다. `data/raw_manifest.csv`는 hash/status/time 색인, `.csv.gz`는 전체 비밀정보 제거 조회 인자다. 원본에 이미 있던 parquet에 대한 새로운 이용허락을 뜻하지 않는다. [데이터 계약](data/data_dictionary.md), [사양](PROJECT_SPEC.md), [실행·수정 이력](EXECUTION_LOG.md)을 참조한다.

PDF는 reportlab로 생성한다. macOS의 설치된 AppleGothic 또는 `GOVERNANCE_KOREAN_FONT` TTF를 사용하며 폰트 파일은 게시하지 않는다. 환경에 폰트가 없으면 CID 한국어 폰트로 생성하므로 렌더링을 확인해야 한다. PDF 시각 QA는 별도 기록한다.

기존 중간 fallback 실행기는 보존하되 이번 실증 출력 디렉터리를 덮어쓸 수 없다. 현재 실행 인터페이스는 `src.cli`다. 최종 계산·문서 작성은 완료됐지만 주 가설 확인과 잔존 편향 해소를 완료한 것으로 표시하지 않는다.
