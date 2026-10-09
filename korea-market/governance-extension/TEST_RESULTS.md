# 실제 실행 검증

2026-10-09 UTC (한국시간 2026-10-10에 마무리). Python 3.12, requirements.txt의 설치 버전.

```sh
python -m pytest -q tests
```

**53 passed, 0 failed, 0 skipped, 7.19s** (최종 보고서 코드 작성 전 핵심 코드 검증). 합성 자료는 테스트 안의 SYNTHETIC_TEST_ONLY fixture이며 실증 결과 파일에는 사용하지 않는다. 공개 수치 재계산과 전체 최종 재실행 결과는 아래 추가 기록한다.

- 기존 33개 테스트와 실제 공개일/거래달력, 지연 구기간 정정, 미래 G/BM 불변, 가중치합, 결측 터미널 수익률, 중위수 동점, 전체 모집단 상위5 제외, 연속월 최소 표본, 재실행 해시 검사.
- API HTTP/content-type/status 실패 처리, 키 echo 저장 방지, 예외 비밀정보 미노출, 재무 금액 100배 오류와 비KRW 거부.
- 동일 NI 두 행을 검증한 중복 제거와 서로 다른 금액/결측 충돌 거부.
- 원본 120개 파일 SHA256 모두 일치; 실제 결합 38,742건의 G·재무 공개일 누수 0.
- `python -m src.cli verify-public`: PASS, N=24, 평균/알파/HAC CI/성과 및 차이 항등식 일치, 원자료·키 불필요.
- `python -m compileall -q src`: exit 0.

수정 과정에서 재무 증빙 테스트 1개가 간단한 fixture의 `dart_be_eok` 열 부재로 실패했다(52 passed/1 failed). 정규화값이 없을 때 이미 검증한 BE를 사용하도록 호환 처리하고 재실행해 53개 모두 통과했다. 실패를 숨기거나 실제 단위 오류를 허용하지 않았다.

자동 테스트 통과가 생존편향·전체시장 대표성·상폐 대가·업종·독립 OOS를 입증하지는 않는다. 해당 판정은 bias_audit.csv의 UNVERIFIED로 유지한다.

## 최종 검증

`python -m src.cli test`: **54 passed, 0 failed, 0 skipped, 7.49s**. 공개 실제 결과를 인증키 없이 재계산하는 통합 검증 1개 추가. `python -m compileall -q src`: exit 0. PDF 시각 QA: 연구노트 10쪽, 요약 2쪽 모두 통과(PDF_QA.md).
