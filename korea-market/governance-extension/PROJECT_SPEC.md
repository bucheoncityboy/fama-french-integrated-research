# 확장 연구 사양

대상: bucheoncityboy/fama-french-integrated-research의 신규 `korea-market/governance-extension/`만. 기존 120개 파일 보존. G가 실제 공개된 뒤의 월수익률을 BM군 내에서 비교하고 FF3 통제 알파를 양측 검정한다.

주 사양은 `config/pre_registration.yaml`, 해시는 `.lock.json`에 동결했다. 중위수 동점 Low, 전월 실제 ME VW, 셀별 5종목, 전수 4셀 공통 최장 연속 최소 24개월, 550일 신선도, HAC3. 관측월 독립 단위는 월이다. 재무 BE의 원천 금액·공개 접수일과 실제 KRX 월말 이용 가능일을 모두 요구한다.

별도 보조 150단계 누적 수집군의 두 가치 셀 가용성 설계는 `supplementary_availability_design.json`에서 성과 확인 전에 등록했다. 가용성 선택 편향을 인정하며 주 분석 기준을 대체하지 않는다. 단순 과거 가격수익률, 조건부 벤치마크, 비용 전 결과이고 독립 OOS는 없다.

실제 확보 관측, 결합 관측, 개별 유효월, 연속 공통월을 구분한다. 미추정 수치는 null/NA와 이유를 남긴다. 전체 실행은 코드·실제 공시·공개 결과표·자동 테스트·편향 감사·보고서로 연결한다. 통계적 비유의는 효과가 0이라는 증명이 아니다. 경제적 COE 경로는 hypothesis이며 본 연구에서 COE나 인과효과를 직접 추정하지 않는다.

G 취득 전 사전등록 commit e0b5ce6, smoke 구성 수정 0ada647, 성과 확인 전 코드 4b754c3faaa28ef24a20b03caab64ede9b1cca76. 로컬 이력과 이후 자료 품질 수정은 얇은 Git bundle과 EXECUTION_LOG로 보존한다. 원천 개정 때문에 제3자의 최신 API 수집이 과거 immutable cache와 동일하다고 보장하지 않는다.
