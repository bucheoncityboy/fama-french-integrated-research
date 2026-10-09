# 출처와 연구 프레임의 역할

본 연구는 미래에셋 Quant/ESG 보고서의 연구 질문·해석 프레임을 참고한 개인 연구다. 증권사의 공식 산출물이 아니며 상용 점수·백테스트·COE 추정치를 본 연구 결과로 전용하지 않았다.

| 참고 자료 | 확인 범위와 사용 역할 |
|---|---|
| 2025.11 「거버넌스 퀀트투자 방법론」 | 사용자 제공 MiraeAsset Master MD의 PDF 요약. 거버넌스 위험 - 요구수익률 - PBR 해석과 PBR x G 설계 참고. 상용 MSCI와 본 연구 사외이사 비율의 차이를 인정 |
| 2025.12 「ESG 레이팅 Preview」 | 같은 사용자 MD. 점수 원천/집단/결측/규모 차이 및 비단조성 질문 참고. 해당 점수를 재현했다고 주장하지 않음 |
| 2026.05 「멀티플 탈출」 | 같은 사용자 MD. 가치 함정과 비재무 정보의 조건부 설명력 질문 참고. 외부 성과/정책 효과를 본 연구 수치로 쓰지 않음 |
| 기존 한국 FF3 보고서·코드 | 기준 커밋의 보고서 관련 페이지 3-5/15와 실제 6포트폴리오/패널 감사. 가격수익률/배당 제외 확인; 수정하지 않음 |
| 기존 RESEARCH_NOTE.md | references/INTERIM_RESEARCH_NOTE.md에 보존. 과거 fallback 상태이며 현재 실제 결과는 output/RESEARCH_NOTE.md |

## 공식 데이터와 증빙

- [OpenDART 사외이사 API](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS002&apiId=2020012): 이사 수, 사업연도, 경제기간, 접수번호.
- [OpenDART 공시검색](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019001): 역사적 종목-법인 쌍과 실제 접수·공개일.
- [OpenDART 주요 재무계정](https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS003&apiId=2019016): BE 금액·범위·통화 및 ROE/레버리지 통제.
- [OpenDART 이용약관](https://opendart.fss.or.kr/intro/terms.do): raw 전체 응답의 재배포 허락을 가정하지 않고 집계 결과·hash·공식 링크만 공개.
- [KRX KIND 현재 상장기업 목록](https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13): 2026-10-09 수집 현재 시장/업종 범위 진단. 과거 PIT 분류로 사용하지 않음.
- [2009년 상법 시행령 개정 원문](https://www.law.go.kr/LSW/lsSideInfoP.do?ancNo=21288&ancYd=20090203&chrClsCd=010202&lsNm=%EC%83%81%EB%B2%95+%EC%8B%9C%ED%96%89%EB%A0%B9&urlMode=lsRvsDocInfoR): 자산 규모와 사외이사 법정 기준의 혼동 검토. 표본 각 회사/연도의 법적 예외 및 별도 자산기준 미검증.
- [XKRX 달력 구현](https://github.com/gerrymanoim/exchange_calendars/blob/master/exchange_calendars/exchange_calendar_xkrx.py): 설치 4.13.2, 거래 세션 월말.
- [DL건설 주식교환 공시](https://kind.krx.co.kr/external/2023/12/06/000361/20231206001130/10601.htm), [롯데푸드 합병 평가 공시](https://kind.krx.co.kr/external/2022/03/23/000780/20220323003339/11344.htm): 티커 종료가격과 총 상폐 대가의 차이. 해소된 터미널 수익률이라고 표시하지 않음.
- NAVER 월봉 5종목 독립 부분 감사: URL·수집 SHA256은 output/independent_price_source_audit.csv. zero-volume 봉을 실제 0% 수익률로 간주하지 않음.

모든 실증 수치는 본 작업의 실제 G와 원본 수익률에서 계산했다. 외부 보고서 정량 성과는 최종 보고서에 복제하지 않았다. 공개 월별 결과로 제3자가 통계치를 감사할 수 있으나, 신호 수준 재현은 API 캐시·허가된 원본 입력이 필요하다.
