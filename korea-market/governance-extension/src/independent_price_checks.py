"""Cross-source diagnostics only; never overwrite unknown corporate-action returns."""
import hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from .acquire import ROOT,SOURCE


def audit():
    prices=pd.read_parquet(SOURCE/"data/market_data_long.parquet")
    prices=prices.loc[prices.item_code.eq("S410000700"),["code","date","value"]].copy()
    prices["stock_code"]=prices.code.str.removeprefix("A")
    events=[];summaries=[]
    for file in (ROOT/"data/private").glob("naver_*.xml"):
        code=file.stem.removeprefix("naver_")
        body=file.read_bytes()
        tree=ET.fromstring(body.decode("euc-kr"))
        bars=[]
        for item in tree.iter("item"):
            values=item.attrib.get("data","").split("|")
            if len(values)==6:
                bars.append({"date":pd.Timestamp(values[0]).to_period("M").to_timestamp(),"close":float(values[4]),
                            "volume":float(values[5]),"bar_date":values[0]})
        d=pd.DataFrame(bars).sort_values("date")
        if d.empty:continue
        d["naver_return"]=d.close.pct_change(fill_method=None)
        other=prices.loc[prices.stock_code.eq(code),["date","value"]].sort_values("date")
        other["source_return"]=other.value.pct_change(fill_method=None)
        merged=d.merge(other,on="date",how="left")
        matched=merged.close.notna()&merged.value.notna()
        # Backward adjusted levels can differ; compare decimal returns separately.
        same=np.isclose(merged.loc[matched,"close"],merged.loc[matched,"value"],rtol=.0001,atol=.01)
        summaries.append({"stock_code":code,"independent_month_bars":len(d),"matched_price_months":int(matched.sum()),
            "exact_level_match_months":int(same.sum()),"zero_volume_bars":int(d.volume.eq(0).sum()),
            "source_hash":hashlib.sha256(body).hexdigest(),
            "source_url":"https://fchart.stock.naver.com/sise.nhn?symbol="+code+"&timeframe=month&count=120&requestType=0",
            "data_use":"diagnostic_only; no inferred corporate-action payoff/no source overwrite"})
        merged.to_parquet(ROOT/"data/private"/("independent_price_compare_"+code+".parquet"),index=False)
        for row in merged.loc[merged.naver_return.abs().gt(1)|merged.volume.eq(0)].itertuples():
            events.append({"stock_code":code,"month":str(row.date.date()),"zero_volume":row.volume==0,
                "extreme_price_return":abs(row.naver_return)>1,"source_hash":hashlib.sha256(body).hexdigest()})
    pd.DataFrame(summaries).to_csv(ROOT/"output/independent_price_source_audit.csv",index=False)
    pd.DataFrame(events).to_csv(ROOT/"output/independent_price_event_flags.csv",index=False)
    evidence=[
       {"stock_code":"001880","event":"share_exchange_DL_construction_to_DL_EC","source":"https://kind.krx.co.kr/external/2023/12/06/000361/20231206001130/10601.htm",
        "known_terms":"0.3704268 new ordinary shares per old ordinary share; scheduled exchange 2024-02-14","resolved_return":False,
        "remaining":"monthly old ticker price is not verified as full successor consideration; no guessed terminal return"},
       {"stock_code":"002270","event":"Lotte_food_merger","source":"https://kind.krx.co.kr/external/2022/03/23/000780/20220323003339/11344.htm",
        "known_terms":"official merger valuation report","resolved_return":False,"remaining":"old-ticker zero-volume last bar is not a 0% liquidation return"},
       {"stock_code":"001140","event":"terminal_trading_price_observed_2026_01","source":"https://fchart.stock.naver.com/sise.nhn?symbol=001140&timeframe=month&count=120&requestType=0",
        "known_terms":"independent month bar has actual volume and closing price; terminal loss not zero","resolved_return":False,
        "remaining":"separate event/price basis cross-check required before modifying frozen original-return primary"}
    ]
    pd.DataFrame(evidence).to_csv(ROOT/"output/corporate_action_evidence.csv",index=False)
    print(json.dumps({"independent_price_series":len(summaries),"corporate_action_evidence":len(evidence),"primary_returns_overwritten":False}),flush=True)

if __name__=="__main__":audit()
