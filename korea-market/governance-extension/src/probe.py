import json
from pathlib import Path
from .dart_client import Client, APIError

def main():
    client=Client()
    results=[]
    for endpoint,params in [
        ("corpCode.xml",{}),
        ("outcmpnyDrctrNdChangeSttus.json",{"corp_code":"00126380","bsns_year":"2023","reprt_code":"11011"}),
        ("list.json",{"corp_code":"00126380","bgn_de":"20240101","end_de":"20241231","pblntf_ty":"A","page_count":100,"last_reprt_at":"N"}),
        ("fnlttSinglAcnt.json",{"corp_code":"00126380","bsns_year":"2023","reprt_code":"11011"}),
    ]:
        try:
            d,m=client.request(endpoint,params)
            r={"endpoint":endpoint,"status":m["status"],"rows":m["rows"],"hash":m["source_hash"]}
            if isinstance(d,dict):
                r["sample"]=d.get("list",[])[:2]
            results.append(r)
        except APIError as e:results.append({"endpoint":endpoint,"error":str(e)})
    Path("data/probe.json").write_text(json.dumps(results,ensure_ascii=False,indent=2))
    print(json.dumps(results,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
