"""Formation uses only known features. Missing future returns invalidate cell months."""
import numpy as np
import pandas as pd

CELLS=["LowBM_LowG","LowBM_HighG","HighBM_LowG","HighBM_HighG"]


def classify(month,cut=.5,exclude_top5=False,market=None,max_age=550):
    d=month.loc[month.joint_eligible&month.age_days.le(max_age)].copy()
    if exclude_top5:
        top=month.loc[month.lag_me.gt(0)].nlargest(5,"lag_me").stock_code
        d=d.loc[~d.stock_code.isin(top)]
    if market is not None:d=d.loc[d.corp_cls_reported.eq(market)]
    if not len(d):return d.assign(cell=pd.Series(dtype=str))
    bmcut=d.bm_pit.median()
    d["bm_group"]=np.where(d.bm_pit<=bmcut,"LowBM","HighBM")
    pieces=[]
    for label,g in d.groupby("bm_group"):
        cutoff=g.g_value.quantile(cut)
        g=g.copy();g["cell"]=label+"_"+np.where(g.g_value<=cutoff,"LowG","HighG")
        g["g_cutoff"]=cutoff;g["bm_cutoff"]=bmcut
        pieces.append(g)
    return pd.concat(pieces,ignore_index=True)


def form(p,weighting="VW",cut=.5,exclude_top5=False,market=None,max_age=550,rebalance="monthly"):
    rows=[];holdings=[];feature_rows=[];previous={}
    for date,month in p.groupby("date",sort=True):
        members=classify(month,cut,exclude_top5,market,max_age)
        locked=rebalance!="monthly" and any(len(v) for v in previous.values()) and not (date.month in ([1,4,7,10] if rebalance=="quarterly" else [7]))
        if locked:
            wanted=pd.concat([v[["stock_code","cell"]] for v in previous.values()])
            members=wanted.merge(month,on="stock_code",how="left",validate="one_to_one")
        for cell in CELLS:
            group=members.loc[members.cell.eq(cell)].copy()
            if weighting not in ("VW","EW"):raise ValueError("invalid_weighting")
            if locked and cell in previous:
                old=previous[cell].set_index("stock_code")
                drift=old.weight*(1+old["return"])
                raw=group.stock_code.map(drift)
                raw.index=group.index
            else:raw=group.lag_me if weighting=="VW" else pd.Series(1.,index=group.index)
            group["weight"]=raw/raw.sum() if len(group) and raw.sum()>0 else np.nan
            enough=len(group)>=5;missing=int(group["return"].isna().sum())
            valid=enough and not missing and group.weight.notna().all()
            ret=float((group.weight*group["return"]).sum()) if valid else np.nan
            if len(group) and not np.isclose(group.weight.sum(),1):raise ValueError("weight_sum_not_one")
            old=previous.get(cell)
            turnover=np.nan
            if old is None:turnover=1. if valid else np.nan
            elif valid and old["return"].notna().all():
                oldw=old.set_index("stock_code").weight*(1+old.set_index("stock_code")["return"])
                oldw=oldw/oldw.sum()
                neww=group.set_index("stock_code").weight
                union=oldw.index.union(neww.index)
                turnover=float((oldw.reindex(union,fill_value=0)-neww.reindex(union,fill_value=0)).abs().sum())
            hhi=float((group.weight**2).sum()) if len(group) else np.nan
            rows.append({"date":date,"cell":cell,"return":ret,"n_formed":len(group),"n_missing_return":missing,
                  "weight_sum":float(group.weight.sum()) if len(group) else np.nan,"status":"estimated" if valid else "insufficient_cell_or_missing_return",
                  "turnover_one_way_sum_abs":turnover,"hhi":hhi,"effective_N":1/hhi if hhi>0 else np.nan,
                  "top5_weight":float(group.weight.nlargest(5).sum()) if len(group) else np.nan,
                  "g_mean":float(group.g_value.mean()) if len(group) else np.nan,
                  "g_unique":int(group.g_value.nunique()) if len(group) else 0,
                  "g_tie_at_cut_fraction":float(group.g_value.eq(group.g_cutoff).mean()) if len(group) and "g_cutoff" in group else np.nan,
                  "mean_log_me":float(np.log(group.lag_me).mean()) if len(group) else np.nan,
                  "mean_bm":float(group.bm_pit.mean()) if len(group) else np.nan,
                  "mean_age_days":float(group.age_days.mean()) if len(group) else np.nan,
                  "mean_roe":float(group.roe.mean()) if len(group) and "roe" in group else np.nan,
                  "mean_leverage":float(group.leverage.mean()) if len(group) and "leverage" in group else np.nan,
                  "assets_ge_2trillion_fraction":float(group.total_assets_krw.ge(2e12).mean()) if len(group) and "total_assets_krw" in group else np.nan,
                  "reported_Y_fraction":float(group.corp_cls_reported.eq("Y").mean()) if len(group) else np.nan,
                  "reported_K_fraction":float(group.corp_cls_reported.eq("K").mean()) if len(group) else np.nan})
            if len(group):holdings.extend(group[["stock_code","date","cell","weight","return","lag_me","bm_pit","g_value","rcept_no","filing_date"]].to_dict("records"))
            previous[cell]=group
    returns=pd.DataFrame(rows)
    holding=pd.DataFrame(holdings)
    return returns,holding


def wide(returns):
    d=returns.pivot(index="date",columns="cell",values="return").reindex(columns=CELLS)
    d["spread"]=d.HighBM_HighG-d.HighBM_LowG
    return d.reset_index()


def longest_block(dates):
    dates=pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    if not len(dates):return dates
    breaks=np.r_[True,np.diff(dates.to_period("M").asi8)!=1]
    groups=np.cumsum(breaks)
    choices=[dates[groups==i] for i in np.unique(groups)]
    return sorted(choices,key=lambda g:(-len(g),g[0]))[0]
