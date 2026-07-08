from __future__ import annotations

import json, sys, time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import Config
from model_gpu import simulate_market_gpu
from metrics import summarize_stylized_facts, _acf

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT=Path('graph_ssm_abm/V_goal04_variant_V_multilag_control')
RESULTS=ROOT/'results_gpu_round2'
MEMO=ROOT/'検証メモ.md'
SEEDS=[1,2]
LAGS=[1,2,3,5,10,20,60]
LEV_LAGS=[1,2,3,5,7,10,20,40,60]
LEV_WINDOWS=[1,3,5,7,10,20]

def cfg(**overrides):
    # R03_more_tail base
    params=dict(
        price_impact=0.030,
        idio_vol=0.0060,
        exog_common_sigma=0.0030,
        exog_common_jump_prob=0.006,
        exog_common_jump_sigma=0.040,
        exog_common_clip=0.080,
        realized_vol_lambda=0.970,
        vol_sensitivity_mean=0.55,
        vol_sensitivity_std=0.65,
        wealth_sigma=1.00,
        wealth_vol_corr=0.80,
        participation_vol_power=0.90,
        impact_activity_scale=0.70,
        impact_activity_clip=2.00,
        impact_crash_threshold=1.25,
        impact_crash_scale=0.30,
        impact_crash_power=2.00,
        asym_pi_scale=1.20,
        asym_pi_centered=True,
        down_ewma_decay=0.55,
        stoploss_universal_scale=0.015,
        stoploss_universal_threshold=0.004,
        market_anchor_strength=0.006,
        market_anchor_clip=0.005,
        market_anchor_gap_scale=0.70,
        market_anchor_drift=0.00025,
        score_centering=0.80,
        market_risk_premium_score=0.0,
        momentum_score_weight=0.00,
        firm_return_clip=0.08,
        rare_shock_prob=0.020,
        rare_shock_sigma=0.13,
        portfolio_rebalance_rate=0.0,
    )
    params.update(overrides)
    return Config(**params)

CONFIGS={
    # V02周辺。sq_acfを下げすぎず、decile10 leverageを抑える妥協点を探す。
    'V04_v02_mid': cfg(asym_pi_scale=0.90, down_ewma_decay=0.48, stoploss_universal_scale=0.006, participation_vol_power=0.82, impact_activity_scale=0.58),
    'V05_v02_more_sq': cfg(asym_pi_scale=1.00, down_ewma_decay=0.50, stoploss_universal_scale=0.006, participation_vol_power=0.85, impact_activity_scale=0.62),
    'V06_v02_low_abs': cfg(asym_pi_scale=0.75, down_ewma_decay=0.42, stoploss_universal_scale=0.004, participation_vol_power=0.75, impact_activity_scale=0.50),
}

def leverage_deciles(r, windows=LEV_WINDOWS, n_deciles=10):
    s=pd.Series(r).dropna().reset_index(drop=True)
    vol=s.abs(); dec=np.ceil(vol.rank(pct=True)*n_deciles).clip(1,n_deciles).astype(int)
    out={}
    for w in windows:
        fv=vol.rolling(w).mean().shift(-w)
        for d in range(1,n_deciles+1):
            idx=dec[dec==d].index; common=idx.intersection(fv.dropna().index)
            out[f'lev_dec{d}_{w}d']=float(np.corrcoef(s.iloc[common],fv.iloc[common])[0,1]) if len(common)>5 else np.nan
    return out

def global_lev(r):
    out={}; r=np.asarray(r); a=np.abs(r); sq=r*r
    for k in LEV_LAGS:
        if len(r)>k+5:
            out[f'glob_lev_abs_lag{k}']=float(np.corrcoef(r[:-k],a[k:])[0,1])
            out[f'glob_lev_sq_lag{k}']=float(np.corrcoef(r[:-k],sq[k:])[0,1])
    return out

def metric_row(df,label):
    r=summarize_stylized_facts(df,label)
    sp=df['sp500'].astype(float).to_numpy(); a=np.abs(sp); sq=sp*sp
    for lag in LAGS:
        r[f'r_acf{lag}']=_acf(sp,lag); r[f'abs_acf{lag}']=_acf(a,lag); r[f'sq_acf{lag}']=_acf(sq,lag)
    for q in [0.95,0.99,0.995,0.999,1.0]:
        r[f'abs_q{str(q).replace(".","")}']=float(np.quantile(a,q))
    r['abs_q999_q99_ratio']=r['abs_q0999']/max(r['abs_q099'],1e-12)
    r['ann_return_approx']=float((df['sp500_abs'].iloc[-1]/max(df['sp500_abs'].iloc[0],1e-12))**(252/len(df))-1)
    r.update(leverage_deciles(sp)); r.update(global_lev(sp))
    return r

def build_targets(hist):
    return metric_row(hist,'real')

def score(row,tgt):
    # ACF curve score: absolute error normalized by max(|target|, 0.05)
    vals=[]
    for kind,w in [('abs_acf',1.0),('sq_acf',1.0),('r_acf',0.5)]:
        for lag in LAGS:
            k=f'{kind}{lag}'
            denom=max(abs(tgt[k]),0.05)
            vals.append(w*abs(row[k]-tgt[k])/denom)
    # leverage decile 1 and 10 multiple windows
    for d,wgt in [(1,0.8),(10,1.0)]:
        for win in LEV_WINDOWS:
            k=f'lev_dec{d}_{win}d'
            denom=max(abs(tgt[k]),0.05)
            vals.append(wgt*abs(row[k]-tgt[k])/denom)
    # global leverage curve
    for k_lag in [1,3,5,10,20,60]:
        for prefix in ['glob_lev_abs_lag','glob_lev_sq_lag']:
            k=f'{prefix}{k_lag}'
            denom=max(abs(tgt[k]),0.05)
            vals.append(0.5*abs(row[k]-tgt[k])/denom)
    # tail should not collapse; weakly include q999/max/kurt
    vals.append(0.3*abs(row['abs_q0999']-tgt['abs_q0999'])/max(tgt['abs_q0999'],0.02))
    vals.append(0.2*abs(row['kurt_sp500']-tgt['kurt_sp500'])/max(tgt['kurt_sp500'],5.0))
    return float(np.nanmean(vals))

def aggregate(rows,label,tgt):
    out={'label':label,'n_samples':len(rows)}
    keys=['std_sp500','skew_sp500','kurt_sp500','abs_q0999','abs_q10','ann_return_approx']
    for kind in ['r_acf','abs_acf','sq_acf']:
        for lag in LAGS: keys.append(f'{kind}{lag}')
    for d in [1,10]:
        for w in LEV_WINDOWS: keys.append(f'lev_dec{d}_{w}d')
    for k in keys:
        vals=np.array([r.get(k,np.nan) for r in rows],float)
        out[k+'_mean']=float(np.nanmean(vals)); out[k+'_std']=float(np.nanstd(vals))
    scores=np.array([score(r,tgt) for r in rows])
    out['score_mean']=float(np.nanmean(scores)); out['score_std']=float(np.nanstd(scores))
    return out

def append(text):
    with MEMO.open('a',encoding='utf-8') as f: f.write(text.rstrip()+'\n\n')

def main():
    print('device',DEVICE)
    hist=pd.read_csv('output.csv')
    n_days=min(15120,len(hist)-2)
    real_df=hist.tail(n_days).reset_index(drop=True)
    tgt=build_targets(real_df)
    MEMO.write_text('# V_goal04_variant_V_multilag_control 検証メモ\n\n',encoding='utf-8')
    append('## Round2: V02周辺の妥協点探索\n\nRound1ではV02が最良。V02はleverageと長期ACFを抑えるがsq_acfを下げすぎるため、少しだけpersistenceを戻した候補を比較する。')
    all_rows=[]; aggs=[]
    for label,c0 in CONFIGS.items():
        rows=[]; t0=time.time()
        append(f'### {label}\n\nparams: part={c0.participation_vol_power}, impact={c0.impact_activity_scale}, down={c0.down_ewma_decay}, fear={c0.stoploss_universal_scale}, asym={c0.asym_pi_scale}, rebal={c0.portfolio_rebalance_rate}')
        for seed in SEEDS:
            print('running',label,seed,flush=True)
            c=Config(**{**asdict(c0),'seed':seed,'n_days':n_days})
            outdir=RESULTS/label/f'seed_{seed}'; outdir.mkdir(parents=True,exist_ok=True)
            gen,firms,investors,aux=simulate_market_gpu(hist,c,device=DEVICE)
            gen.to_csv(outdir/'generated_paths.csv',index=False)
            if seed==SEEDS[0]:
                with open(outdir/'config.json','w',encoding='utf-8') as f: json.dump(aux['config'],f,ensure_ascii=False,indent=2)
            row=metric_row(gen,f'{label}_seed{seed}'); row['score']=score(row,tgt); rows.append(row); all_rows.append(row)
        agg=aggregate(rows,label,tgt); agg['elapsed_sec']=time.time()-t0; aggs.append(agg)
        pd.DataFrame(rows).to_csv(RESULTS/label/'full_metrics.csv',index=False)
        append(pd.DataFrame([agg]).to_markdown(index=False))
    aggdf=pd.DataFrame(aggs); fulldf=pd.DataFrame(all_rows)
    aggdf.to_csv(ROOT/'comparison_round2.csv',index=False)
    fulldf.to_csv(ROOT/'full_metrics_round2.csv',index=False)
    append('### Round2総合比較\n\n'+aggdf.to_markdown(index=False))
    cols=['label','score','std_sp500','kurt_sp500','r_acf1','abs_acf1','abs_acf5','abs_acf20','abs_acf60','sq_acf1','sq_acf5','sq_acf20','sq_acf60','lev_dec1_5d','lev_dec10_5d','lev_dec10_20d','glob_lev_abs_lag5','glob_lev_sq_lag5']
    print(fulldf[cols].to_string(index=False))
    print('\nAGG')
    print(aggdf[['label','score_mean','abs_acf1_mean','abs_acf5_mean','abs_acf20_mean','abs_acf60_mean','sq_acf1_mean','sq_acf5_mean','sq_acf20_mean','sq_acf60_mean','lev_dec1_5d_mean','lev_dec10_5d_mean','lev_dec10_20d_mean']].to_string(index=False))

if __name__=='__main__': main()
