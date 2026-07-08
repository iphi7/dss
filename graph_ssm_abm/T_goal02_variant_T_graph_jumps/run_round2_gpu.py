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
ROOT=Path('graph_ssm_abm/T_goal02_variant_T_graph_jumps')
RESULTS=ROOT/'results_gpu_round2'
MEMO=ROOT/'検証メモ.md'
SEEDS=[1,2]

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
        firm_return_clip=0.18,
        rare_shock_prob=0.020,
        rare_shock_sigma=0.13,
        portfolio_rebalance_rate=0.0,
    )
    params.update(overrides)
    return Config(**params)

CONFIGS={
    'T05_mixed_milder': cfg(graph_jump_prob=0.0010, graph_jump_sigma=0.045, graph_jump_df=2.8, graph_jump_clip=0.13, graph_jump_neg_prob=0.64, graph_jump_sources=2, graph_jump_degree_power=1.2, graph_jump_spread=0.55, graph_jump_hops=2, graph_jump_mode='abs_noise'),
    'T06_mixed_neg': cfg(graph_jump_prob=0.0010, graph_jump_sigma=0.050, graph_jump_df=2.7, graph_jump_clip=0.14, graph_jump_neg_prob=0.72, graph_jump_sources=2, graph_jump_degree_power=1.2, graph_jump_spread=0.55, graph_jump_hops=2, graph_jump_mode='abs_noise'),
    'T07_signed_mixed_hybrid': cfg(graph_jump_prob=0.0010, graph_jump_sigma=0.052, graph_jump_df=2.7, graph_jump_clip=0.145, graph_jump_neg_prob=0.68, graph_jump_sources=3, graph_jump_degree_power=1.3, graph_jump_spread=0.62, graph_jump_hops=2, graph_jump_mode='signed'),
}

def leverage_deciles(r, windows=(1,3,5,7), n_deciles=10):
    s=pd.Series(r).dropna().reset_index(drop=True)
    vol=s.abs(); dec=np.ceil(vol.rank(pct=True)*n_deciles).clip(1,n_deciles).astype(int)
    out={}
    for w in windows:
        fv=vol.rolling(w).mean().shift(-w)
        for d in [8,9,10]:
            idx=dec[dec==d].index; common=idx.intersection(fv.dropna().index)
            out[f'lev_dec{d}_{w}d']=float(np.corrcoef(s.iloc[common],fv.iloc[common])[0,1]) if len(common)>5 else np.nan
    return out

def metric_row(df,label):
    r=summarize_stylized_facts(df,label)
    sp=df['sp500'].astype(float).to_numpy(); a=np.abs(sp)
    for lag in [1,2,3,5,10,20,60]:
        r[f'r_acf{lag}']=_acf(sp,lag); r[f'abs_acf{lag}']=_acf(a,lag); r[f'sq_acf{lag}']=_acf(sp**2,lag)
    for q in [0.95,0.99,0.995,0.999,1.0]:
        r[f'abs_q{str(q).replace(".","")}']=float(np.quantile(a,q))
    r['abs_q999_q99_ratio']=r['abs_q0999']/max(r['abs_q099'],1e-12)
    r['min_sp500_abs']=float(df['sp500_abs'].min())
    r['ann_return_approx']=float((df['sp500_abs'].iloc[-1]/max(df['sp500_abs'].iloc[0],1e-12))**(252/len(df))-1)
    r.update(leverage_deciles(sp))
    return r

def aggregate(rows,label):
    cols=['std_sp500','skew_sp500','kurt_sp500','r_acf1','abs_acf5','sq_acf5','leverage_sp500_lag1_20','abs_q099','abs_q0999','abs_q10','abs_q999_q99_ratio','lev_dec9_5d','lev_dec10_5d','ann_return_approx']
    out={'label':label,'n_samples':len(rows)}
    for c in cols:
        vals=np.array([r[c] for r in rows if np.isfinite(r.get(c,np.nan))],float)
        out[c+'_mean']=float(vals.mean()) if len(vals) else np.nan
        out[c+'_std']=float(vals.std()) if len(vals) else np.nan
        out[c+'_min']=float(vals.min()) if len(vals) else np.nan
        out[c+'_max']=float(vals.max()) if len(vals) else np.nan
    return out

def append(text):
    with MEMO.open('a',encoding='utf-8') as f: f.write(text.rstrip()+'\n\n')

def main():
    print('device',DEVICE)
    hist=pd.read_csv('output.csv')
    n_days=min(15120,len(hist)-2)
    MEMO.write_text('# T_goal02_variant_T_graph_jumps 検証メモ\n\n',encoding='utf-8')
    real=metric_row(hist.tail(n_days).reset_index(drop=True),'real_full_tailmatch')
    append('## Round2: T03周辺の穏健化と負歪み調整\n\nRound1ではT03がabs/sq ACFに良かったが、seed2のstd/tailがやや強い。T03を穏健化しつつ、負歪みを少し入れる。\n\n実データ参照:\n\n'+pd.DataFrame([real]).to_markdown(index=False))
    all_rows=[]; aggs=[]
    for label,c0 in CONFIGS.items():
        rows=[]; t0=time.time()
        append(f'### {label}\n\nparams: mode={c0.graph_jump_mode}, prob={c0.graph_jump_prob}, sigma={c0.graph_jump_sigma}, sources={c0.graph_jump_sources}, spread={c0.graph_jump_spread}, hops={c0.graph_jump_hops}, residual_decay={c0.graph_jump_residual_decay}')
        for seed in SEEDS:
            print('running',label,seed,flush=True)
            c=Config(**{**asdict(c0),'seed':seed,'n_days':n_days})
            outdir=RESULTS/label/f'seed_{seed}'; outdir.mkdir(parents=True,exist_ok=True)
            gen,firms,investors,aux=simulate_market_gpu(hist,c,device=DEVICE)
            gen.to_csv(outdir/'generated_paths.csv',index=False)
            if seed==SEEDS[0]:
                with open(outdir/'config.json','w',encoding='utf-8') as f: json.dump(aux['config'],f,ensure_ascii=False,indent=2)
            row=metric_row(gen,f'{label}_seed{seed}')
            rows.append(row); all_rows.append(row)
        agg=aggregate(rows,label); agg['elapsed_sec']=time.time()-t0; aggs.append(agg)
        pd.DataFrame(rows).to_csv(RESULTS/label/'full_metrics.csv',index=False)
        append(pd.DataFrame([agg]).to_markdown(index=False))
    aggdf=pd.DataFrame(aggs); fulldf=pd.DataFrame(all_rows)
    aggdf.to_csv(ROOT/'comparison_round2.csv',index=False)
    fulldf.to_csv(ROOT/'full_metrics_round2.csv',index=False)
    append('### Round2総合比較\n\n'+aggdf.to_markdown(index=False))
    cols=['label','std_sp500','skew_sp500','kurt_sp500','r_acf1','abs_acf5','sq_acf5','leverage_sp500_lag1_20','abs_q099','abs_q0999','abs_q10','abs_q999_q99_ratio','lev_dec9_5d','lev_dec10_5d','ann_return_approx']
    print(fulldf[cols].to_string(index=False))
    print('\nAGG')
    print(aggdf[['label','std_sp500_mean','skew_sp500_mean','kurt_sp500_mean','r_acf1_mean','abs_acf5_mean','sq_acf5_mean','leverage_sp500_lag1_20_mean','abs_q0999_mean','abs_q999_q99_ratio_mean','lev_dec9_5d_mean','lev_dec10_5d_mean']].to_string(index=False))

if __name__=='__main__': main()
