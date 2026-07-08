
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
ROOT=Path('graph_ssm_abm/X_goal06_variant_X_R_issue_fix')
RESULTS=ROOT/'results_gpu_round3'
MEMO=ROOT/'検証メモ.md'
SEEDS=[1,2]
LAGS=[1,2,3,5,10,20,60]
LEV_WINDOWS=[1,3,5,7,10,20]

def cfg(**overrides):
    params=dict(
        price_impact=0.030,
        idio_vol=0.0060,
        exog_common_sigma=0.0030,
        exog_common_jump_prob=0.004,
        exog_common_jump_sigma=0.030,
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
    )
    params.update(overrides)
    return Config(**params)

CONFIGS={
    # Round3: 短期fearをさらに弱め、弱い非対称性を長めに残す。
    # 10d/20d decile10を少し戻しつつ、1d/3dの過剰化を避ける狙い。
    'X13_X09_slow_weak_asym': cfg(
        idio_vol=0.0081, exog_common_sigma=0.0044,
        exog_common_jump_prob=0.008, exog_common_jump_sigma=0.040,
        rare_shock_prob=0.024, rare_shock_sigma=0.145,
        participation_vol_power=1.45, impact_activity_scale=1.85, impact_activity_clip=3.60,
        asym_pi_scale=0.45, down_ewma_decay=0.72,
        stoploss_universal_scale=0.002, stoploss_universal_threshold=0.006,
        portfolio_rebalance_rate=0.00030,
    ),
    'X14_X12_activity_slow_tail': cfg(
        idio_vol=0.0081, exog_common_sigma=0.0043,
        exog_common_jump_prob=0.006, exog_common_jump_sigma=0.036,
        rare_shock_prob=0.040, rare_shock_sigma=0.16,
        participation_vol_power=1.50, impact_activity_scale=1.90, impact_activity_clip=3.70,
        asym_pi_scale=0.45, down_ewma_decay=0.70,
        stoploss_universal_scale=0.002, stoploss_universal_threshold=0.006,
        portfolio_rebalance_rate=0.00025,
    ),
    'X15_no_rebal_slow_min_fear': cfg(
        idio_vol=0.0074, exog_common_sigma=0.0039,
        exog_common_jump_prob=0.007, exog_common_jump_sigma=0.038,
        rare_shock_prob=0.026, rare_shock_sigma=0.145,
        participation_vol_power=1.25, impact_activity_scale=1.40, impact_activity_clip=3.00,
        asym_pi_scale=0.35, down_ewma_decay=0.75,
        stoploss_universal_scale=0.0, stoploss_universal_threshold=0.006,
        portfolio_rebalance_rate=0.0,
    ),
    'X16_X10_medium_decay': cfg(
        idio_vol=0.0078, exog_common_sigma=0.0043,
        exog_common_jump_prob=0.008, exog_common_jump_sigma=0.040,
        rare_shock_prob=0.026, rare_shock_sigma=0.15,
        participation_vol_power=1.45, impact_activity_scale=1.90, impact_activity_clip=3.60,
        asym_pi_scale=0.55, down_ewma_decay=0.70,
        stoploss_universal_scale=0.003, stoploss_universal_threshold=0.0055,
        portfolio_rebalance_rate=0.0010,
    ),
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

def metric_row(df,label):
    r=summarize_stylized_facts(df,label)
    sp=df['sp500'].astype(float).to_numpy(); a=np.abs(sp); sq=sp*sp
    for lag in LAGS:
        r[f'r_acf{lag}']=_acf(sp,lag); r[f'abs_acf{lag}']=_acf(a,lag); r[f'sq_acf{lag}']=_acf(sq,lag)
    for q in [0.95,0.99,0.995,0.999,1.0]:
        r[f'abs_q{str(q).replace(".","")}']=float(np.quantile(a,q))
    r['ann_return_approx']=float((df['sp500_abs'].iloc[-1]/max(df['sp500_abs'].iloc[0],1e-12))**(252/len(df))-1)
    r.update(leverage_deciles(sp)); return r

def real_metrics(hist):
    df=hist.copy()
    if 'sp500' not in df.columns and 'SP500' in df.columns:
        df=df.rename(columns={'SP500':'sp500','SP500_abs':'sp500_abs','DGS10':'dgs10','DGS10_abs':'dgs10_abs'})
    return metric_row(df,'real')

def score(row,tgt):
    vals=[]
    # さっきの問題を主目的に置く: decile10 multi-window
    for w in LEV_WINDOWS:
        k=f'lev_dec10_{w}d'; denom=max(abs(tgt[k]),0.05)
        vals.append(2.2*abs(row[k]-tgt[k])/denom)
    # decile1の異常な正相関/負相関も抑える
    for w in LEV_WINDOWS:
        k=f'lev_dec1_{w}d'; denom=max(abs(tgt[k]),0.05)
        vals.append(0.8*abs(row[k]-tgt[k])/denom)
    # multi-lag ACF
    for kind,wt in [('r_acf',0.45),('abs_acf',1.0),('sq_acf',1.0)]:
        for lag in LAGS:
            k=f'{kind}{lag}'; denom=max(abs(tgt[k]),0.05)
            vals.append(wt*abs(row[k]-tgt[k])/denom)
    # tail/kurt/stdを軽めに
    vals.append(0.35*abs(row['kurt_sp500']-tgt['kurt_sp500'])/max(tgt['kurt_sp500'],5.0))
    vals.append(0.3*abs(row['std_sp500']-tgt['std_sp500'])/max(tgt['std_sp500'],0.005))
    return float(np.nanmean(vals))

def aggregate(rows,label,tgt):
    out={'label':label,'n_samples':len(rows)}
    keys=['std_sp500','kurt_sp500','r_acf1','abs_acf1','abs_acf5','abs_acf20','abs_acf60','sq_acf1','sq_acf5','sq_acf20','sq_acf60']
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
    tgt=real_metrics(real_df)
    append('## Round3: 弱い非対称性を長く残して10d/20dを補う\n\nRound2ではX09 seed2が最良だったが10d/20dが弱い。今回はstop-loss/fearをほぼ消し、asym_piを弱くしつつdown_ewma_decayを長くして、中期側だけ少し戻るかを見る。')
    all_rows=[]; aggs=[]
    for label,c0 in CONFIGS.items():
        rows=[]; t0=time.time()
        append(f'### {label}\n\nparams: rebal={c0.portfolio_rebalance_rate}, fear={c0.stoploss_universal_scale}, asym={c0.asym_pi_scale}, down={c0.down_ewma_decay}, part={c0.participation_vol_power}, impact={c0.impact_activity_scale}, rare=({c0.rare_shock_prob},{c0.rare_shock_sigma})')
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
    aggdf.to_csv(ROOT/'comparison_round3.csv',index=False)
    fulldf.to_csv(ROOT/'full_metrics_round3.csv',index=False)
    append('### Round1総合比較\n\n'+aggdf.to_markdown(index=False))
    cols=['label','score','std_sp500','kurt_sp500','abs_acf5','sq_acf5','sq_acf20','lev_dec10_1d','lev_dec10_3d','lev_dec10_5d','lev_dec10_7d','lev_dec10_10d','lev_dec10_20d','lev_dec1_5d']
    print(fulldf[cols].to_string(index=False))
    print('\nAGG')
    print(aggdf[['label','score_mean','abs_acf5_mean','sq_acf5_mean','sq_acf20_mean','lev_dec10_1d_mean','lev_dec10_3d_mean','lev_dec10_5d_mean','lev_dec10_7d_mean','lev_dec10_10d_mean','lev_dec10_20d_mean','lev_dec1_5d_mean']].to_string(index=False))

if __name__=='__main__': main()
