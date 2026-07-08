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
ROOT=Path('graph_ssm_abm/W_goal05_variant_W_decile10_leverage')
RESULTS=ROOT/'results_gpu_round5'
MEMO=ROOT/'検証メモ.md'
SEEDS=[1,2]
LAGS=[1,2,3,5,10,20,60]
LEV_WINDOWS=[1,3,5,7,10,20]

def cfg(**overrides):
    # Start around V06/V02, but specifically reduce decile10 over-leverage.
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
        participation_vol_power=0.75,
        impact_activity_scale=0.50,
        impact_activity_clip=2.00,
        impact_crash_threshold=1.25,
        impact_crash_scale=0.30,
        impact_crash_power=2.00,
        asym_pi_scale=0.75,
        asym_pi_centered=True,
        down_ewma_decay=0.42,
        stoploss_universal_scale=0.004,
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
    # Round5: 10d/20d側だけを戻すため、delayed_vol_delayを7日前後へ伸ばす。
    # 即時の asym は弱め、短期レバレッジの過剰化を避ける。
    'W16_W10_delay7_mid': cfg(
        stoploss_universal_scale=0.0, asym_pi_scale=0.16, down_ewma_decay=0.62,
        participation_vol_power=0.86, impact_activity_scale=0.78,
        delayed_vol_scale=1.20, delayed_vol_decay=0.95, delayed_vol_delay=7, delayed_vol_clip=1.45,
    ),
    'W17_W10_delay7_strong': cfg(
        stoploss_universal_scale=0.0, asym_pi_scale=0.12, down_ewma_decay=0.62,
        participation_vol_power=0.86, impact_activity_scale=0.78,
        delayed_vol_scale=1.80, delayed_vol_decay=0.96, delayed_vol_delay=7, delayed_vol_clip=1.70,
    ),
    'W18_W07_delay7_strong': cfg(
        stoploss_universal_scale=0.0, asym_pi_scale=0.16, down_ewma_decay=0.60,
        participation_vol_power=0.76, impact_activity_scale=0.62,
        delayed_vol_scale=1.80, delayed_vol_decay=0.96, delayed_vol_delay=7, delayed_vol_clip=1.70,
    ),
    'W19_W09_delay7_mid': cfg(
        stoploss_universal_scale=0.0, asym_pi_scale=0.20, down_ewma_decay=0.55,
        participation_vol_power=0.78, impact_activity_scale=0.66,
        delayed_vol_scale=1.30, delayed_vol_decay=0.95, delayed_vol_delay=7, delayed_vol_clip=1.50,
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

def target_row(hist): return metric_row(hist,'real')

def score(row,tgt):
    vals=[]
    # Heavily weight decile10 multi-window, target less negative than R03/V06.
    for w in LEV_WINDOWS:
        k=f'lev_dec10_{w}d'; denom=max(abs(tgt[k]),0.05)
        vals.append(2.0*abs(row[k]-tgt[k])/denom)
    # Keep decile1 near target/zero.
    for w in LEV_WINDOWS:
        k=f'lev_dec1_{w}d'; denom=max(abs(tgt[k]),0.05)
        vals.append(1.0*abs(row[k]-tgt[k])/denom)
    # Preserve ACF curve reasonably.
    for kind,wt in [('abs_acf',0.9),('sq_acf',0.9),('r_acf',0.4)]:
        for lag in LAGS:
            k=f'{kind}{lag}'; denom=max(abs(tgt[k]),0.05)
            vals.append(wt*abs(row[k]-tgt[k])/denom)
    # Lightly include tail/kurt.
    vals.append(0.2*abs(row['kurt_sp500']-tgt['kurt_sp500'])/max(tgt['kurt_sp500'],5.0))
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
    tgt=target_row(real_df)
    append('## Round5: 7日遅延ボラで中期レバレッジだけを補う\n\nRound4の3〜4日遅延では10d/20dがまだ弱かった。今回は delay=7、decay高め、即時asym弱めにして、1d/3dを抑えながら10d/20dの将来ボラ相関を戻せるかを見る。')
    all_rows=[]; aggs=[]
    for label,c0 in CONFIGS.items():
        rows=[]; t0=time.time()
        append(f'### {label}\n\nparams: asym={c0.asym_pi_scale}, down={c0.down_ewma_decay}, fear={c0.stoploss_universal_scale}, part={c0.participation_vol_power}, impact={c0.impact_activity_scale}')
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
    aggdf.to_csv(ROOT/'comparison_round5.csv',index=False)
    fulldf.to_csv(ROOT/'full_metrics_round5.csv',index=False)
    append('### Round3総合比較\n\n'+aggdf.to_markdown(index=False))
    cols=['label','score','std_sp500','kurt_sp500','abs_acf5','sq_acf5','sq_acf20','lev_dec10_1d','lev_dec10_3d','lev_dec10_5d','lev_dec10_7d','lev_dec10_10d','lev_dec10_20d','lev_dec1_5d']
    print(fulldf[cols].to_string(index=False))
    print('\nAGG')
    print(aggdf[['label','score_mean','abs_acf5_mean','sq_acf5_mean','sq_acf20_mean','lev_dec10_1d_mean','lev_dec10_3d_mean','lev_dec10_5d_mean','lev_dec10_7d_mean','lev_dec10_10d_mean','lev_dec10_20d_mean','lev_dec1_5d_mean']].to_string(index=False))

if __name__=='__main__': main()
