"""
ZA Round K2: abs1/sq1 膨張源の個別診断

K1 の発見: abs1/sq1 膨張 (0.33/0.35 vs G13 0.27/0.23) は flight/rl でなく
lean ベース自体にある。差分は {hub_cap 32, mr_center 6.5, drift_sigma 1.4e-4}。
仮説: ドリフト増強→金利水準の波が拡大→水準ペナルティ (g_i×y/100) 経由の
遅い共通スコア変動→ |r| の持続。

K03 (lean+st1.2+rl0.3) から1要素ずつ戻す:
  K05: drift_sigma 1.4e-4→1.0e-4
  K06: mr_center 6.5→5.5
  K07: hub_cap 32→0
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_SCALABLE_PARAMS

K03 = dict(ZA_SCALABLE_PARAMS, universe_hub_cap=32, dgs10_mr_center=6.5,
           dgs10_drift_sigma=1.4e-4, dgs10_stocktrend_beta=1.2,
           rate_lag_score_beta=0.3)

CONFIGS = {
    'K05_drift10': dict(K03, dgs10_drift_sigma=1.0e-4),
    'K06_mr55': dict(K03, dgs10_mr_center=5.5),
    'K07_nohubcap': dict(K03, universe_hub_cap=0),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'lean2',
        'abs1/sq1膨張源の診断。K05=ドリフト戻し、K06=mr戻し、K07=ハブキャップ戻し。'
        '目標: どれで abs1 0.345→0.27 へ戻るか特定。',
    )
