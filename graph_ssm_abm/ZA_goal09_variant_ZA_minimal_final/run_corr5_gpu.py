"""
ZA 相関非対称ループ Round H5: 状態変数型 flight (危機週全体の結合)

H4 の学び: 単日超過分の結合では7日窓相関の極端分位に現れない
(窓内の他6日が通常結合のままで平均が薄まる)。実データの flight は
危機週全体で株安→金利低下が続くエピソード現象。

機構: |r| > thresh×実現ボラ で flight_state=1 に点火、日次 decay で減衰。
     状態が生きている間は Δy += flight_beta × state × r (全リターンが正結合)。

  H14: H12 + 状態型 (beta1.5, decay0.80, thr2.2)
  H15: 同 beta2.5
  H16: beta2.5, decay0.85, thr2.0
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_SCALABLE_PARAMS

H12 = dict(ZA_SCALABLE_PARAMS, dgs10_stock_beta=2.5, rate_kappa_pos_scale=0.55,
           dgs10_mr_center=6.5, dgs10_drift_sigma=1.4e-4,
           dgs10_flight_thresh=2.2)

CONFIGS = {
    'H14_st_b15_d80': dict(H12, dgs10_flight_beta=1.5, dgs10_flight_decay=0.80),
    'H15_st_b25_d80': dict(H12, dgs10_flight_beta=2.5, dgs10_flight_decay=0.80),
    'H16_st_b25_d85_t20': dict(H12, dgs10_flight_beta=2.5, dgs10_flight_decay=0.85,
                               dgs10_flight_thresh=2.0),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'corr5_state_flight',
        '状態変数型flight (危機週全体が正結合)。'
        '目標: 極端分位+0.3回復かつ rc_q05 -0.47維持、mid≈-0.05。',
    )
