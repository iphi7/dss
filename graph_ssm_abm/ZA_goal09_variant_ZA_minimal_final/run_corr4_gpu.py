"""
ZA 相関非対称ループ Round H4: flight の希少化 + 負側の回復

H3 の診断: flight (閾値1.5σ) が危機プラトー中に多発し、その正結合が
負レジーム窓 (rc_q05) と中央値 (rc_q50) まで押し上げていた。
実データの flight は真の暴落日だけの希少イベント。

対応:
  - flight_thresh 1.5→2.2σ (極端1%分位はほぼ捕捉したまま発動を希少化)
  - stock_beta 2.0→2.5 (平常のレジーム結合を戻し深い負窓を回復)
  - drift_sigma 増強 (実データ並みの高金利遠征 → 負レジーム滞在)

  H11: H08 + thresh2.2 + sb2.5
  H12: H11 + drift_sigma 1.4e-4
  H13: H12 + flight_beta 5.0 (希少化した分、極端日の強度を上げる)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_SCALABLE_PARAMS

H08 = dict(ZA_SCALABLE_PARAMS, dgs10_flight_beta=4.0, dgs10_flight_thresh=1.5,
           dgs10_stock_beta=2.0, rate_kappa_pos_scale=0.55, dgs10_mr_center=6.5)

CONFIGS = {
    'H11_t22_sb25': dict(H08, dgs10_flight_thresh=2.2, dgs10_stock_beta=2.5),
    'H12_t22_drift14': dict(H08, dgs10_flight_thresh=2.2, dgs10_stock_beta=2.5,
                            dgs10_drift_sigma=1.4e-4),
    'H13_f50': dict(H08, dgs10_flight_thresh=2.2, dgs10_stock_beta=2.5,
                    dgs10_drift_sigma=1.4e-4, dgs10_flight_beta=5.0),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'corr4_rare_flight',
        'flightの希少化(2.2σ)+平常結合回復(sb2.5)+ドリフト増強。'
        '目標: rc_q05→-0.5台回復、rc_q50→負、mid→-0.05、極端分位+0.3維持。',
    )
