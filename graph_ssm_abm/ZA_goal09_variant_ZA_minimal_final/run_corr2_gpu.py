"""
ZA 相関非対称ループ Round H2: 正側 kappa の非対称スケール

H1 の結果: flight で極端分位は到達 (H04 ext_lo=+0.33)。残り:
  - 中間分位 +0.13 (実 -0.08): tanh の低金利側の速い飽和で時間加重 kappa が正に偏る
  - rc_q05 が -0.39 に劣化: flight の正結合が危機窓の相関を薄める

対応: rate_kappa_pos_scale — 正側 kappa だけ縮める (負側=rc_q05 は無傷)。

  H05: H03 + pos_scale 0.55
  H06: H03 + pos_scale 0.55 + flight 4.0
  H07: H03 + pos_scale 0.70 + flight 4.0
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_SCALABLE_PARAMS

H03 = dict(ZA_SCALABLE_PARAMS, dgs10_flight_beta=3.0,
           dgs10_flight_thresh=1.5, dgs10_stock_beta=2.0)

CONFIGS = {
    'H05_ps055': dict(H03, rate_kappa_pos_scale=0.55),
    'H06_ps055_f40': dict(H03, rate_kappa_pos_scale=0.55, dgs10_flight_beta=4.0),
    'H07_ps070_f40': dict(H03, rate_kappa_pos_scale=0.70, dgs10_flight_beta=4.0),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'corr2_kappa_asym',
        '正側kappaの非対称スケール導入。目標: ca_mid +0.13→-0.08方向、rc_q05回復、'
        'vshape/hockey維持向上。H05=ps0.55、H06=+f4.0、H07=ps0.70+f4.0。',
    )
