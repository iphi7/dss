"""
ZA Round Z2: 状態依存ファイアセール — leverage と床ラチェット抑制の両立

Round Z1 のトレードオフ: sellγ=0 で leverage 完全復元 (-0.209 = 実) だが
床ラチェット復活 (floor 1.52, freq 0.039)。sellγ=0.25 では床OK だが lev -0.157。
→ ファイアセールは危機時だけの現象。下方ストレス比 down_var/(realized/2) が
  閾値を超えるほど売りγを解除する状態依存に (平時ラチェット抑制・危機増幅)。

ベース = Y01 (γ0.5 + ノイズ復元 + asym_pi2.4 + drift 0.00030)、sellγ base=0.5。
  Z04: relief 1.0, thresh 1.0 (危機で完全解除)
  Z05: relief 1.0, thresh 0.8 (発動しやすく)
  Z06: relief 0.8, thresh 0.9 (中間)
目標: lev_dec10→-0.21 と floor≤1.05 / freq≤0.01 / std~0.011 の同時成立。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_FINAL5_PARAMS

Y01 = dict(ZA_FINAL5_PARAMS, whale_size_power=0.5,
           exog_common_sigma=0.0064, market_vol=0.0045,
           market_anchor_drift=0.00030, asym_pi_scale=2.4)

CONFIGS = {
    'Z04_r10_t10': dict(Y01, whale_fire_sale_relief=1.0, whale_fire_thresh=1.0),
    'Z05_r10_t08': dict(Y01, whale_fire_sale_relief=1.0, whale_fire_thresh=0.8),
    'Z06_r08_t09': dict(Y01, whale_fire_sale_relief=0.8, whale_fire_thresh=0.9),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix6_2',
        '状態依存ファイアセール。目標: lev→-0.21 かつ floor≤1.05/freq≤0.01/std~0.011。',
    )
