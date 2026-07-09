"""
ZA Round M3: 確率ドリフトσの最終調整 (③④の両立点)

M2: lin0.6 で dec3/4 解決 (M07)。確率ドリフト σ1e-6 は ④多様性+abs/sq改善だが
   dec10 -0.15 に弱化、lin との複合 (M09) では ③ 効果が半減。

  M10: lin0.6 + stoch σ=5e-7 (弱め)
  M11: lin0.6 + stoch σ=7e-7
  M12: lin0.9 + stoch σ=5e-7 (lin増で相互作用を補償)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_LEAN_LM_PARAMS

BASE = dict(ZA_LEAN_LM_PARAMS, market_anchor_drift=0.00034, down_linear_coef=0.6)

CONFIGS = {
    'M10_s5e7': dict(BASE, market_anchor_drift_sigma=5e-7),
    'M11_s7e7': dict(BASE, market_anchor_drift_sigma=7e-7),
    'M12_lin09_s5e7': dict(BASE, down_linear_coef=0.9, market_anchor_drift_sigma=5e-7),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix4_3',
        '③④両立点: dec3/4=-0.04維持 + dec10>-0.18 + 終値多様性。',
    )
