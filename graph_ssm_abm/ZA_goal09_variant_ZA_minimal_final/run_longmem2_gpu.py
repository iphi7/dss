"""
ZA Round L2: さらに遅い記憶成分 (lag20-60 の裾)

L1: 第2余震(0.96)で sp lag2-10 回復、γ0.65 で金利全lag底上げ+dy_std完璧。
残り: sp abs lag20 (0.08 vs 実0.18)、sq lag20 (0.02 vs 0.07)、金利 lag60 (0.08 vs 0.18)。

  L07: L06 + 余震2を (6, 0.985) 半減期46日
  L08: L06 + 余震2を (5, 0.99) 半減期69日
  L09: L06 + γ 0.65→0.85
  L10: L07 + γ0.85 (複合)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_LEAN_PARAMS

L06 = dict(ZA_LEAN_PARAMS, jump_aftershock2_scale=5.0, jump_aftershock2_decay=0.96,
           investor_stress_decay_max=0.995, dgs10_vol_gamma=0.65)

CONFIGS = {
    'L07_a2_985': dict(L06, jump_aftershock2_scale=6.0, jump_aftershock2_decay=0.985),
    'L08_a2_99': dict(L06, jump_aftershock2_scale=5.0, jump_aftershock2_decay=0.99),
    'L09_g085': dict(L06, dgs10_vol_gamma=0.85),
    'L10_both': dict(L06, jump_aftershock2_scale=6.0, jump_aftershock2_decay=0.985,
                     dgs10_vol_gamma=0.85),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'longmem2',
        'さらに遅い記憶。目標: sp_abs lag20→0.15前後、金利lag60→0.12前後、既存維持。',
    )
