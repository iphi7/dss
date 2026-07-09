"""
ZA Round M2: ④確率ドリフトの副作用確認 + ③線形応答の規模修正

M1: ④はdrift 0.00034で成長率解決 (7.6%/年)。確率ドリフト(σ2e-6)は多様性+全般改善だが
   dec10 -0.153 に弱化 → σ半減で再検証。
   ③は係数が1桁小さすぎた (線形寄与がdown_varベースラインの1%未満) → 0.6/1.2へ。

  M06: drift34 + 確率ドリフト σ=1e-6
  M07: drift34 + lin 0.6
  M08: drift34 + lin 1.2
  M09: drift34 + 確率σ1e-6 + lin 0.6 (複合)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_LEAN_LM_PARAMS

BASE = dict(ZA_LEAN_LM_PARAMS, market_anchor_drift=0.00034)

CONFIGS = {
    'M06_stoch1e6': dict(BASE, market_anchor_drift_sigma=1e-6),
    'M07_lin06': dict(BASE, down_linear_coef=0.6),
    'M08_lin12': dict(BASE, down_linear_coef=1.2),
    'M09_combo': dict(BASE, market_anchor_drift_sigma=1e-6, down_linear_coef=0.6),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix4_2',
        '④確率ドリフト副作用確認 (dec10維持で多様性) + ③線形応答×10 (dec3-5→-0.04〜-0.10)。',
    )
