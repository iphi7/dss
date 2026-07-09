"""
ZA Round M4: 確率ドリフトの rng 分離後の再検証

M3 の異常 (σ=5e-7 でも dec10 -0.04 変化) の原因は、確率ドリフトが共有 rng から
毎日1回引くことで以降の全乱数がずれ、パス全体が別抽選になっていたこと
(共通乱数法の破壊)。専用 rng に分離して機構の真の効果を再測定する。

  M13: lin0.6 + σ5e-7 (rng分離)
  M14: lin0.6 + σ1e-6 (rng分離)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_LEAN_LM_PARAMS

BASE = dict(ZA_LEAN_LM_PARAMS, market_anchor_drift=0.00034, down_linear_coef=0.6)

CONFIGS = {
    'M13_iso_s5e7': dict(BASE, market_anchor_drift_sigma=5e-7),
    'M14_iso_s1e6': dict(BASE, market_anchor_drift_sigma=1e-6),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix4_4',
        'rng分離後の確率ドリフト再検証。M07 (σ=0, dec10=-0.190, dec3/4=-0.042) と比較し、'
        '真の機構効果を判定。',
    )
