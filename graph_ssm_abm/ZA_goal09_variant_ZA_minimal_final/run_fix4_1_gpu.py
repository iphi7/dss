"""
ZA Round M1: 課題④ (最終値バイアス) + ③ (dec4 leverage) の並行検証

④: アンカードリフト 0.00025 (年率6.5%) + 危機負ドリフト (-1.2%/年) で
   実効成長 5.3% vs 実 7.7% → 60年で×3不足、全パスが実データを下回る。
   再校正 + パスごと確率ドリフト (終値の多様性)。
③: leverage機構が neg²+閾値で小さい下落を無視するため dec3-5 が 0 近傍。
   閾値なしの弱い線形成分を併設。

  M01: drift 0.00034
  M02: drift 0.00036
  M03: drift 0.00034 + 確率ドリフト (σ=2e-6, ρ=0.9995)
  M04: down_linear_coef 0.08
  M05: down_linear_coef 0.15
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_LEAN_LM_PARAMS

CONFIGS = {
    'M01_drift34': dict(ZA_LEAN_LM_PARAMS, market_anchor_drift=0.00034),
    'M02_drift36': dict(ZA_LEAN_LM_PARAMS, market_anchor_drift=0.00036),
    'M03_drift34_stoch': dict(ZA_LEAN_LM_PARAMS, market_anchor_drift=0.00034,
                              market_anchor_drift_sigma=2e-6),
    'M04_lin008': dict(ZA_LEAN_LM_PARAMS, down_linear_coef=0.08),
    'M05_lin015': dict(ZA_LEAN_LM_PARAMS, down_linear_coef=0.15),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix4_1',
        '④終値バイアス (目標: 年率7.7%±、確率ドリフトで終値多様性) と'
        '③dec4 leverage (目標: dec3-5 → -0.04〜-0.10) の並行検証。',
    )
