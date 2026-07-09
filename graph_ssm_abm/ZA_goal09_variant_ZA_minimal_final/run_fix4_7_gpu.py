"""
ZA Round O1: ②リフト強化 + ①静穏フロアの状態依存化

N2: エピソード限定flightで正しいリフト構造 (負レジームでも暴落週は正方向へ+0.14)。
   さらに強め、①(中心質量: P(|r|<0.2%)=13-17% vs 実23%、静穏フロア0.008 vs 実0.006)
   を共通ノイズのボラ連動で対処。

ベース = M14 + episode_flight。
  O01: ef 3.5
  O02: ef 2.5 + exog_sigma_vol_coupling 0.5
  O03: ef 2.5 + coupling 0.8
  O04: ef 3.5 + coupling 0.5
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_LEAN_LM_PARAMS

M14 = dict(ZA_LEAN_LM_PARAMS, market_anchor_drift=0.00034,
           market_anchor_drift_sigma=1e-6, down_linear_coef=0.6)

CONFIGS = {
    'O01_ef35': dict(M14, dgs10_episode_flight=3.5),
    'O02_ef25_c05': dict(M14, dgs10_episode_flight=2.5, exog_sigma_vol_coupling=0.5),
    'O03_ef25_c08': dict(M14, dgs10_episode_flight=2.5, exog_sigma_vol_coupling=0.8),
    'O04_ef35_c05': dict(M14, dgs10_episode_flight=3.5, exog_sigma_vol_coupling=0.5),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix4_7',
        '②リフト強化 + ①静穏フロア。目標: 極端リフト+0.15以上、std→0.011、'
        'abs_acf1非劣化 (coupling の副作用監視)。',
    )
