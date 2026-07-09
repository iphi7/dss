"""
ZA Round N2: 危機エピソード限定 flight (ハイブリッド構造)

N1 の学び: レジーム符号付き増幅だけでは、危機がそのパスの高金利期に落ちると
極端分位が強い負になる (時期配置は seed 運)。実データでは本物の暴落 (1987) は
高金利時代でも金利急落 = flight は暴落イベント固有。
→ ハイブリッド: 通常の高ボラ日はレジーム相関 (必要なら弱い増幅)、
  危機エピソード中だけ正結合の flight。

ベース = M14。
  N04: episode_flight 1.5
  N05: episode_flight 2.5
  N06: episode_flight 2.0 + regime_amp 1.0 (弱い増幅も併用)
  N07: episode_flight 2.0 + mega_triggers_episode 1.0 (メガ暴落にもflight波及)
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
    'N04_ef15': dict(M14, dgs10_episode_flight=1.5),
    'N05_ef25': dict(M14, dgs10_episode_flight=2.5),
    'N06_ef20_amp10': dict(M14, dgs10_episode_flight=2.0, dgs10_regime_amp=1.0),
    'N07_ef20_megatrig': dict(M14, dgs10_episode_flight=2.0, mega_triggers_episode=1.0),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix4_6',
        'エピソード限定flight。目標: 極端分位が mid より正側へ (+0.1〜0.3リフト)、'
        'dy_std<0.08 維持、rc分布・leverage非劣化。',
    )
