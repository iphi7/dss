"""
ZA アブレーション Round D: 簡素化構造の最終調整

C系の結果:
  - C04 (余震持続化 30/0.80): 0.295 — 最良
  - C01 (危機σ 0.046): 0.302 だが kurt 16.5 と回復方向
  - 残る差 (+0.013 vs ベースライン0.282) は kurt (14-16.5 vs 17.7) に集中

D系は有望2要素の組合せ:
  D01: B04 + 危機σ0.046 + 余震(30, 0.80)
  D02: D01 の危機σ 0.050 (kurt をさらに)
  D03: D01 + プラトー 8→10 (sq5 の補強)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from z117_config import Z117_PARAMS

B04 = dict(
    Z117_PARAMS,
    stoploss_universal_scale=0.0,
    jump_vol_coupling=0.0,
    rate_trend_scale=0.0,
    mega_triggers_episode=0.0,
    jump_aftershock2_scale=0.0,
)

CONFIGS = {
    'D01_ds046_la': dict(B04, disaster_sigma=0.046,
                         jump_aftershock_decay=0.80, jump_aftershock_scale=30.0),
    'D02_ds050_la': dict(B04, disaster_sigma=0.050,
                         jump_aftershock_decay=0.80, jump_aftershock_scale=30.0),
    'D03_ds046_la_p10': dict(B04, disaster_sigma=0.046,
                             jump_aftershock_decay=0.80, jump_aftershock_scale=30.0,
                             disaster_plateau=10),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'ablation5_final',
        'B04構造の最終調整: 危機σ増×余震持続化の組合せ。'
        'D01=σ0.046+余震(30,0.80)、D02=σ0.050、D03=+プラトー10。'
        '目標: ベースライン0.282と同等 (≤0.29)。',
    )
