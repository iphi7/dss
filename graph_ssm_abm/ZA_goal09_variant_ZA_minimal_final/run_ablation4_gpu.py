"""
ZA アブレーション Round C: B04 構造 (5機構除去) の再調整

B系の結果:
  - B04 (fear/ボラ連動/金利トレンド/メガ誘発/第2余震の5機構除去) が最良: 0.299
    (ベースライン 0.282、ノイズフロア ±0.01 をやや超える劣化)
  - 劣化は kurt (17.7→14.1) と sq_acf5 (0.16→0.13) に集中
  - asym飽和・投資家別ストレス記憶は除去すると劣化 → 保持確定

C系は B04 の残存機構を軽く再調整して Z117 水準への回復を狙う:
  C01: 危機を少し大きく (disaster_sigma 0.040→0.046) — kurt/sq5
  C02: 中規模ジャンプを少し大きく (jump2_sigma 0.016→0.020) — kurt
  C03: 両方を中間で (0.044 / 0.018)
  C04: 余震を長く (decay 0.72→0.80, scale 35→30) — sq5 (第2余震の代替)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from z117_config import Z117_PARAMS

# B04 = 5機構除去の簡素化構造
B04 = dict(
    Z117_PARAMS,
    stoploss_universal_scale=0.0,   # fear
    jump_vol_coupling=0.0,          # ボラ連動
    rate_trend_scale=0.0,           # 金利トレンド補正
    mega_triggers_episode=0.0,      # メガ誘発
    jump_aftershock2_scale=0.0,     # 第2余震
)

CONFIGS = {
    'C01_ds046': dict(B04, disaster_sigma=0.046),
    'C02_j2s020': dict(B04, exog_common_jump2_sigma=0.020),
    'C03_both': dict(B04, disaster_sigma=0.044, exog_common_jump2_sigma=0.018),
    'C04_longafter': dict(B04, jump_aftershock_decay=0.80, jump_aftershock_scale=30.0),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'ablation4_retune',
        'B04構造(5機構除去)の再調整。劣化(kurt 14.1, sq5 0.13)の回復を狙う。'
        'C01=危機σ増、C02=jump2σ増、C03=両方中間、C04=余震持続化。'
        '目標: ベースライン0.282と同等。',
    )
