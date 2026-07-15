"""
ZA Round R: 暴騰・暴落エピソードの頻度過剰 + 中期の弱い平均回帰

診断 (P01/Q01 histinit):
  - 12ヶ月変化率のピーク規模は実データと同等 (seed6 +89% vs 実+77%、下落も同等)。
  - だが >50%上昇の局面が実データの4倍以上の頻度 (seed6 368日 vs 実87日)。
  - 直近の中期平均回帰も弱い (VR63 7.48 vs 実6.42)。
  → 主因は trend_weight×delta (信念ドリフトの外挿) の自己強化トレンド。
    trend_weight ~ N(0.28, 0.12) を下げてバブル頻度と方向持続を実データへ。

ベース = ZA_FINAL4 (Q01)。既存指標 (abs/sq_acf, leverage, 尖度, 相関非対称) の
非劣化を注視 (trend は momentum/クラスタにも寄与しうるため)。
  R01: trend_weight_mean 0.20
  R02: trend_weight_mean 0.15
  R03: trend_weight_mean 0.10
  R04: trend_weight_mean 0.15 + std 0.08 (分散も縮小)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_FINAL4_PARAMS

CONFIGS = {
    'R01_tw20': dict(ZA_FINAL4_PARAMS, trend_weight_mean=0.20),
    'R02_tw15': dict(ZA_FINAL4_PARAMS, trend_weight_mean=0.15),
    'R03_tw10': dict(ZA_FINAL4_PARAMS, trend_weight_mean=0.10),
    'R04_tw15_s08': dict(ZA_FINAL4_PARAMS, trend_weight_mean=0.15, trend_weight_std=0.08),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix5_2',
        'trend_weight↓でバブル頻度(freq_12m_gt50)と中期持続を実データへ。'
        '目標: max_12m_up→+0.8前後, freq_12m_gt50↓, vol_floor_drift↓。'
        '監視: abs/sq_acf1, lev_dec10, kurt, ca_vshape 非劣化。',
    )
