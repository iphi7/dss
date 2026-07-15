"""
ZA Round W: キャパシティ制約 (大口の執行制約) — 富の凝縮の市場支配を抑える

Round U (即時交代) / V (段階的清算) はともに失敗:
  退出時点でホエールが富の99%を握っており、どんな退出も市場地震になる
  (V: worst_21d -0.46〜-0.59, freq 0.07-0.10)。
→ 凝縮「後」の対処ではなく、凝縮の市場支配そのものを抑える。
  注文サイズ ×= (富シェア×n_inv)^(-γ)、平均以下は不変。
  実市場の大規模ファンドの執行制約 (インパクト回避で富の小割合しか動かさない)
  に対応するミクロ機構。市場支配の飽和と複利加速の両方を創発的に抑える。

ベース = ZA_FINAL5 (S01)。
  W01: γ=0.5 (単独)
  W02: γ=1.0 (単独)
  W03: γ=0.5 + turnover 20y/liq63 (パラメータ鮮度も維持)
監視: std/freq_12m_gt50/max_12m_up/vol_floor_drift の改善と
  abs/sq_acf・lev_dec10・kurt・ca_vshape・worst_21d・成長率の非劣化。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_FINAL5_PARAMS

CONFIGS = {
    'W01_g05': dict(ZA_FINAL5_PARAMS, whale_size_power=0.5),
    'W02_g10': dict(ZA_FINAL5_PARAMS, whale_size_power=1.0),
    'W03_g05_to20y': dict(ZA_FINAL5_PARAMS, whale_size_power=0.5,
                          investor_turnover_mean_years=20.0,
                          investor_exit_liq_days=63.0),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix5_7',
        'キャパシティ制約で富の凝縮の市場支配を抑制。目標: freq→0.006, '
        'max12m→0.77, floor_drift→1.0, std→0.0105。監視: 既存指標非劣化。',
    )
