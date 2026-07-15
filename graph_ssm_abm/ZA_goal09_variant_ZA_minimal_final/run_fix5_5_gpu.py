"""
ZA Round U: 投資家の資本回転 (退出・参入) — 富の凝縮の根治

診断の系譜: 暴騰暴落の頻度過剰・床ラチェットの真犯人 = 富の凝縮
(60年で実効投資家数 14.8→1人、金額imbalance 0.39→0.91 に飽和)。
マクロ介入 (ボラアンカー) は無効、rebalance も無効 (投資家間の凝縮は直せない)。

機構 (ユーザー設計): 幾何寿命 staggered の資本回転。交代時に
  退出 (ポートフォリオごとアクティブ売買から離脱) →
  参入 (現金のみ、規模=既存投資家の現金中央値スケール、行動パラメータ再ドロー)。
専用rng (seed+770001)。教訓2件: 参入資金を富中央値にすると資金流入スパイラルで
発散 / rho_per_dim の更新漏れで float32 発散 (t=1433)。

1seedサニティ (seed6): turnover 20y で std 1.04%(実1.05)・freq>50% 0.79%(実0.59)。

ベース = ZA_FINAL5 (S01)。
  U01: turnover 30y
  U02: turnover 20y
  U03: turnover 10y
監視: 既存指標 (abs/sq_acf, lev_dec10, kurt, ca_vshape, worst_21d, dy_*) 非劣化。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_FINAL5_PARAMS

CONFIGS = {
    'U01_to30y': dict(ZA_FINAL5_PARAMS, investor_turnover_mean_years=30.0),
    'U02_to20y': dict(ZA_FINAL5_PARAMS, investor_turnover_mean_years=20.0),
    'U03_to10y': dict(ZA_FINAL5_PARAMS, investor_turnover_mean_years=10.0),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix5_5',
        '資本回転で富の凝縮を根治。目標: std→0.0105, freq_12m_gt50→0.006, '
        'vol_floor_drift→1.0。監視: abs/sq_acf・lev_dec10・kurt・ca_vshape・成長率 非劣化。',
    )
