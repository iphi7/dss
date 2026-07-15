"""
ZA Round V: 資本回転 + 段階的清算 (Round U の教訓)

Round U (即時交代) は全seedで暴騰が増加 (max_12m_up 0.9-2.5 vs 実0.77)。
原因: ホエールの即時消滅 = 売り供給の瞬間蒸発 + 参入現金の構造的ネット買い。
→ ユーザー設計どおり「退出者は数週かけて保有を市場に清算してから去る」を実装。
  退出発火 → exit_liq_days かけて毎日 ~3/日数 の割合で強制売却 (買いは停止)
  → 清算完了後に新規参入者 (現金中央値スケール、パラメータ再ドロー) と交代。

ベース = ZA_FINAL5 (S01)。
  V01: turnover 20y, liq 63d
  V02: turnover 10y, liq 63d
  V03: turnover 20y, liq 126d (よりゆっくり清算)
監視: std/freq_12m_gt50/max_12m_up/vol_floor_drift の改善と
  abs/sq_acf・lev_dec10・kurt・ca_vshape・成長率の非劣化。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_FINAL5_PARAMS

CONFIGS = {
    'V01_to20y_lq63': dict(ZA_FINAL5_PARAMS, investor_turnover_mean_years=20.0,
                           investor_exit_liq_days=63.0),
    'V02_to10y_lq63': dict(ZA_FINAL5_PARAMS, investor_turnover_mean_years=10.0,
                           investor_exit_liq_days=63.0),
    'V03_to20y_lq126': dict(ZA_FINAL5_PARAMS, investor_turnover_mean_years=20.0,
                            investor_exit_liq_days=126.0),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'fix5_6',
        '資本回転+段階的清算。目標: std→0.0105, freq→0.006, max12m→0.77, '
        'floor_drift→1.0。監視: 既存指標非劣化。',
    )
