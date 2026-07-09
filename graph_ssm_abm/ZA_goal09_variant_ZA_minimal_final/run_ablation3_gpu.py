"""
ZA アブレーション Round B: 無寄与機構のまとめて除去（組合せ検証）

A1/A2 の個別除去の結果:
  必須: jump2, プラトー, リバランス, 市場アンカー, 危機負ドリフト
  除去候補 (Δ≤ノイズフロア±0.01): fear, ボラ連動, 金利トレンド補正, asym飽和, メガ誘発
  グレー (+0.017-0.018): 第2余震, 投資家別ストレス記憶
  保持 (指標劣化明確): 参加ノイズ (abs_acf1)

個別で無寄与でも組合せで相互作用が出る可能性があるため段階的に切る。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from z117_config import Z117_PARAMS

# 除去候補セット
DROP_SAFE = dict(
    stoploss_universal_scale=0.0,   # fear
    jump_vol_coupling=0.0,          # ボラ連動
    rate_trend_scale=0.0,           # 金利トレンド補正
    mega_triggers_episode=0.0,      # メガ誘発
)

CONFIGS = {
    # 安全4機構のまとめて除去
    'B01_drop4': dict(Z117_PARAMS, **DROP_SAFE),
    # + asym飽和も除去 (個別では中位分位が改善)
    'B02_drop5': dict(Z117_PARAMS, **DROP_SAFE, asym_pi_sat=0.0),
    # + グレー2機構も除去 (最簡素候補)
    'B03_drop7': dict(
        Z117_PARAMS, **DROP_SAFE, asym_pi_sat=0.0,
        jump_aftershock2_scale=0.0,
        investor_stress_decay_min=0.0, investor_stress_decay_max=0.0,
    ),
    # B01 + 第2余震のみ追加除去 (グレーの切り分け)
    'B04_drop4_a2': dict(Z117_PARAMS, **DROP_SAFE, jump_aftershock2_scale=0.0),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'ablation3_combo',
        '無寄与機構のまとめて除去。B01=安全4機構(fear/ボラ連動/金利トレンド/メガ誘発)、'
        'B02=+asym飽和、B03=+第2余震+hetero記憶(最簡素)、B04=B01+第2余震。'
        '判定基準: Z117ベースライン score 0.282 ± ノイズフロア0.01。',
    )
