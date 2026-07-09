"""
ZA アブレーション Round A1: ベースライン + ジャンプ/ショック系機構の個別除去

各腕は Z117 から機構を1つだけ切る。ベースライン (Z117 再現) との score 差 = その機構の寄与。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from z117_config import Z117_PARAMS

CONFIGS = {
    # ベースライン (Z117 そのもの)
    'A00_baseline': dict(Z117_PARAMS),
    # ほぼ無効の全員一律fear (除去候補筆頭)
    'A01_no_fear': dict(Z117_PARAMS, stoploss_universal_scale=0.0),
    # 遅い第2余震 (2-3週スケール)
    'A02_no_aftershock2': dict(Z117_PARAMS, jump_aftershock2_scale=0.0),
    # ジャンプのボラ連動
    'A03_no_volcoupling': dict(Z117_PARAMS, jump_vol_coupling=0.0),
    # 中規模ジャンプ第2層
    'A04_no_jump2': dict(Z117_PARAMS, exog_common_jump2_prob=0.0),
    # メガクラッシュのエピソード誘発
    'A05_no_megatrigger': dict(Z117_PARAMS, mega_triggers_episode=0.0),
    # 投資家別ストレス記憶長 (スカラー記憶に戻る)
    'A06_no_hetero_stress': dict(
        Z117_PARAMS, investor_stress_decay_min=0.0, investor_stress_decay_max=0.0,
    ),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'ablation1',
        'ベースライン(Z117) + ジャンプ/ショック系6機構の個別除去。'
        'score差 = 機構の寄与。A01=fear、A02=第2余震、A03=ボラ連動、'
        'A04=中規模ジャンプ、A05=メガ誘発、A06=投資家別ストレス記憶。',
    )
