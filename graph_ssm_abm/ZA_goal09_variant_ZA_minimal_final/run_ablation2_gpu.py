"""
ZA アブレーション Round A2: 投資家/価格/金利系機構の個別除去
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from z117_config import Z117_PARAMS

CONFIGS = {
    # 参加率の日次ノイズ
    'A07_no_pnoise': dict(Z117_PARAMS, participation_noise_sigma=0.0),
    # asym_pi の tanh 飽和 (線形に戻る)
    'A08_no_asymsat': dict(Z117_PARAMS, asym_pi_sat=0.0),
    # ポートフォリオ・リバランス
    'A09_no_rebalance': dict(Z117_PARAMS, portfolio_rebalance_rate=0.0),
    # 金利トレンド補正 (レジームは水準のみで決まる)
    'A10_no_ratetrend': dict(Z117_PARAMS, rate_trend_scale=0.0),
    # 危機プラトー (幾何減衰のみに戻る)
    'A11_no_plateau': dict(Z117_PARAMS, disaster_plateau=0),
    # 危機の負ドリフト
    'A12_no_disastermu': dict(Z117_PARAMS, disaster_mu=0.0),
    # 市場アンカー (長期安定化)
    'A13_no_anchor': dict(Z117_PARAMS, market_anchor_strength=0.0),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'ablation2',
        '投資家/価格/金利系7機構の個別除去。A07=参加ノイズ、A08=asym飽和、'
        'A09=リバランス、A10=金利トレンド補正、A11=プラトー、A12=危機負ドリフト、A13=市場アンカー。',
    )
