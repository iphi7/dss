"""
ZA スケール検証 Round G3: tail 生成源の減衰 (λ は 0.019 のまま)

G1 (λ=0.019): std は全スケール適正 (0.011-0.012)、過剰なのは kurt (28-33) と q999 (0.082-0.096)
G2 (λ=0.013): 逆効果 — 取引ボラが弱まり静かなベースにジャンプが乗って kurt 43-64 に爆発

診断: ユニバース制限で「平常日の指数」が銘柄間分散効果でやや静かになり、
危機・ジャンプの相対サイズが過大になった。基礎ボラでなく tail 源を減衰させる。

  G09: ×1 (60/80),   減衰A (disaster_sigma 0.046→0.038, jump2 0.016→0.014, mega 0.20→0.18)
  G10: ×2 (120/160), 減衰A
  G11: ×1 (60/80),   減衰B 弱め (0.041 / 0.015 / 0.19)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_FINAL_V2_PARAMS

UNI = dict(universe_mode='sector_graph', universe_random_extra=5, universe_max_size=36)
DAMP_A = dict(disaster_sigma=0.038, exog_common_jump2_sigma=0.014, mega_crash_size=0.18)
DAMP_B = dict(disaster_sigma=0.041, exog_common_jump2_sigma=0.015, mega_crash_size=0.19)

CONFIGS = {
    'G09_s1_dampA': dict(ZA_FINAL_V2_PARAMS, **UNI, **DAMP_A,
                         n_investors=60, n_firms=80, n_sectors=8),
    'G10_s2_dampA': dict(ZA_FINAL_V2_PARAMS, **UNI, **DAMP_A,
                         n_investors=120, n_firms=160, n_sectors=16),
    'G11_s1_dampB': dict(ZA_FINAL_V2_PARAMS, **UNI, **DAMP_B,
                         n_investors=60, n_firms=80, n_sectors=8),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'scale3_damp',
        'λ=0.019のままtail源を減衰。G09=×1減衰A、G10=×2減衰A、G11=×1減衰B(弱め)。'
        '目標: kurt 28-33→20前後、q999 0.09→0.07、他は維持。',
    )
