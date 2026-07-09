"""
ZA スケール検証 Round G2: 価格インパクトの粒度再校正

G1 の結果:
  - 主目的達成: 全スケールで内生ボラ・leverage が生存 (旧F02は崩壊していた)
  - 課題: 参加者密度 60→26人/銘柄 で粒度 sd(I)~1/sqrt(N_eff) が×1.5になり、
    全スケールで kurt=28-33・q999 が過剰 (基準スケールでも v2 の 0.291→0.392)

対応: λ (price_impact) を 0.019→0.013 (≈×1/1.5) に再校正して4スケール再走。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_FINAL_V2_PARAMS

UNI = dict(universe_mode='sector_graph', universe_random_extra=5, universe_max_size=36,
           price_impact=0.013)

CONFIGS = {
    'G05_s30_40': dict(ZA_FINAL_V2_PARAMS, **UNI, n_investors=30, n_firms=40, n_sectors=4),
    'G06_s60_80': dict(ZA_FINAL_V2_PARAMS, **UNI, n_investors=60, n_firms=80, n_sectors=8),
    'G07_s120_160': dict(ZA_FINAL_V2_PARAMS, **UNI, n_investors=120, n_firms=160, n_sectors=16),
    'G08_s240_320': dict(ZA_FINAL_V2_PARAMS, **UNI, n_investors=240, n_firms=320, n_sectors=32),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'scale2_lambda',
        'G1の粒度過剰 (kurt28-33) に対し λ 0.019→0.013 で再校正した4スケール掃引。'
        'G05=半分、G06=基準、G07=2倍、G08=4倍。',
    )
