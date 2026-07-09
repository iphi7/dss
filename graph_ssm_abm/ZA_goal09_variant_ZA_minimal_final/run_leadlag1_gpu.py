"""
ZA フェーズ③ Round J1: リードラグの導入

実データ (notebook 9節, 20dMA相互相関):
  金利先行側 (lag -7): -0.15、株先行側 (lag +20/+45): +0.10/+0.12。
  現モデルは lag0 中心の対称形でこの非対称構造がない。

機構 (どちらも遅い成分のみ、r_acf1汚染ループなし):
  1. dgs10_stocktrend_beta: 株の20日平均リターン → 金利ドリフト (株先行の正)
  2. rate_lag_score_beta: 外生金利変化の20日平均 → スコアを負に圧迫 (金利先行の負)

  J01: st_beta 0.4 のみ
  J02: st_beta 0.4 + rl_beta 3.0
  J03: st_beta 0.8 + rl_beta 6.0
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_PHASE2_PARAMS

CONFIGS = {
    'J01_st04': dict(ZA_PHASE2_PARAMS, dgs10_stocktrend_beta=0.4),
    'J02_st04_rl30': dict(ZA_PHASE2_PARAMS, dgs10_stocktrend_beta=0.4,
                          rate_lag_score_beta=3.0),
    'J03_st08_rl60': dict(ZA_PHASE2_PARAMS, dgs10_stocktrend_beta=0.8,
                          rate_lag_score_beta=6.0),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'leadlag1',
        'リードラグ2チャネル導入。目標: ll_m7→-0.15、ll_p20→+0.10、ll_asym→+0.25。'
        '既存指標 (r_acf1, dec10, rc分布) の非劣化。',
    )
