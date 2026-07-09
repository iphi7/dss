"""
ZA フェーズ② Round I1: ハブ参加キャップによる大スケール leverage の回復

G系の残課題: ×4 (240/320) で dec10 leverage が -0.21→-0.15 に希釈。
原因仮説: BAハブは全セクターの隣接集合に入るため、スケールとともに
ハブ銘柄の参加者数だけが増え (粒度が死に)、時価総額加重の指数では
ハブの比重が大きいので指数レベルの leverage が薄まる。

対応: universe_hub_cap — 隣接/ランダム経由の参加を銘柄あたり cap 人までに制限
(専門セクターとしての参加は常に許可)。

  I01: ×1 + cap32 (基準スケールへの副作用確認)
  I02: ×4 capなし (相関版パラメータでの対照)
  I03: ×4 + cap32
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_common import run_ablation
from za_final_config import ZA_CORR_PARAMS

CONFIGS = {
    'I01_s1_hub32': dict(ZA_CORR_PARAMS, universe_hub_cap=32,
                         n_investors=60, n_firms=80, n_sectors=8),
    'I02_s4_nohub': dict(ZA_CORR_PARAMS,
                         n_investors=240, n_firms=320, n_sectors=32),
    'I03_s4_hub32': dict(ZA_CORR_PARAMS, universe_hub_cap=32,
                         n_investors=240, n_firms=320, n_sectors=32),
}

if __name__ == '__main__':
    run_ablation(
        CONFIGS, 'scale6_hubcap',
        'ハブ参加キャップで×4のleverage希釈を回復する。'
        '目標: ×4 dec10_5d -0.15→-0.19以上、×1への副作用なし。',
    )
