# R_base7_variant_R_acf_control

`Q_base6_variant_Q_longrun_stable` で判明した問題、特に以下を抑えるための再検証ディレクトリです。

1. 全期間で `|r_t|` / `r_t^2` の自己相関が強すぎる。
2. raw return の lag 1 ACF が正に出すぎる。

Q系では60年安定化には成功した一方で、`Q17/Q20` の全期間ACFが実データよりかなり大きくなっていました。
本ディレクトリでは、ひとつ前のP/Q系構造をベースに、過剰ACFを明示的に抑える方向で再探索しました。

## 主な変更点

### 1. momentum 項の設定化

P/Qでは投資家スコアに

$$
0.25 r_{t-1}
$$

が固定で入っていました。これが raw return の正の自己相関を作っている可能性が高かったため、

```python
momentum_score_weight
```

として設定化しました。R系では基本的に `0.0` にしています。

### 2. activity / participation の弱体化

Q系で強かった volatility clustering は、以下の相互作用で過剰化していました。

- 高ボラ時に投資家の参加率が上がる。
- 出来高が増えると price impact が増える。
- 下落後に fear / asymmetric price impact が持続する。

そのため、R系では

- `participation_vol_power`
- `impact_activity_scale`
- `impact_activity_clip`
- `down_ewma_decay`

を下げて探索しました。

### 3. 弱い portfolio turnover の試行

全期間ACFには、60年間同じ投資家の保有・現金制約が固定されることによる長期レジームも効いている可能性がありました。
そこで弱い rebalancing を試しました。

$$
H_{i,t} \leftarrow (1-\epsilon)H_{i,t} + \epsilon H_{i,t}^{\mathrm{target}}.
$$

ただし、これは強すぎるとボラクラを消しすぎるため、現時点では慎重に扱うべきです。

## 実行ファイル

- `run_round1_gpu.py`: momentum削除、activity/fearを大きく弱体化。
- `run_round2_gpu.py`: 短い記憶でボラクラ/leverageを戻す。
- `run_round3_gpu.py`: portfolio rebalancing の検証。
- `run_round4_gpu.py`: tiny rebalancing とactivity復元の妥協案。

## 主要出力

- `candidate_ranking_by_full_score.csv`: 全候補の全期間スコア順ランキング。
- `all_full_metrics.csv`: 全候補・全seedの全期間指標。
- `real_Q17_R12_comparison.csv`: 実データ、Q17、R12の比較。
- `検証メモ.md`: 各ラウンドの自動追記メモ。
- `検証レポート.md`: 結果まとめ。

## 暫定候補

ACF抑制を重視するなら、現時点では `R03_more_tail` が最も良いです。

生成パス:

- `results_gpu_round1/R03_more_tail/seed_1/generated_paths.csv`
- `results_gpu_round1/R03_more_tail/seed_2/generated_paths.csv`

ただし seed1 ではボラクラが弱く、seed2 では良い、というseed依存があります。
より stationarity を重視する候補として `R12_tiny_rebal_active` も残していますが、こちらはボラクラ/leverageが弱いです。
