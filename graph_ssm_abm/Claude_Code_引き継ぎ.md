# Claude Code 引き継ぎメモ: graph_ssm_abm

## 0. このプロジェクトの大きな目標

架空企業ネットワークと投資家エージェントのミクロな相互作用から、SP500 と DGS10 のような日次パスを60年程度生成する。

最終的に重視している stylized facts は以下。

- 生リターンの自己相関が過大でないこと
- 絶対リターン・二乗リターンの ACF、すなわち volatility clustering がある程度出ること
- leverage effect が出ること
  - 特に、当期ボラティリティ分位ごとの将来ボラ相関を 1d/3d/4d/5d/7d/10d/20d で見る
  - 10分位、つまり当期 \\(|r_t|\\) が大きい日の leverage が強すぎないこと
- 尖度・tail が実データに近いこと
  - 中心部の分布はそこまで悪くないため、中心を壊さず超レア大ジャンプでtailを厚くする方針が有望

現在の評価では `stylized_facts_analysis.ipynb` を使って、生成パスCSVを実データと比較している。

---

## 1. 現在のディレクトリ命名

現在は、元々の `variant_X` の記号をディレクトリ先頭に出す命名へ統一している。

例:

```text
R_base7_variant_R_acf_control
S_goal01_variant_S_tail_events
T_goal02_variant_T_graph_jumps
Y_goal07_variant_Y_investor_regime
```

命名対応表は以下。

- `graph_ssm_abm/DIRECTORY_NAMING.md`

注意: IDEで開かれている古い `base6_variant_R_acf_control` などのパスは現在の実体とは異なる。現在のR系は次。

```text
graph_ssm_abm/R_base7_variant_R_acf_control
```

---

## 2. 現状の結論

### 2.1 以前の暫定本命

R系の代表候補は以下。

```text
graph_ssm_abm/R_base7_variant_R_acf_control/results_gpu_round1/R03_more_tail/seed_2/generated_paths.csv
```

R03は raw return ACF、abs/sq ACF の過剰をかなり抑えた。ただし以下が残る。

- decile10 leverage の短期が強い
- 10d/20d が弱い
- 尖度が足りない
- seed依存がある

### 2.2 パラメータ調整系の改善: X

Rをベースに複数案を試したディレクトリ。

```text
graph_ssm_abm/X_goal06_variant_X_R_issue_fix
```

単一サンプルの総合バランスでは以下が良かった。

```text
graph_ssm_abm/X_goal06_variant_X_R_issue_fix/results_gpu_round2/X09_X03_less_short_leverage/seed_2/generated_paths.csv
```

X09はR03より短期decile10過剰をやや抑え、5dも近い。ただし10d/20dと尖度はまだ弱い。

### 2.3 機構導入系の改善: Y

ユーザー方針により、パラメータ調整ではなく機構導入で改善する方向に移った。

```text
graph_ssm_abm/Y_goal07_variant_Y_investor_regime
```

追加した主な機構:

- 下落後のストレス記憶 \\(q_t\\)
- リスク選好的投資家はストレス下で参加率・注文量を上げる
- リスク回避的投資家は退避する
- 市場ボラを直接操作せず、投資家行動から中期ボラを作る

有望候補:

```text
graph_ssm_abm/Y_goal07_variant_Y_investor_regime/results_gpu_round1/Y04_X04_pref_activity/seed_2/generated_paths.csv
```

Y04はX09よりも中期、特に10d decile10 leverage が自然になった。だが短期decile10がまだ強く、尖度が弱い。

その後、ユーザー指摘により「中心部を壊さず、超レア大ジャンプで尖度を補う」方向を試した。

尖度検証としては以下が有効。

```text
graph_ssm_abm/Y_goal07_variant_Y_investor_regime/results_gpu_round4/Y17_disaster_wideclip_lowlev/seed_2/generated_paths.csv
```

Y17 seed2 は kurtosis が約20まで上がり、実データの約21.8に近い。ただし abs/sq ACF が弱すぎるため、最終候補というより「超レア大ジャンプが尖度に効く」確認用。

現在の実用候補は以下の中間を狙うのがよい。

- Y14 seed2: ACFと10d leverageを保つが、短期decile10が強め
- Y15 seed2: 尖度と短期decile10抑制は少し良いが、ACFが弱い
- Y17 seed2: 尖度は良いがボラクラが消えすぎ

次の改善方針は、Y14/Y15の中間。

---

## 3. 主要ファイルと役割

### 3.1 全体・問題整理

| ファイル | 内容 |
|---|---|
| `graph_ssm_abm/問題点.md` | これまで出た問題点・仮説・次の課題の蓄積。長いが重要。 |
| `graph_ssm_abm/DIRECTORY_NAMING.md` | 現在のディレクトリ命名規則と旧名対応。 |
| `graph_ssm_abm/Claude_Code_引き継ぎ.md` | このファイル。Claude Code向け引き継ぎ。 |

### 3.2 評価・分析

| ファイル | 内容 |
|---|---|
| `stylized_facts_analysis.ipynb` | 生成パスと実データを比較するノートブック。ユーザーが主に見ている。 |
| `analyze_stylized_facts.py` | stylized facts 分析用スクリプト。過去からある。 |
| `output.csv` | 実データ。列は `Date`, `sp500_abs`, `DGS10_abs`, `sp500`, `DGS10`。 |

ノートブック内で重要な評価:

- 分位数プロット: リターンや絶対リターンの分位。分位1付近は最大級tailを見る。
- leverage heatmap: 当期 \\(|r_t|\\) 分位ごとに、将来平均ボラとの相関を見る。
- ACF: 生リターン、絶対リターン、二乗リターンを複数lagで見るべき。

### 3.3 R系: ACF制御ベース

| ファイル | 内容 |
|---|---|
| `graph_ssm_abm/R_base7_variant_R_acf_control/README.md` | R系の目的と主要結果。 |
| `graph_ssm_abm/R_base7_variant_R_acf_control/検証レポート.md` | R系の検証まとめ。 |
| `graph_ssm_abm/R_base7_variant_R_acf_control/model.py` | ConfigとCPUモデル。主にConfig確認用。 |
| `graph_ssm_abm/R_base7_variant_R_acf_control/model_gpu.py` | GPU実装。以後のX/Yもここから派生。 |
| `graph_ssm_abm/R_base7_variant_R_acf_control/run_round*.py` | R系の検証ラウンド。 |
| `graph_ssm_abm/R_base7_variant_R_acf_control/R_candidates_multilag_decile_diagnosis.csv` | R内候補をmulti-lag/decile leverageで再評価した表。 |

R03 path:

```text
graph_ssm_abm/R_base7_variant_R_acf_control/results_gpu_round1/R03_more_tail/seed_2/generated_paths.csv
```

### 3.4 X系: Rベースのパラメータ探索

| ファイル | 内容 |
|---|---|
| `graph_ssm_abm/X_goal06_variant_X_R_issue_fix/README.md` | X系の目的。 |
| `graph_ssm_abm/X_goal06_variant_X_R_issue_fix/検証メモ.md` | X各roundの自動追記メモ。 |
| `graph_ssm_abm/X_goal06_variant_X_R_issue_fix/検証レポート.md` | X系の検証まとめ。 |
| `graph_ssm_abm/X_goal06_variant_X_R_issue_fix/model.py` | R由来のConfig。 |
| `graph_ssm_abm/X_goal06_variant_X_R_issue_fix/model_gpu.py` | R由来GPU実装。機構追加なし。 |
| `graph_ssm_abm/X_goal06_variant_X_R_issue_fix/run_round1_gpu.py` | R03/R12/R10周辺の一括比較。 |
| `graph_ssm_abm/X_goal06_variant_X_R_issue_fix/run_round2_gpu.py` | X04/X03周辺の調整。 |
| `graph_ssm_abm/X_goal06_variant_X_R_issue_fix/run_round3_gpu.py` | 弱い長期非対称性の試行。 |
| `graph_ssm_abm/X_goal06_variant_X_R_issue_fix/selected_candidate_comparison.csv` | X系主要候補比較。 |

X系の最良寄り候補:

```text
graph_ssm_abm/X_goal06_variant_X_R_issue_fix/results_gpu_round2/X09_X03_less_short_leverage/seed_2/generated_paths.csv
```

### 3.5 Y系: 投資家ヘテロ性機構 + 超レアジャンプ

| ファイル | 内容 |
|---|---|
| `graph_ssm_abm/Y_goal07_variant_Y_investor_regime/README.md` | Y系の目的と追加機構概要。 |
| `graph_ssm_abm/Y_goal07_variant_Y_investor_regime/検証メモ.md` | Y各roundの自動追記メモ。 |
| `graph_ssm_abm/Y_goal07_variant_Y_investor_regime/検証レポート.md` | Y系の最新まとめ。まず読むべき。 |
| `graph_ssm_abm/Y_goal07_variant_Y_investor_regime/model.py` | Y系で追加したConfig項目を含む。 |
| `graph_ssm_abm/Y_goal07_variant_Y_investor_regime/model_gpu.py` | 投資家ストレス記憶・リスク選好ブースト等の実装本体。 |
| `graph_ssm_abm/Y_goal07_variant_Y_investor_regime/run_round1_gpu.py` | 投資家ヘテロ性機構の初回比較。 |
| `graph_ssm_abm/Y_goal07_variant_Y_investor_regime/run_round2_gpu.py` | Y04軸で中期記憶を伸ばした比較。 |
| `graph_ssm_abm/Y_goal07_variant_Y_investor_regime/run_round3_gpu.py` | 超レア大ジャンプ + decile10抑制。 |
| `graph_ssm_abm/Y_goal07_variant_Y_investor_regime/run_round4_gpu.py` | 超レア大ジャンプのclip拡大。 |
| `graph_ssm_abm/Y_goal07_variant_Y_investor_regime/selected_candidate_comparison.csv` | Y系主要候補比較。 |

Y系の有望候補:

```text
graph_ssm_abm/Y_goal07_variant_Y_investor_regime/results_gpu_round1/Y04_X04_pref_activity/seed_2/generated_paths.csv
```

尖度改善確認用:

```text
graph_ssm_abm/Y_goal07_variant_Y_investor_regime/results_gpu_round4/Y17_disaster_wideclip_lowlev/seed_2/generated_paths.csv
```

---

## 4. Y系で追加したConfig項目

`Y_goal07_variant_Y_investor_regime/model.py` に追加。

```python
investor_stress_scale
investor_stress_decay
investor_stress_threshold
investor_stress_clip
risk_pref_participation_scale
risk_pref_size_scale
risk_averse_withdraw_scale
risk_pref_buy_tilt
risk_pref_sell_tilt
```

実装場所は `model_gpu.py`。

ざっくり挙動:

1. 日次リターン下落からストレス記憶を更新する。
2. `vol_sensitivity > 0` の投資家はストレス下で参加率・注文量が増える。
3. `vol_sensitivity < 0` の投資家は退避する。
4. optionalでリスク選好的投資家に買い/売りtiltを入れられる。

注意:

- `risk_pref_buy_tilt` を強くすると、Y03 seed2のように leverage が正に反転してしまう。
- 逆張り買いはかなり小さく使うべき。

---

## 5. 現在の未解決課題

### 5.1 尖度とボラクラのトレードオフ

超レア大ジャンプと `exog_common_clip` 拡大は尖度に効く。

例:

```text
Y17_disaster_wideclip_lowlev_seed2
kurt ≈ 20.13
```

しかし、ボラクラが弱くなる。

```text
abs_acf5 ≈ 0.042
sq_acf5 ≈ 0.009
```

実データはおおよそ:

```text
kurt ≈ 21.76
abs_acf5 ≈ 0.289
sq_acf5 ≈ 0.228
```

したがって、Y17をそのまま使うのではなく、Y14/Y15との中間を狙う。

### 5.2 decile10 leverage の短期過大

Y04/Y14は10dが良いが、3d〜5dが少し強い。

Y14 seed2:

```text
dec10 3d ≈ -0.266
dec10 4d ≈ -0.253
dec10 5d ≈ -0.232
dec10 10d ≈ -0.169
```

実データ:

```text
dec10 3d ≈ -0.203
dec10 4d ≈ -0.207
dec10 5d ≈ -0.209
dec10 10d ≈ -0.176
```

短期だけ少し弱めたい。

### 5.3 20d leverage が弱い

多くの候補で20dが弱い。

これは単一の市場共通ストレス記憶では限界がある可能性が高い。

次は投資家ごとに異なる記憶長を持たせる案が有望。

$$
q_{i,t} = \rho_i q_{i,t-1} + (1-\rho_i)\max(-r_t,0)^2
$$

リスク選好的な大口投資家ほど \\(\rho_i\\) を大きくする。

---

## 6. 次にClaude Codeへ依頼すると良さそうな作業

### 推奨タスクA: Y14/Y15中間案

目的:

- Y14のACF・10d leverageを維持
- Y15程度まで短期decile10を弱める
- 尖度を少し上げる

具体案:

- Y14をベースにする
- `asym_pi_scale` と `investor_stress_scale` を少し下げる
- `exog_common_clip` は広げるが、Y17ほど極端にしない
- `exog_common_jump_prob` は 0.0015〜0.002 程度
- `exog_common_jump_sigma` は 0.10〜0.13 程度

### 推奨タスクB: 投資家ごとのストレス記憶長

目的:

- 20d leverage をミクロ機構から戻す

実装案:

- `investor_stress_state` をスカラーではなく `(n_inv,)` テンソルにする
- 投資家ごとに `stress_decay_i` を持つ
- `wealth_factor` と `vol_sensitivity` が高い投資家ほど decay を大きくする
- その状態を参加率・注文量に入れる

### 推奨タスクC: tail event のクラスタ化

目的:

- 超レア大ジャンプを単発ではなく、数日だけ余震的に残す
- ただし中心部は壊さない

実装案:

- market jump発生時に `jump_aftershock_state` を作る
- 数日で指数減衰
- 企業ノイズまたは投資家ストレスにだけ入れる
- 価格へ直接入れすぎない

---

## 7. 直近で見るべき候補CSV

Y04:

```text
graph_ssm_abm/Y_goal07_variant_Y_investor_regime/results_gpu_round1/Y04_X04_pref_activity/seed_2/generated_paths.csv
```

Y14:

```text
graph_ssm_abm/Y_goal07_variant_Y_investor_regime/results_gpu_round3/Y14_Y04_jump_only/seed_2/generated_paths.csv
```

Y15:

```text
graph_ssm_abm/Y_goal07_variant_Y_investor_regime/results_gpu_round4/Y15_Y12_wideclip_balanced/seed_2/generated_paths.csv
```

Y17:

```text
graph_ssm_abm/Y_goal07_variant_Y_investor_regime/results_gpu_round4/Y17_disaster_wideclip_lowlev/seed_2/generated_paths.csv
```

X09:

```text
graph_ssm_abm/X_goal06_variant_X_R_issue_fix/results_gpu_round2/X09_X03_less_short_leverage/seed_2/generated_paths.csv
```

R03:

```text
graph_ssm_abm/R_base7_variant_R_acf_control/results_gpu_round1/R03_more_tail/seed_2/generated_paths.csv
```

---

## 8. 注意事項

- 生成パスCSVは `path_id`, `Date`, `sp500_abs`, `DGS10_abs`, `sp500`, `DGS10` の形式。
- 実データ `output.csv` の列名は小文字 `sp500`, `DGS10` など。
- 既存の古いopen tabのパスは現在のディレクトリ名と一致しない可能性がある。
- GPU環境なので、長い60年パス生成は `*_gpu.py` を使う。
- 評価は単一seedだけで判断しない。最低でも seed1/seed2 を見る。
- ただしseed1が低ボラ・高尖度になることがしばしばあり、seed依存自体も問題として認識する。
