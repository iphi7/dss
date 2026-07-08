# base3 variant N: 外生ジャンプ + リスク選好型大口投資家

## 目的

`M_experiment_M_riskpref` では、共通ショック \(c_t\) の分散が市場活動や `market_stress` によって増幅される余地があり、ボラティリティクラスタリングの主要因が「市場感応的な共通ノイズ」になってしまう懸念があった。

本 variant N では、以下を狙う。

1. \(c_t\) は市場活動から切り離し、低分散の通常ノイズ + 稀な外生ジャンプにする。
2. リスク選好型投資家が資産規模でも大きくなりやすいようにする。
3. 高ボラ時に、リスク選好型投資家は注文サイズだけでなく売買参加確率も上げる。
4. 取引量増加は \(c_t\) ではなく、注文インパクトを通じて価格変動に反映する。

つまり、外乱は外乱として残しつつ、ボラクラは投資家行動側の内生フィードバックで説明することを目標にした。

## モデル変更

### 1. 外生ジャンプとしての \(c_t\)

従来のように

$$
\bar{\sigma}^2_t = \sigma^2_{\mathrm{mkt}}(1 + a\,\mathrm{market\_stress}_t + \cdots)
$$

で \(c_t\) の分散を市場活動に連動させるのではなく、N 系では

$$
c_t = \epsilon_t + J_t
$$

とした。

通常成分は

$$
\epsilon_t \sim \mathcal{N}(0, \sigma_c^2)
$$

で、ジャンプ成分は

$$
J_t =
\begin{cases}
\sigma_J z_t, & u_t < p_J,\\
0, & \text{otherwise},
\end{cases}
\qquad z_t \sim t_3
$$

である。

この \(c_t\) は `market_stress` や取引量に依存しない。

### 2. 投資家が見るボラティリティ

投資家は \(c_t\) の内部的な `market_var` ではなく、生成された S&P リターンの EWMA 実現ボラを見る。

$$
\hat{\sigma}^2_t
= \lambda \hat{\sigma}^2_{t-1}
+ (1-\lambda) r^2_t
$$

次期の投資家行動は

$$
\frac{\hat{\sigma}_{t-1}}{\sigma_0}
$$

に反応する。

### 3. リスク選好と資産規模の相関

投資家 \(i\) のボラ感応度を \(\gamma_i\) とする。

$$
\gamma_i > 0
$$

なら高ボラ時に積極化するリスク選好型、

$$
\gamma_i < 0
$$

なら高ボラ時に縮小するリスク回避型である。

N 系では、資産規模 \(W_i\) を

$$
\log W_i
= \xi_i + \rho_W \cdot \frac{\gamma_i - \bar{\gamma}}{s_\gamma}
$$

で生成し、\(\rho_W > 0\) にすることで、リスク選好型が大口になりやすい構造を入れた。

### 4. 高ボラ時の参加確率と注文サイズ

ボラ比率を

$$
q_t = \frac{\hat{\sigma}_{t-1}}{\sigma_0}
$$

とし、投資家ごとの反応係数を

$$
f_{i,t} = \mathrm{clip}\{1 + \gamma_i(q_t - 1), f_{\min}, f_{\max}\}
$$

とする。

注文サイズは

$$
\mathrm{size}_{i,t}
\propto
\mathrm{risk\_tolerance}_i \cdot f_{i,t} \cdot (0.25 + \mathrm{conviction}_{i,t})
$$

で決まる。

また、売買確率の softmax odds も

$$
z^{buy}_{i,j,t} \leftarrow z^{buy}_{i,j,t} f_{i,t}^\alpha,
\qquad
z^{sell}_{i,j,t} \leftarrow z^{sell}_{i,j,t} f_{i,t}^\alpha
$$

とし、高ボラ時にリスク選好型は `keep` から売買へ移りやすくなる。

### 5. 取引量による注文インパクト増幅

初期実装では

$$
\mathrm{imbalance}_{j,t} = \frac{B_{j,t} - S_{j,t}}{B_{j,t} + S_{j,t}}
$$

のみで価格インパクトを決めていたため、注文量が増えても正規化で消えてしまった。

そこで改善版では、取引量の fast/slow EWMA 比率

$$
\nu_t = \frac{V^{fast}_{t-1}}{V^{slow}_{t-1}}
$$

を用いて、価格インパクトを

$$
\lambda_t
= \lambda_0\left[1 + a\max(\nu_t - 1,0)\right]
$$

とした。

企業リターンは

$$
r_{j,t}
= \lambda_t \mathrm{imbalance}_{j,t} + c_t + \epsilon_{j,t}
$$

である。

重要なのは、\(\nu_t\) は \(c_t\) の分散ではなく、注文インパクトにだけ入る点である。

## 出力

各実験の結果は以下に保存される。

- `results/<variant>/generated_paths.csv`
- `results/<variant>/firms.csv`
- `results/<variant>/investors.csv`
- `results/<variant>/config.json`

全体比較は以下。

- `comparison_summary.csv`

`generated_paths.csv` の形式はこれまでと同じく、`path_id`, `Date`, `sp500_abs`, `DGS10_abs`, `sp500`, `DGS10` を持つ。

## 実行

```bash
/home/u00121/.venv/bin/python -m graph_ssm_abm.N_base3_variant_N_exog_jump_riskwealth.run
```
