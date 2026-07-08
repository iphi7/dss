# N19 アーキテクチャ README

## 位置づけ

N19 は `N_base3_variant_N_exog_jump_riskwealth` の中で、過剰に尖度を上げすぎず、かつボラティリティクラスタリングをある程度維持することを狙ったバランス案である。

N19 の主要な思想は、共通ショック \(c_t\) にボラティリティクラスタリングを担わせすぎず、ミクロな投資家行動からボラクラを作ることである。

特に、以下の経路を入れている。

$$
\text{高実現ボラ}
\to
\text{リスク選好型大口投資家の参加率・注文量増加}
\to
\text{市場取引量の増加}
\to
\text{注文インパクト増幅}
\to
\text{リターン変動の持続}
$$

## N19 の設定

N19 の主要パラメータは以下である。

```python
Config(
    vol_sensitivity_mean=0.80,
    vol_sensitivity_std=0.80,
    wealth_sigma=1.20,
    wealth_vol_corr=1.20,
    participation_vol_power=1.8,
    price_impact=0.080,
    impact_activity_scale=3.0,
    impact_activity_clip=4.0,
    exog_common_sigma=0.0060,
    exog_common_jump_prob=0.001,
    exog_common_jump_sigma=0.015,
    realized_vol_lambda=0.985,
)
```

## 企業状態

企業 \(j\) は 20 次元の潜在状態を持つ。

$$
x_{j,t} \in \mathbb{R}^{20}
$$

状態は以下に分かれる。

- public dims: 全投資家が観測可能
- sector dims: セクター専門家のみ強く観測
- private dims: 直接観測されず、企業グラフを通じて推定

各次元 \(k\) は企業ネットワーク \(W\) 上で伝播する。

$$
x_{k,t+1}
= \phi_k x_{k,t}
+ \rho_k W x_{k,t}
+ \eta_{k,t}
$$

ただし、N19 のグラフアブレーション結果から見ると、現状の集計 S&P リターンに対しては、この企業グラフ伝播の寄与はまだ小さい。

## 観測

公開成分はノイズ付きで全投資家に観測される。

$$
y^{pub}_{j,k,t}
= x_{j,k,t}
+ m_k (x_{j,k,t} - x_{j,k,t-1})
+ \varepsilon_{j,k,t}
$$

セクター成分は、投資家の専門セクターに対応する企業についてのみスコアに強く入る。

## 投資家の主観推定

投資家 \(i\) は主観グラフ \(W_i\) を持つ。通常の N19 では、これは真の企業グラフ \(W\) に欠損・誤認を入れたものである。

public 推定、sector 推定、private 推定を足して、企業 \(j\) の総合評価を作る。

$$
\hat{x}^{(i)}_{j,t}
= \hat{x}^{pub}_{j,t}
+ \hat{x}^{sec,(i)}_{j,t}
+ \hat{x}^{priv,(i)}_{j,t}
$$

private 推定は簡略化されており、公開観測の平均を主観グラフで近傍集約する。

$$
\hat{x}^{priv,(i)}_t
= \beta^{(i)}_h W_i \bar{y}^{pub}_t
$$

来期予測は

$$
\hat{x}^{(i)}_{t+1}
= \phi_i \hat{x}^{(i)}_t
+ \rho^{(i)}_s W_i \hat{x}^{(i)}_t
$$

である。

## 売買スコア

投資家 \(i\) の企業 \(j\) に対するスコアは

$$
S^{(i)}_{j,t}
= a_i \hat{x}^{(i)}_{j,t+1}
+ b_i(\hat{x}^{(i)}_{j,t+1} - \hat{x}^{(i)}_{j,t})
- c_i \sigma_i
- d_i r^f_t
+ e r^{SP}_{t-1}
$$

である。

このスコアから buy/sell/keep の確率を softmax 風に作る。

$$
z^{buy}_{i,j,t} = \exp(T_i S^{(i)}_{j,t})
$$

$$
z^{sell}_{i,j,t} = \exp(-L_i T_i S^{(i)}_{j,t})
$$

## 実現ボラに反応するリスク選好

投資家は \(c_t\) の内部 variance ではなく、S&P リターンの EWMA 実現ボラを見る。

$$
\hat{\sigma}^2_t
= \lambda \hat{\sigma}^2_{t-1}
+ (1-\lambda)(r^{SP}_t)^2
$$

N19 では

$$
\lambda = 0.985
$$

である。

投資家 \(i\) のボラ感応度を \(\gamma_i\) とすると、ボラ反応係数は

$$
f_{i,t}
= \mathrm{clip}\left(
1 + \gamma_i \left(\frac{\hat{\sigma}_{t-1}}{\sigma_0} - 1\right),
0.05,
4.0
\right)
$$

である。

\(\gamma_i > 0\) の投資家は高ボラ時に積極化し、\(\gamma_i < 0\) の投資家は縮小する。

N19 では \(\gamma_i\) の平均を正にしている。

$$
\gamma_i \sim \mathcal{N}(0.80, 0.80^2)
$$

## リスク選好と資産規模の相関

N19 では、リスク選好型投資家が資産規模でも大きくなりやすい。

$$
\log A_i
= \xi_i
+ \rho_A \frac{\gamma_i - \bar{\gamma}}{s_\gamma}
$$

ここで

$$
\rho_A = 1.20
$$

である。

これにより、高ボラ時に積極化する投資家が市場インパクトを持ちやすくなる。

## 高ボラ時の売買参加確率

N19 では、ボラ反応係数 \(f_{i,t}\) は注文サイズだけでなく、売買参加確率にも入る。

$$
z^{buy}_{i,j,t}
\leftarrow
z^{buy}_{i,j,t} f_{i,t}^{\alpha}
$$

$$
z^{sell}_{i,j,t}
\leftarrow
z^{sell}_{i,j,t} f_{i,t}^{\alpha}
$$

N19 では

$$
\alpha = 1.8
$$

である。

つまり高ボラ時のリスク選好型投資家は、単に大きく注文するだけでなく、そもそも `keep` から buy/sell に移りやすい。

## 注文サイズ

注文サイズは

$$
q^{(i)}_{j,t}
\propto
\mathrm{risk\_tolerance}_i
\cdot f_{i,t}
\cdot (0.25 + \mathrm{conviction}^{(i)}_{j,t})
$$

で決まる。

ここで conviction は

$$
\mathrm{conviction}^{(i)}_{j,t}
= \min\left(1, \frac{|S^{(i)}_{j,t}|}{0.12}\right)
$$

である。

## 価格形成

企業 \(j\) の buy value と sell value から注文不均衡を作る。

$$
I_{j,t}
= \frac{B_{j,t} - S_{j,t}}{B_{j,t} + S_{j,t} + \epsilon}
$$

ただし、これだけだと注文量の増加が分母で正規化されて消えてしまう。そこで N19 では、取引量の fast/slow EWMA 比率を用いて価格インパクトを増幅する。

$$
\nu_t
= \frac{V^{fast}_{t-1}}{V^{slow}_{t-1}}
$$

$$
\lambda_t
= \lambda_0
\left[1 + a \max(\nu_t - 1,0)\right]
$$

N19 では

$$
\lambda_0 = 0.080,
\qquad
a = 3.0
$$

である。

企業リターンは

$$
r_{j,t}
= \lambda_t I_{j,t}
+ c_t
+ \epsilon_{j,t}
$$

である。

## 外生共通ショック \(c_t\)

N19 では \(c_t\) は市場活動に依存しない。

$$
c_t = \varepsilon^c_t + J_t
$$

通常ノイズは

$$
\varepsilon^c_t \sim \mathcal{N}(0, 0.006^2)
$$

であり、ジャンプは確率 0.001 で発生する。

$$
J_t = 0.015 z_t,
\qquad z_t \sim t_3
$$

重要なのは、\(c_t\) の分散は `market_stress` や取引量で増幅されない点である。

## 指数リターン

各企業の時価総額ウェイトを \(w_{j,t}\) として、S&P 型指数リターンは

$$
r^{SP}_t
= \sum_j w_{j,t} r_{j,t}
$$

である。

## N19 の代表結果

| 指標 | real_tail | N19 |
|---|---:|---:|
| std | 0.0107 | 0.0087 |
| skew | 0.1584 | -0.7773 |
| kurtosis | 10.1046 | 12.1274 |
| absacf1 | 0.1626 | 0.1319 |
| absacf5 | 0.1977 | 0.1139 |
| leverage | -0.0427 | -0.0033 |
| SP-DGS10 corr | -0.0591 | 0.0290 |

## 解釈

N19 は N18/N20 よりボラクラ指標は控えめだが、尖度が過剰になりすぎず、標準偏差も実データに近づいている。

一方で、以下は未解決である。

- skew が負に寄りすぎる
- leverage effect が弱い
- DGS10 との相関が正で、実データの負相関と逆
- 企業グラフ・主観グラフの寄与がまだ小さい

特に最後の点は `N19_グラフアブレーション.md` を参照。
