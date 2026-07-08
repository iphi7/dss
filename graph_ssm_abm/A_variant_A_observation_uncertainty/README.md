# Graph-SSM ABM MVP

LLMを使わず、企業ネットワーク上の潜在状態伝播と、投資家ごとの主観グラフに基づく状態推定から、S&P500型の指数パスを生成するMVP。

## モデル概要

真の企業ネットワークを $G=(V,E)$、正規化隣接行列を $W$ とする。企業潜在状態 $x_t\in\mathbb{R}^N$ は

$$
x_{t+1}
=
\phi x_t
+ \rho W x_t
+ \eta_t
$$

で伝播する。観測されるIR情報・ニュースシグナルは

$$
y_t = Hx_t + \epsilon_t
$$

で与える。このMVPでは $H=I$ とする。

投資家 $i$ は、真の $W$ ではなく欠損した主観ネットワーク $W_i$ を持つ。

$$
\tilde{E}_i \subseteq E
$$

投資家は主観モデル

$$
\hat{x}^{(i)-}_t = A_i \hat{x}^{(i)}_{t-1},
\qquad
A_i = \phi_i I + \rho_i W_i
$$

に基づいて予測し、観測 $y_t$ を見て対角近似カルマンフィルタで更新する。

$$
K^{(i)}_{j,t}
=
\frac{P^{(i)-}_{j,t}}
{P^{(i)-}_{j,t} + R_i}
$$

$$
\hat{x}^{(i)}_{j,t}
=
\hat{x}^{(i)-}_{j,t}
+ K^{(i)}_{j,t}
\left(
y_{j,t}-\hat{x}^{(i)-}_{j,t}
\right)
$$

投資家の売買スコアは

$$
s_{i,j,t}
=
a_i\hat{x}^{(i)}_{j,t}
+ b_i\Delta \hat{x}^{(i)}_{j,t}
- c_i \sqrt{P^{(i)}_{j,t}}
- d_i r^f_t
+ e_i m_{j,t}
$$

で定義する。ここで $m_{j,t}$ は直近モメンタム、$r^f_t$ はDGS10水準である。

buy / hold / sell 確率は

$$
p^{buy}_{i,j,t}
=
\frac{\exp(\tau_i s_{i,j,t})}
{\exp(\tau_i s_{i,j,t})+1+\exp(-\lambda_i\tau_i s_{i,j,t})}
$$

$$
p^{sell}_{i,j,t}
=
\frac{\exp(-\lambda_i\tau_i s_{i,j,t})}
{\exp(\tau_i s_{i,j,t})+1+\exp(-\lambda_i\tau_i s_{i,j,t})}
$$

$$
p^{hold}_{i,j,t}
=
\frac{1}
{\exp(\tau_i s_{i,j,t})+1+\exp(-\lambda_i\tau_i s_{i,j,t})}
$$

で与える。 $\lambda_i>1$ のとき、悪材料に対する売り反応が買い反応より強くなる。

銘柄リターンは注文不均衡、企業潜在状態、市場共通ショック、個別ノイズから

$$
r_{j,t+1}
=
\kappa I_{j,t}
+ \beta_x x_{j,t}
+ c_t
+ \sigma_{j,t}\xi_{j,t}
$$

$$
I_{j,t}
=
\frac{B_{j,t}-S_{j,t}}
{B_{j,t}+S_{j,t}+\epsilon}
$$

で更新する。ここで $c_t$ は市場共通ショックであり、GARCH型の市場分散から生成する。

$$
v_t
=
(1 - \alpha_m - \beta_m)b_t
+ \alpha_m c_{t-1}^2
+ \beta_m v_{t-1}
+ \ell_m \max(-c_{t-1},0)^2
$$

$$
c_t = \sigma^m_t z_t,
\qquad
\sigma^m_t = \sqrt{v_t}
$$

$$
b_t
=
(\bar{\sigma}^m)^2
\left(
1 + a_s\,\mathrm{stress}_t
+ a_d\,\mathrm{downside}_t
\right)
$$

この項により、指数全体に残る共通ショックとボラティリティクラスタリングを表現する。

指数リターンは時価総額ウェイトで

$$
r^{SP}_{t+1}
=
\sum_j \omega_{j,t}r_{j,t+1}
$$

として集計する。

## 使い方

```bash
python -m graph_ssm_abm.run
```

生成物は `graph_ssm_abm/results/` に保存される。

- `generated_paths.csv`  
  `path_id,Date,sp500_abs,DGS10_abs,sp500,DGS10`
- `generated_path_output_format.csv`  
  `output.csv` と同じ `Date,sp500_abs,DGS10_abs,sp500,DGS10`
- `stylized_facts_summary.csv`
- `firms.csv`
- `investors.csv`
- `config.json`

## 注意

DGS10は実データ `output.csv` の末尾5年分を流用し、日付だけ生成期間に合わせて付け替える。
