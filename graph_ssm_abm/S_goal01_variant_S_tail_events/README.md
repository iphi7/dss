# S_goal01_variant_S_tail_events

`R_base7_variant_R_acf_control` の `R03_more_tail` をベースに、最上位tailと尖度不足を補うための検証ディレクトリです。

## 背景

R03 seed2 は raw return ACF と通常の volatility clustering はかなり改善していました。
一方で、以下の問題がありました。

- `|r|` の最上位分位、特に 99.9% 付近が実データより立ち上がらない。
- 全期間尖度が低い。
- 例外的な特大ジャンプが不足している。

そこで、状態を持続させる fear/activity を強めるのではなく、ACFを増やしにくい iid 的な稀な市場ジャンプを追加しました。

## notebook の「分位数」の意味

### stylized_facts_analysis.ipynb の 3

3番の「分位数」は下からの累積割合です。
コード上は

```python
q = np.linspace(0, 1, n)
np.quantile(abs_negative_returns, q)
np.quantile(positive_returns, q)
```

なので、`q=0.99` は下から99%点、つまり上位1%境界付近です。
`q=1` に近いほど最大級の変動幅を見ています。

### stylized_facts_analysis.ipynb の 5

5番の「十分位」は、当期絶対リターン `|r_t|` を10等分したグループです。
コード上は

```python
vol_decile = ceil(abs(r).rank(pct=True) * 10)
```

なので、1が低ボラ群、10が高ボラ群です。
これは `0.1, 0.2, ..., 1.0` の点そのものではなく、各10%区間に入るサンプル群を指します。

## 追加機構

`model.py` / `model_gpu.py` に以下を追加しました。

```python
market_tail_jump_prob
market_tail_jump_sigma
market_tail_jump_df
market_tail_jump_clip
market_tail_jump_neg_prob
```

価格形成では、企業全体に共通な稀なジャンプ

$$
J_t = I_t \cdot s_t \cdot |\sigma_J t_\nu|
$$

を加えます。

$$
r_{j,t}
= r_{j,t}^{\mathrm{base}} + J_t.
$$

ここで $I_t$ は低確率で1になるベルヌーイ変数、$s_t$ は符号です。

重要な点として、tail jump 用の乱数は通常の市場シミュレーション乱数と分離しました。
これにより、R03 の基礎ダイナミクスを変えずに、ジャンプだけを重ねる検証になります。

## ラウンド

- Round1: 稀な大きめジャンプを追加。ただし既存RNGを消費してしまい、純粋な比較になっていなかった。
- Round2: 頻度高め・サイズ小さめの中ジャンプ。tailは改善したが、まだ純粋比較ではなかった。
- Round3: tail専用RNGに分離。R03の基礎乱数列を保ったままジャンプだけ追加。
- Round4: 最大級の数点を補う disaster jump。

## 暫定候補

現時点では `S11_disaster_mix seed2` が、R03 seed2 の問題意識に対して最も良い候補です。

生成パス:

- `results_gpu_round4/S11_disaster_mix/seed_2/generated_paths.csv`

比較:

| model | std | skew | kurt | r_acf1 | abs_acf5 | sq_acf5 | abs_q999 | max | q999/q99 | lev decile 10, 5d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real | 0.0105 | -0.677 | 21.76 | -0.0147 | 0.289 | 0.228 | 0.070 | 0.205 | 2.01 | -0.209 |
| R03 seed2 | 0.0126 | 0.062 | 5.89 | 0.0081 | 0.328 | 0.238 | 0.062 | 0.075 | 1.49 | -0.255 |
| S11 seed2 | 0.0149 | -0.203 | 13.15 | 0.0084 | 0.292 | 0.066 | 0.083 | 0.180 | 1.83 | -0.301 |

## 現時点の評価

改善した点:

- R03で不足していた最上位tailは改善。
- `abs_q999` は実データを上回る程度まで上がった。
- max absolute return も実データの 0.205 に近い 0.18 まで上がった。
- raw return ACF は低いまま。
- `abs_acf5` は実データにかなり近い。
- 高ボラ十分位での leverage は十分出ている。

残る問題:

- 全期間尖度は 13 程度で、実データの 21.8 にはまだ届かない。
- `sq_acf5` が低くなりすぎている。
- global leverage 指標 `corr(r_t, r_{t+1:t+20}^2)` は弱い。
- seed1側では低ボラ基調のため、同じジャンプが尖度を過剰に押し上げる。

## 次の方向性

S11は「R03 seed2のtail不足」には効いています。
次にやるなら、ジャンプ自体ではなく、ジャンプ後に2〜3日だけ限定的にボラが残る短期余韻を入れるのが良さそうです。
これにより `sq_acf5` と全期間尖度を少し戻しつつ、Q系のような長すぎるACFは避けられる可能性があります。
