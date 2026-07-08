# V_goal04_variant_V_multilag_control

R03をベースに、複数lagのACFと複数日horizonのleverageを目的関数に入れて再検証したディレクトリです。

## 背景

これまで `abs_acf5` や単一の leverage 指標を見ることが多かったが、U01/R03の確認で以下が分かった。

- `abs_acf5` は近くても、lag1/10/20/60で乖離が残る。
- leverage も5dだけでなく、1d/3d/5d/7d/10d/20dを見る必要がある。
- 特に低ボラ十分位1と高ボラ十分位10に局所的な歪みが出る。

そのため、評価関数に以下を入れた。

- raw / abs / squared return ACF: lag 1/2/3/5/10/20/60
- global leverage curve: lag 1/2/3/5/7/10/20/40/60
- volatility decile leverage: decile 1/10, window 1/3/5/7/10/20

## 試した候補

Round1:

- `V00_R03_recheck`: R03相当。
- `V01_less_persistence`: 参加率・出来高インパクト・down memoryを弱める。
- `V02_less_leverage`: 非対称impact/fearを弱め、高ボラ十分位leverageを抑える。
- `V03_tiny_turnover`: 非常に弱いturnover。

Round2:

- `V04_v02_mid`: V02より少しpersistenceを戻す。
- `V05_v02_more_sq`: `sq_acf` を少し戻す候補。
- `V06_v02_low_abs`: ACF/leverage抑制寄り。

## 主要比較 seed2

| model | score | abs_acf1 | abs_acf5 | abs_acf20 | abs_acf60 | sq_acf1 | sq_acf5 | sq_acf20 | sq_acf60 | dec1 5d | dec10 5d | dec10 20d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real | - | 0.261 | 0.289 | 0.180 | 0.101 | 0.230 | 0.228 | 0.067 | 0.032 | -0.031 | -0.209 | -0.160 |
| R03 | 0.481 | 0.426 | 0.328 | 0.297 | 0.306 | 0.344 | 0.238 | 0.197 | 0.205 | 0.002 | -0.255 | -0.070 |
| V02 | 0.371 | 0.385 | 0.273 | 0.261 | 0.264 | 0.285 | 0.168 | 0.158 | 0.160 | -0.007 | -0.253 | -0.075 |
| V05 | 0.353 | 0.358 | 0.249 | 0.231 | 0.236 | 0.246 | 0.143 | 0.123 | 0.129 | 0.010 | -0.289 | -0.109 |
| V06 | 0.312 | 0.363 | 0.251 | 0.236 | 0.236 | 0.252 | 0.141 | 0.129 | 0.133 | -0.051 | -0.262 | -0.100 |

## 評価

新しい複数lagスコアでは `V06_v02_low_abs seed2` が最良。
R03に比べて以下が改善した。

- raw return ACF は実データに近い。
- absolute return ACF の長い尾が弱まる。
- decile1 5d leverage が実データの負方向に近づく。
- decile10 20d leverage がR03より実データに近づく。

一方で、悪化した点もある。

- `sq_acf5/20/60` が実データより低くなりすぎる。
- 尖度は低いまま。
- seed1/seed2の差は依然大きい。

## 暫定結論

目的を「複数lag ACFと複数日leverageの形状改善」に置くなら、V06はR03より良い。
ただし、R03の良い二乗リターンACFを少し壊している。

したがって、現時点では以下の位置づけ。

- 全体第一候補: R03 seed2
- 複数lag/leverage形状改善候補: V06 seed2

生成パス:

- `results_gpu_round2/V06_v02_low_abs/seed_2/generated_paths.csv`
