# W_goal05_variant_W_decile10_leverage

高ボラ十分位10の leverage が負に強すぎる問題を抑えるための検証ディレクトリです。

## 背景

R03/V06では、volatility decile 10 における

$$
\mathrm{corr}(r_t, \mathrm{future\ mean}\ |r|)
$$

が実データより負に強すぎる傾向がありました。
特に 1d〜7d horizon で過剰です。

そこで、down-memory / asymmetric price impact / market-wide fear を弱め、decile10 leverage を実データ側へ戻すことを目的にしました。

## 試した候補

Round1:

- `W00_V06_recheck`: V06相当。
- `W01_no_fear`: market-wide fearをゼロにする。
- `W02_low_asym`: 非対称impactをさらに弱める。
- `W03_min_asym_keep_acf`: 非対称性を最小寄りにし、activityは少し残す。

Round2:

- `W04_mid_asym035`: W02/W03中間。
- `W05_mid_asym040`: W04より非対称impactを少し戻す。
- `W06_mid_more_sq`: `sq_acf` を少し戻す候補。

## 主要比較 seed2

| model | abs_acf5 | sq_acf5 | sq_acf20 | dec10 1d | dec10 3d | dec10 5d | dec10 7d | dec10 10d | dec10 20d | dec1 5d |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real | 0.289 | 0.228 | 0.067 | -0.147 | -0.203 | -0.209 | -0.180 | -0.176 | -0.160 | -0.031 |
| V06 | 0.251 | 0.141 | 0.129 | -0.303 | -0.317 | -0.262 | -0.205 | -0.159 | -0.100 | -0.051 |
| W02 | 0.247 | 0.131 | 0.143 | -0.300 | -0.278 | -0.230 | -0.185 | -0.146 | -0.088 | -0.020 |
| W03 | 0.234 | 0.106 | 0.117 | -0.264 | -0.238 | -0.189 | -0.148 | -0.121 | -0.079 | -0.009 |
| W04 | 0.247 | 0.127 | 0.137 | -0.293 | -0.262 | -0.208 | -0.162 | -0.126 | -0.073 | -0.022 |
| W05 | 0.284 | 0.164 | 0.166 | -0.311 | -0.276 | -0.216 | -0.169 | -0.131 | -0.076 | 0.025 |

## 評価

今回の目的、つまり「decile10 5d を実データ近くまで弱める」ことに対しては `W04_mid_asym035 seed2` が最も良いです。

- decile10 5d: real `-0.209`, W04 `-0.208`
- decile10 7d: real `-0.180`, W04 `-0.162`
- decile1 5d: real `-0.031`, W04 `-0.022`

一方で、問題もあります。

- decile10 1d/3d はまだ負に強い。
- decile10 10d/20d は逆に弱くなりすぎる。
- `sq_acf5` は実データより低い。
- 尖度は低いまま。

## 暫定候補

- 分位10 leverage 調整候補: `W04_mid_asym035 seed2`
- 生成パス: `results_gpu_round2/W04_mid_asym035/seed_2/generated_paths.csv`

総合第一候補はまだR03だが、分位10 leverageを抑える目的ではW04が有用。
