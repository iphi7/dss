# L_experiment_L_mktcap

## 概要

base3 (d=20次元企業状態) をベースに **時価総額の偏在化** を実装し、スタイライズドファクトへの影響を検証したモデル。

---

## 時価総額の偏在化

```python
base_shares  ~ Pareto(a) + 1.0          # a=1.2: 偏在大, a→∞: 均一
degree_factor = (degree / mean_degree) ^ power   # BA次数との相関
shares = base_shares × degree_factor    # ハブ企業 = 大企業
```

BA グラフ上で次数の高いハブ企業ほど大企業（時価総額大）になる設計。
実データでは上位数社が指数の 20–30% を占めるが、現実的な Pareto 分布で再現を試みた。

---

## 検証の主な発見

### 1. 時価総額偏在の直接効果は小さい

Pareto (a=1.2) + 次数相関 (power=0.7) を加えても absacf5 は 0.077→0.075 とほぼ不変。

**理由**: インデックスへの price_impact チャネルの SNR が約 0.07 と低く、大企業へのウェイト集中が自己相関に繋がらない。

### 2. price_impact 強化が最も効果的

```
price_impact: 0.008 → 0.018 で absacf5: 0.077 → 0.093 (+21%)
```

### 3. market_stress によるボラ変動がボラクラの源泉

GARCH omega を固定（stress_scale=0 や rolling 実現ボラ）にすると absacf5 が 0.052 まで低下。
market_stress の変動こそが base_var の時変性を生み、ボラクラを支えていた。

### 4. 実現ボラ EWMA 方式の根本的問題 (Round 4)

t₅ 分布を使うと E[c²] = (5/3)×market_var → 有効持続性 α×(5/3)+β = 1.037 > 1 となり GARCH が発散。
外部 rolling 分散注入 (Round 5) は安定するが autocorrelation が低下した。

---

## 結果サマリ

| 設定 | std | kurt | absacf5 | leverage |
|---|---:|---:|---:|---:|
| 実データ | 0.0107 | 10.10 | 0.198 | −0.043 |
| base3 | 0.0128 | 6.93 | 0.077 | −0.011 |
| Pareto のみ | 0.0128 | 6.80 | 0.075 | −0.008 |
| **strong_pi (best)** | 0.0120 | 5.95 | **0.093** | −0.012 |
| rolling 実現ボラ | 0.0138 | 6.15 | 0.052 | −0.006 |

詳細は [検証レポート.md](検証レポート.md) を参照。
