# M_experiment_M_riskpref（第4基準モデル）

## 概要

base3 (d=20次元企業状態) に以下の2つの機構を追加したモデル。

| 変更点 | 内容 |
|---|---|
| **c_t の外生化** | GARCH omega を market_stress から切り離し、c_t を純粋な外生ショックとして機能させる |
| **vol_sensitivity** | 投資家がボラ水準に応じてポジションサイズを変える（リスク選好型: 高ボラ時に積極売買） |
| **vol_activity チャネル** | 取引量の fast/slow EWMA 比率を GARCH omega にフィードバックする |

---

## c_t 外生化の動機

従来モデルでは `garch_stress_scale=6.0` により：

```
market_stress = mean(|agg_est|) + 0.6 × |mean(imbalance)|
base_var = σ²_mkt × (1 + 6 × market_stress + ...)
c_t ~ t₅(0, √market_var)
```

c_t の分散が投資家の活動強度に強く依存しており、「外乱」ではなく「市場感応型ノイズ」になっていた。
本モデルでは `garch_stress_scale=0` として、GARCH 固有の持続性 (β=0.88) のみで変動させる。

また GARCH の安定条件を t₅ 分布に対して正確に満たすよう変更：

$$\alpha \times \frac{5}{3} + \beta < 1 \quad \Rightarrow \quad \alpha=0.05,\; \beta=0.88 \quad (0.083 + 0.88 = 0.963 < 1)$$

---

## vol_sensitivity と vol_activity チャネル

### vol_sensitivity (リスク選好パラメータ)

各投資家に `vol_sensitivity ~ N(+0.40, 0.60)` を付与：

```python
vol_ratio   = prev_market_vol_t / config.market_vol   # baseline との比
vol_factor  = clip(1 + vol_sensitivity × (vol_ratio - 1), 0.05, 4.0)
size_frac   = risk_tolerance × vol_factor × (0.25 + conviction)
```

| vol_sensitivity | 挙動 |
|---|---|
| > 0 (リスク選好型) | 高ボラ時: vol_factor > 1 → 積極売買 |
| < 0 (リスク回避型) | 高ボラ時: vol_factor < 1 → 縮小 |

平均 +0.40 により市場全体として弱いリスク選好バイアスを持つ。

### vol_activity チャネル (fast/slow 二重 EWMA)

```python
fast_ewma = 0.94 × fast_ewma + 0.06 × total_trade   # 半減期 ~11日
slow_ewma = 0.99 × slow_ewma + 0.01 × total_trade   # 半減期 ~69日
volume_ratio = fast_ewma / slow_ewma                  # mean-revert + 持続性
base_var = σ²_mkt × (1 + vol_activity_scale × max(0, volume_ratio - 1))
```

ボラクラのメカニズム：

```
大きな c_t → market_vol_t 上昇
           → リスク選好投資家の vol_factor 上昇 → 大量売買
           → fast_ewma 上昇 → volume_ratio > 1
           → base_var 増大 → 次の c_t 大
           → ~11日間持続 → ファットテール + ボラクラ
```

---

## 主な結果 (最良: R8)

設定: `vol_sensitivity_mean=0.40, vol_activity_scale=2.0, garch_stress_scale=0.0`

| 指標 | R0 (旧 base3) | R1 (c_t 外生化) | **R8 (最良)** | 実データ |
|---|---:|---:|---:|---:|
| std | 0.0127 | 0.0103 | **0.0109** | 0.0107 |
| kurtosis | 7.13 | 7.13 | **9.27** | 10.10 |
| absacf1 | 0.104 | 0.120 | **0.123** | 0.163 |
| absacf5 | 0.078 | 0.090 | **0.092** | 0.198 |
| leverage | −0.010 | −0.020 | **−0.015** | −0.043 |

詳細は [検証レポート.md](検証レポート.md) を参照。

---

## 設計上の注意

- `vol_sensitivity` のポジションサイズ scaling 単体では price_impact チャネルの SNR が低すぎて有効に機能しない
- ボラクラへの主な寄与は `vol_activity_scale=2` の fast/slow EWMA フィードバック
- `vol_activity_scale=4` はキャップ (0.012²) への頻繁な到達でむしろ kurtosis が低下する
- fast EWMA の λ=0.97 (半減期 23日) は鋭さが失われ kurtosis が悪化する (最適は λ=0.94)
