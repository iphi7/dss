# G_variant_G_pred_score

## 概要

アブレーション: **取引スコアを来期予測ベースに変更**し、グラフ $W_i$ の個性がスタイライズドファクトに影響するかを検証する。

カルマンフィルタは廃止（variant_F で不要を確認済み）。

## 変更点

| 項目 | base（これまで） | このバリアント |
|---|---|---|
| カルマン | あり | **なし** |
| スコア基準 | 現在推定値 `updated` | **来期予測値 `pred_next`** |
| グラフ $W_i$ の使われ方 | カルマン中間変数 | **スコアに直接入る** |

```python
pred_next = belief_phi * y + belief_rho * (W_i @ y)
score = value_weight * pred_next + trend_weight * (pred_next - y) + ...
```

## 結果

pred_score と base_no_graph（グラフ不使用）はほぼ一致。グラフ多様性の効果はなし。

副次的発見: |r| ACF lag-5 が 0.020 → 0.069 に改善（実データ 0.198 に近づく）。

→ 詳細は [検証レポート.md](検証レポート.md) を参照。

## 残課題

グラフが有効でない根本原因: $y \approx x$（観測から真の状態がほぼ回収できる）。  
次の方向: 公開/非公開成分の分離（$y_j = x_j^{\text{pub}} + \varepsilon$）。
