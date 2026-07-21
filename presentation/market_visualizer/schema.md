# market_visualizer ログスキーマ

## `generated_paths.csv`

市場集計パス。

| column | meaning |
|---|---|
| `Date` | 日付 |
| `sp500_abs` | SP500水準 |
| `DGS10_abs` | DGS10水準 |
| `sp500` | SP500日次リターン |
| `DGS10` | DGS10日次変化 |
| `_hhi` | 投資家富集中度の簡易HHI |
| `_imbal` | 企業平均の需給不均衡 |

## `firm_states.csv`

可視化用の日次企業状態。`visual_log_stride` ごとに保存。

| column | meaning |
|---|---|
| `day`, `Date` | 時刻 |
| `firm_id`, `sector` | 企業ID・セクター |
| `price`, `return` | 企業価格・日次リターン |
| `market_cap`, `market_weight` | 時価総額・市場ウェイト |
| `latent_signal` | 企業潜在状態の加重集約値 |
| `imbalance` | 買い注文額と売り注文額の不均衡 |
| `buy_value`, `sell_value` | 当該企業への買い/売り注文額 |
| `firm_vol` | 企業個別ボラ状態 |
| `common_noise`, `jump_abs`, `disaster_intensity` | 外生/危機状態の簡易ログ |

## `investor_states.csv`

可視化用の日次投資家状態。

| column | meaning |
|---|---|
| `investor_id` | 投資家ID |
| `cash` | 現金 |
| `holding_value` | 保有株式評価額 |
| `wealth` | 現金 + 保有価値 |
| `buy_value`, `sell_value` | その日の総買い/総売り注文額 |
| `expertise_sector` | 専門セクター |
| `vol_sensitivity` | 高ボラ時に動く/退避する傾向 |
| `risk_tolerance` | 注文サイズに効くリスク許容度 |
| `graph_quality` | 真の企業グラフとの近さ |
| `recognized_edges` | 主観グラフ上で認識している辺数 |
| `universe_size` | 売買対象銘柄数 |

## `orders.csv`

上位注文ログ。全注文を保存すると巨大化するため、デフォルトでは日次 value top-k。

| column | meaning |
|---|---|
| `investor_id`, `firm_id` | 誰がどの企業を売買したか |
| `side` | `buy` or `sell` |
| `value`, `quantity`, `price` | 注文額・株数・約定価格近似 |
| `cash_before`, `cash_after` | 注文集計前後の現金 |
| `position_before`, `position_after` | 当該銘柄の保有株数前後 |

## `true_graph_edges.csv`

```text
src_firm_id, dst_firm_id, true_weight
```

## `subjective_graph_edges.csv`

```text
investor_id, src_firm_id, dst_firm_id, perceived_weight
```
