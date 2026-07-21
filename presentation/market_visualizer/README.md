# market_visualizer — ABM市場の内部状態ビジュアライザ

最終モデル (`../final_model/`, ZA-FINAL7) の**内部で何が起きているか**を記録・再生するツールです。
生成された SP500/DGS10 のパスだけでなく、

- どの企業がどう動いたか (価格・時価総額・需給不均衡・ボラ状態)
- どの投資家がどの企業をいくら買った/売ったか (注文単位)
- 投資家の現金・保有価値・富がどう推移したか
- 真の企業グラフと、投資家ごとの主観グラフ (欠損した見取り図) の違い
- 外生状態 (共通ノイズ・ジャンプ・危機強度) の推移

を日次で保存し、**単一HTMLのアニメーション**として再生できます。

## ディレクトリ

```text
market_visualizer/
  README.md            このファイル
  schema.md            ログCSVの列定義 (全テーブル)
  animate_market.py    ログ → 自己完結HTMLアニメーション
  visualize_static.py  ログ → 静的PNGサマリー
  test_s01_final/      実行環境 (最終モデル + 可視化ログ拡張)
    model.py           final_model/model.py + visual_log_* 設定フィールド
    model_gpu.py       final_model/model_gpu.py + 内部状態ログ (挙動は同一)
    za_final_config.py / z117_config.py   final_model と同一のパラメータ
    run_trial.py       ログ付き生成の実行スクリプト
    results/           実行結果 (1実行=1ディレクトリ)
```

`test_s01_final/` のモデルは `../final_model/` のコピーに**可視化ログを追加しただけ**のもので、
`visual_log_enabled=False` なら生成結果は final_model と完全に一致します
(ディレクトリ名は歴史的経緯で、モデル自体は ZA-FINAL7 に同期済み)。

## 使い方

### 1. ログ付き生成

```bash
cd presentation/market_visualizer/test_s01_final
python run_trial.py --seed 1 --n-days 120 --n-firms 40 --n-investors 20 --device cuda
```

主なオプション:

| オプション | 意味 |
|---|---|
| `--n-days` | 生成日数 (可視化用途では 100〜500日程度を推奨。ログは日次×企業×投資家なので長いと重い) |
| `--n-firms` / `--n-investors` / `--n-sectors` | 市場の規模 (小さくすると見やすい) |
| `--log-stride N` | N日ごとにログ (長期間を軽く記録したいとき) |
| `--top-orders N` | 1日あたり保存する注文の上限 (金額上位N件) |
| `--out-dir` | 出力先 (省略時 `results/trial_seedNN/`) |
| `--output-csv` | 実データCSV (初期値・履歴参照用。既定 `/home/u00121/output.csv`) |

出力 (1実行=1ディレクトリ、列定義は `schema.md`):

```text
generated_paths.csv     市場集計パス (+ 富集中度 _hhi, 平均不均衡 _imbal)
firm_states.csv         日次×企業の状態
investor_states.csv     日次×投資家の状態 (現金・保有・富・売買額)
orders.csv              注文ログ (金額上位のみ)
true_graph_edges.csv    真の企業グラフ
subjective_graph_edges.csv  投資家ごとの主観グラフ (上位エッジ)
market_log.xlsx         上記をまとめたExcel
config.json             使用した全パラメータ
```

### 2. HTMLアニメーション

```bash
python ../animate_market.py --result-dir results/trial_seed01
# → results/trial_seed01/market_animation.html をブラウザで開く
```

- 企業をグラフレイアウトで配置し、日ごとに「価格変化で色、時価総額でサイズ」を更新
- その日の注文を投資家→企業の矢印で表示 (`--max-orders-per-day` で本数制限)
- 下部に SP500/DGS10 パスと再生スライダー
- 生成されるHTMLは自己完結 (外部ライブラリ・ネット接続不要)

### 3. 静的サマリーPNG

```bash
python ../visualize_static.py --result-dir results/trial_seed01
```

## 設計

### ログの仕組み (test_s01_final/model_gpu.py)

- `Config` に `visual_log_*` フィールドを追加 (`model.py`)。`visual_log_enabled=True` のとき、
  シミュレーションループ内の4箇所でスナップショットを取る:
  1. **注文確定直後**: 金額上位 `visual_log_top_orders` 件の (投資家, 企業, 売買方向, 金額,
     約定前後の現金・ポジション) を記録
  2. **価格確定後**: 企業ごとの価格・リターン・時価総額・不均衡・ボラ状態・外生状態
  3. **同タイミング**: 投資家ごとの現金・保有時価・富・当日売買額
  4. **終了時**: 真のグラフと主観グラフのエッジ一覧
- ログはPythonリストに貯めて最後にDataFrame化 (`aux` 辞書で返す)。`visual_log_stride` で間引ける
- `visual_log_enabled=False` の場合はログ分岐が全てスキップされるため、
  final_model と同一seedで同一パスを生成する

### アニメーションの仕組み (animate_market.py)

- `build_payload()` がログCSV群を読み、日ごとの企業状態・注文・パスをJSON化
- 企業レイアウトは真のグラフに基づく決定論的配置 (`firm_layout()`)
- JSONをテンプレートHTMLに埋め込み、素のJavaScriptで再生 (依存なし・オフラインで動作)

### モデル本体との同期について

`test_s01_final/` のモデルは final_model のコピーに上記ログを差し込んだものです。
final_model 側を更新した場合は、`model_gpu.py` のログ4箇所 (`visual_log` で検索可能) を
保ったまま差分を取り込むこと。パラメータ (`za_final_config.py`, `z117_config.py`) は
そのままコピーで同期できます。

## 既知の制限 / 今後の候補

- ログは O(日数 × 企業数 + 日数 × 投資家数 + 日数 × 上位注文数)。60年フル生成のログは
  非現実的なので、可視化は数百日規模の実行を想定
- 主観グラフの時間変化 (資本回転で投資家が入れ替わる場合) は未対応 (最終版では回転無効のため影響なし)
- アニメーションの描画は企業数 ~80 までを想定
