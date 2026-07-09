# ZA_goal09_variant_ZA_minimal_final

**目標**: Z117（Z_goal08 の最終候補）をベースにアブレーションを行い、
**なるべく簡潔なモデルで Z117 程度の結果を出す**最終モデルを作る。

## 出発点

- ベースモデル: Z117（SP500 と DGS10 を1966年初期値から相互作用しながら60年共生成）
- モデルの全体像・数式・パラメータ決定根拠: `Z117_モデル詳細.md`
- コード: `model.py`（Config、CPU参照実装）+ `model_gpu.py`（PyTorch 実装、60年1本 ≈ 55秒）
- 確定パラメータ: `z117_config.py` の `Z117_PARAMS`（フラット辞書、147項目。
  Z系の run スクリプトの10段継承チェーンを展開したもので、config.json と一致することを検証済み）

## アブレーションの方針

1. Z117 には検証の過程で足された機構が多数ある。各機構を個別に切って
   （`dict(Z117_PARAMS, <key>=<off値>)`）、指標への寄与を測る
2. 寄与が小さい機構は恒久的に除去し、必要なら残った機構を再調整する
3. 評価は Z系と同じ: 株（multi-lag ACF、分位別 leverage、裾分位点、尖度）+
   金利（変化 std/尖度/クラスタ、水準の大波）+ 相関（90日ローリングの分布）を
   seed 1/2/3 で比較。**Z117 の成績が判定基準**

### 切る候補（機構と off の指定）

| 機構 | off 指定 | 予想 |
|---|---|---|
| 第2余震（2-3週） | `jump_aftershock2_scale=0` | sq_acf lag10-20 が痩せる? |
| ジャンプボラ連動 | `jump_vol_coupling=0` | 静穏期の尖度爆発? |
| 中規模ジャンプ第2層 | `exog_common_jump2_prob=0` | q99-q995 の中間裾? |
| メガクラッシュ誘発 | `mega_triggers_episode=0` | sq_acf 希釈? |
| 投資家別ストレス記憶 | `investor_stress_decay_min=0, investor_stress_decay_max=0` | 10-20d leverage? |
| 参加率日次ノイズ | `participation_noise_sigma=0` | abs_acf1 上振れ? |
| asym_pi 飽和 | `asym_pi_sat=0` | 分位9/10 過剰? |
| 弱い全員一律 fear | `stoploss_universal_scale=0`（現状ほぼ無効） | 影響なしの見込み → 除去候補筆頭 |
| ポートフォリオ・リバランス | `portfolio_rebalance_rate=0` | 長期レジーム固定化? |
| 金利トレンド補正 | `rate_trend_scale=0` | 相関の2020s型レジーム? |
| 危機プラトー | `disaster_plateau=0` | sq_acf の lag5 平坦性? |
| 危機の負ドリフト | `disaster_mu=0` | dec10 leverage? |

## 結果 (完了)

アブレーション A1/A2 (個別13機構) → B (組合せ) → C/D (再調整) の5ラウンドで完了。

**最終モデル: ZA-final (D03)** — Z117 から5機構を除去 (fear/ボラ連動/金利トレンド補正/
メガ誘発/第2余震、パラメータ8個削減) し、3パラメータを再調整。
score 0.293 (Z117: 0.282、ノイズフロア±0.01内で同等)。q999/sq_acf/rc_q05 はむしろ改善。
詳細は `検証レポート.md`、パラメータは `za_final_config.py`。

## ファイル

| ファイル | 内容 |
|---|---|
| `Z117_モデル詳細.md` | ベースモデルの全体像・数式（全記号定義付き）・パラメータ決定根拠 |
| `model.py` / `model_gpu.py` / `metrics.py` | Z_goal08 からのコピー（変更なしで開始） |
| `z117_config.py` | Z117 確定パラメータのフラット辞書 |
