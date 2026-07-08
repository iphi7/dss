"""
Q系: 60年長期安定モデルのGPU検証ループ。

P系のGPU belief-state SSM + leverage機構を土台に、企業別ファンダメンタル価値への
弱い平均回帰を追加し、60年生成でも価格崩壊しないかを複数seedで評価する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import Config
from model_gpu import simulate_market_gpu
from metrics import summarize_stylized_facts, _acf, _kurtosis_pearson

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = Path("graph_ssm_abm/Q_base6_variant_Q_longrun_stable")
RESULTS = ROOT / "results_gpu"
MEMO = ROOT / "検証メモ.md"


def base_config(**overrides) -> Config:
    # P系で長期検証の土台になった G33 近傍。
    # absacf/kurtosis が良く、leverage は弱め。Qでは長期安定性を優先し、fundamental anchorを追加。
    params = dict(
        price_impact=0.050,
        exog_common_sigma=0.0040,
        exog_common_jump_prob=0.006,
        exog_common_jump_sigma=0.035,
        exog_common_clip=0.100,
        realized_vol_lambda=0.985,
        vol_sensitivity_mean=0.80,
        vol_sensitivity_std=0.80,
        wealth_sigma=1.20,
        wealth_vol_corr=1.20,
        participation_vol_power=1.80,
        impact_activity_scale=2.50,
        impact_activity_clip=6.00,
        impact_crash_threshold=1.20,
        impact_crash_scale=2.00,
        impact_crash_power=2.00,
        asym_pi_scale=2.50,
        asym_pi_centered=True,
        down_ewma_decay=0.80,
        stoploss_universal_scale=0.0,
        stoploss_universal_threshold=0.005,
        fundamental_strength=0.0,
    )
    params.update(overrides)
    return Config(**params)


def candidate_configs() -> dict[str, Config]:
    return {
        # Q0: PのG33相当。長期では崩壊するはずの対照。
        "Q0_no_anchor_G33": base_config(fundamental_strength=0.0),
        # Q1-Q3: 弱いファンダメンタル平均回帰。短期SFをなるべく壊さない。
        "Q1_anchor_k002": base_config(fundamental_strength=0.002, fundamental_clip=0.003, fundamental_gap_scale=0.60),
        "Q2_anchor_k004": base_config(fundamental_strength=0.004, fundamental_clip=0.004, fundamental_gap_scale=0.55),
        "Q3_anchor_k006": base_config(fundamental_strength=0.006, fundamental_clip=0.005, fundamental_gap_scale=0.50),
        # Q4: leverageを少し補うため、ごく弱いmarket fearを加える。
        "Q4_anchor_k004_mf001": base_config(
            fundamental_strength=0.004, fundamental_clip=0.004, fundamental_gap_scale=0.55,
            stoploss_universal_scale=0.01,
        ),
    }


def metric_row(df: pd.DataFrame, label: str) -> dict:
    row = summarize_stylized_facts(df, label)
    sp = df["sp500"].astype(float).to_numpy()
    row["min_sp500_abs"] = float(df["sp500_abs"].min())
    row["max_sp500_abs"] = float(df["sp500_abs"].max())
    row["end_sp500_abs"] = float(df["sp500_abs"].iloc[-1])
    row["ann_return_approx"] = float((df["sp500_abs"].iloc[-1] / max(df["sp500_abs"].iloc[0], 1e-12)) ** (252 / len(df)) - 1)
    row["r_acf3_sp500"] = _acf(sp, 3)
    row["absacf3_sp500"] = _acf(np.abs(sp), 3)
    return row


def rolling_rows(df: pd.DataFrame, label: str, window: int = 1260) -> list[dict]:
    rows = []
    n = len(df)
    for start in range(0, max(n - window + 1, 1), window):
        sub = df.iloc[start:start+window].reset_index(drop=True)
        if len(sub) < window:
            continue
        r = metric_row(sub, f"{label}_w{start//window:02d}")
        r["window"] = start // window
        rows.append(r)
    return rows


def aggregate(rows: list[dict], label: str) -> dict:
    cols = [
        "std_sp500", "kurt_sp500", "absacf5_sp500", "absacf3_sp500",
        "leverage_sp500_lag1_20", "mean_sp500", "r_acf3_sp500",
        "min_sp500_abs", "end_sp500_abs", "ann_return_approx",
    ]
    out = {"label": label, "n_samples": len(rows)}
    for c in cols:
        vals = np.array([r[c] for r in rows if np.isfinite(r.get(c, np.nan))], dtype=float)
        out[c + "_mean"] = float(vals.mean()) if len(vals) else np.nan
        out[c + "_std"] = float(vals.std(ddof=0)) if len(vals) else np.nan
        out[c + "_min"] = float(vals.min()) if len(vals) else np.nan
        out[c + "_max"] = float(vals.max()) if len(vals) else np.nan
    return out


def score(agg: dict, target: dict) -> float:
    # 60年で重視: 崩壊回避 + ACF + leverage + kurtosis。標準化した粗いスコア。
    terms = []
    for key, scale in [
        ("std_sp500", 0.006),
        ("kurt_sp500", 8.0),
        ("absacf5_sp500", 0.12),
        ("leverage_sp500_lag1_20", 0.05),
    ]:
        terms.append(abs(agg[key + "_mean"] - target[key]) / scale)
    # price collapse penalty
    if agg["min_sp500_abs_min"] < 1.0:
        terms.append(10.0)
    if agg["ann_return_approx_mean"] < -0.05 or agg["ann_return_approx_mean"] > 0.20:
        terms.append(2.0)
    return float(sum(terms))


def append_memo(text: str) -> None:
    MEMO.parent.mkdir(parents=True, exist_ok=True)
    with MEMO.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")


def run_one(hist: pd.DataFrame, cfg: Config, label: str, seed: int, n_days: int) -> tuple[pd.DataFrame, dict, list[dict]]:
    cfg = Config(**{**asdict(cfg), "seed": seed, "n_days": n_days})
    outdir = RESULTS / label / f"seed_{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    gen, firms, investors, aux = simulate_market_gpu(hist, cfg, device=DEVICE)
    gen.to_csv(outdir / "generated_paths.csv", index=False)
    # firms/investors are useful but large enough; save only first seed per label.
    if seed == min(SEEDS):
        firms.to_csv(outdir / "firms.csv", index=False)
        investors.to_csv(outdir / "investors.csv", index=False)
        with open(outdir / "config.json", "w", encoding="utf-8") as f:
            json.dump(aux["config"], f, ensure_ascii=False, indent=2)
    full = metric_row(gen, f"{label}_seed{seed}_full")
    rolls = rolling_rows(gen, f"{label}_seed{seed}")
    return gen, full, rolls


SEEDS = [1, 2, 3]


def main() -> None:
    import time
    print(f"device: {DEVICE}")
    hist = pd.read_csv("output.csv")
    # 実データ全期間に合わせた約60年。model requires len(hist) >= n_days + 2.
    n_days = min(15120, len(hist) - 2)
    real_full = hist.tail(n_days).reset_index(drop=True)
    real_rolls = rolling_rows(real_full, "real")
    target = metric_row(hist.tail(1260).reset_index(drop=True), "real_tail1260")
    real_roll_agg = aggregate(real_rolls, "real_rolling")

    append_memo(f"# Q系 60年長期安定モデル 検証メモ\n\n- device: `{DEVICE}`\n- n_days: {n_days}\n- seeds: {SEEDS}\n- 目標: ACF・leverage・kurtosisをそこそこ再現しつつ、60年価格崩壊を避ける。\n")
    append_memo("## 実データ rolling 1260日の参考値\n\n" + pd.DataFrame([real_roll_agg]).to_markdown(index=False))

    all_full = []
    all_roll = []
    all_agg = []
    configs = candidate_configs()
    for label, cfg in configs.items():
        t0 = time.time()
        append_memo(f"## Round: {label}\n\n開始。主要パラメータ: `fundamental_strength={cfg.fundamental_strength}`, `asym_pi={cfg.asym_pi_scale}`, `mktfear={cfg.stoploss_universal_scale}`, `crash={cfg.impact_crash_scale}`")
        full_rows = []
        roll_rows = []
        for seed in SEEDS:
            print("running", label, "seed", seed)
            _, full, rolls = run_one(hist, cfg, label, seed, n_days)
            full_rows.append(full)
            roll_rows.extend(rolls)
        full_df = pd.DataFrame(full_rows)
        roll_df = pd.DataFrame(roll_rows)
        full_agg = aggregate(full_rows, label + "_full")
        roll_agg = aggregate(roll_rows, label + "_rolling")
        roll_agg["score_vs_tail"] = score(roll_agg, target)
        roll_agg["elapsed_sec"] = time.time() - t0
        all_full.extend(full_rows)
        all_roll.extend(roll_rows)
        all_agg.append(roll_agg)

        # 保存
        (RESULTS / label).mkdir(parents=True, exist_ok=True)
        full_df.to_csv(RESULTS / label / "full_metrics.csv", index=False)
        roll_df.to_csv(RESULTS / label / "rolling_metrics.csv", index=False)

        memo = "### 結果\n\n" + pd.DataFrame([roll_agg]).to_markdown(index=False)
        memo += "\n\n### 考察\n\n"
        if roll_agg["min_sp500_abs_min"] < 1.0:
            memo += "- 価格崩壊が発生。アンカー不足または下落フィードバック過剰。\n"
        else:
            memo += "- 60年で価格崩壊は回避。\n"
        memo += f"- rolling平均: std={roll_agg['std_sp500_mean']:.4f}, kurt={roll_agg['kurt_sp500_mean']:.2f}, absacf5={roll_agg['absacf5_sp500_mean']:.3f}, leverage={roll_agg['leverage_sp500_lag1_20_mean']:.3f}.\n"
        append_memo(memo)

    pd.DataFrame(all_full).to_csv(ROOT / "full_metrics_all.csv", index=False)
    pd.DataFrame(all_roll).to_csv(ROOT / "rolling_metrics_all.csv", index=False)
    agg_df = pd.DataFrame(all_agg).sort_values("score_vs_tail")
    agg_df.to_csv(ROOT / "comparison_longrun.csv", index=False)
    append_memo("## 総合比較\n\n" + agg_df.to_markdown(index=False))
    print(agg_df[["label", "std_sp500_mean", "kurt_sp500_mean", "absacf5_sp500_mean", "leverage_sp500_lag1_20_mean", "min_sp500_abs_min", "ann_return_approx_mean", "score_vs_tail"]].to_string(index=False))


if __name__ == "__main__":
    main()
