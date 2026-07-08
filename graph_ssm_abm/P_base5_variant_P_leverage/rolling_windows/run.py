"""
rolling_windows/run.py
======================
アプローチ2: 60年パスのローリングウィンドウ統計

longrun_drift の検証から判明:
  - exog_drift を +0.002/日 まで上げても価格崩壊を防げない
  - 崩壊は約2500日（10年）で発生 → 有効ウィンドウは最初の1〜2個のみ

本スクリプトでは:
  1. drift=0 で60年パスを生成（既存 G33_60y の結果を再利用）
  2. 1260日ウィンドウを順番に計算し、価格水準と統計量を確認
  3. 「いつから」統計量が崩れるかを記録
  4. 比較: 実データの対応ウィンドウ統計
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics import summarize_stylized_facts


BASE = Path(__file__).parent
REPO = Path(__file__).parent.parent.parent.parent

# G33_60y の既存結果を再利用（drift=0）
GEN_CSV = Path(__file__).parent.parent / "results_gpu" / "G33_60y" / "generated_paths.csv"


def main():
    if not GEN_CSV.exists():
        print(f"ERROR: {GEN_CSV} が見つかりません。generate_g33_60y.py を先に実行してください。")
        return

    gen = pd.read_csv(GEN_CSV)
    hist = pd.read_csv(REPO / "output.csv")

    window = 1260
    n_gen  = len(gen)
    n_hist = len(hist)
    n_windows = min(n_gen, n_hist) // window

    target = {
        "std_sp500": 0.0107,
        "kurt_sp500": 10.10,
        "absacf5_sp500": 0.198,
        "leverage_sp500_lag1_20": -0.043,
    }

    print(f"生成パス: {n_gen}行  実データ: {n_hist}行  ウィンドウ数: {n_windows}")
    print(f"(有効ウィンドウは価格崩壊前のみ = 最初の ~2 個)")

    gen_rows = []
    real_rows = []

    print(f"\n{'Win':>3}  {'実start':>10}  {'生成SP中央':>10}  {'生成std':>8}  "
          f"{'実std':>8}  {'生成kurt':>8}  {'実kurt':>8}  {'生成lev':>9}  {'実lev':>9}  {'有効':>5}")
    print("-" * 95)

    for i in range(n_windows):
        g_chunk = gen.iloc[i*window:(i+1)*window].reset_index(drop=True)
        h_chunk = hist.iloc[i*window:(i+1)*window].reset_index(drop=True)

        sp_abs_vals = g_chunk["sp500_abs"].tolist()
        sp_mid = sp_abs_vals[len(sp_abs_vals)//2]
        sp_end = sp_abs_vals[-1]
        valid = sp_end > 1.0   # 価格が崩壊していないか

        sg = summarize_stylized_facts(g_chunk, f"gen_w{i}")
        sr = summarize_stylized_facts(h_chunk, f"real_w{i}")

        real_date = h_chunk.iloc[0]["Date"] if "Date" in h_chunk.columns else str(i)
        std_g  = sg.get("std_sp500", float("nan"))
        kurt_g = sg.get("kurt_sp500", float("nan"))
        lev_g  = sg.get("leverage_sp500_lag1_20", float("nan"))
        std_r  = sr.get("std_sp500", float("nan"))
        kurt_r = sr.get("kurt_sp500", float("nan"))
        lev_r  = sr.get("leverage_sp500_lag1_20", float("nan"))

        flag = "✓" if valid else "✗崩壊"
        print(f"{i:>3}  {str(real_date)[:10]:>10}  {sp_mid:>10.2f}  "
              f"{std_g:>8.4f}  {std_r:>8.4f}  {kurt_g:>8.2f}  {kurt_r:>8.2f}  "
              f"{lev_g:>9.4f}  {lev_r:>9.4f}  {flag:>5}")

        sg["window"] = i
        sg["real_start"] = str(real_date)[:10]
        sg["sp_mid"] = sp_mid
        sg["sp_end"] = sp_end
        sg["valid"] = valid
        gen_rows.append(sg)

        sr["window"] = i
        sr["real_start"] = str(real_date)[:10]
        real_rows.append(sr)

    # 有効ウィンドウのみの統計
    valid_gen = [r for r in gen_rows if r.get("valid", False)]
    print(f"\n=== 有効ウィンドウ ({len(valid_gen)}個) の統計量 vs 目標 ===")
    print(f"  {'指標':30s} {'生成(mean)':>10s} {'実データ(mean)':>14s} {'目標':>8s}")
    print(f"  {'-'*68}")
    for k, tgt in target.items():
        g_vals = [r.get(k, float("nan")) for r in valid_gen]
        r_vals = [r.get(k, float("nan")) for r in real_rows[:len(valid_gen)]]
        g_vals = [v for v in g_vals if not math.isnan(v)]
        r_vals = [v for v in r_vals if not math.isnan(v)]
        gmu = sum(g_vals)/len(g_vals) if g_vals else float("nan")
        rmu = sum(r_vals)/len(r_vals) if r_vals else float("nan")
        print(f"  {k:30s} {gmu:>10.4f} {rmu:>14.4f} {tgt:>8.4f}")

    print(f"\n=== 全実データウィンドウ統計 (参考) ===")
    print(f"  {'指標':30s} {'mean':>8s} {'std_dev':>9s} {'min':>8s} {'max':>8s}")
    print(f"  {'-'*68}")
    for k in target:
        vals = [r.get(k, float("nan")) for r in real_rows]
        vals = [v for v in vals if not math.isnan(v)]
        if vals:
            mu = sum(vals)/len(vals)
            sigma = math.sqrt(sum((v-mu)**2 for v in vals)/max(len(vals)-1,1))
            print(f"  {k:30s} {mu:>8.4f} {sigma:>9.4f} {min(vals):>8.4f} {max(vals):>8.4f}")

    # CSV保存
    outdir = BASE / "results"
    outdir.mkdir(parents=True, exist_ok=True)

    keys = list(gen_rows[0].keys()) if gen_rows else []
    with open(outdir / "gen_windows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(gen_rows)

    keys_r = list(real_rows[0].keys()) if real_rows else []
    with open(outdir / "real_windows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys_r, extrasaction="ignore")
        w.writeheader()
        w.writerows(real_rows)

    print(f"\n結果保存先: {outdir}")


if __name__ == "__main__":
    main()
