from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
import torch

from model import Config
from model_gpu import simulate_market_gpu
from za_final_config import ZA_FINAL7_PARAMS

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = Path('/home/u00121/output.csv')


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Run logged final_model (ZA-FINAL7) trial for market_visualizer.')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--n-days', type=int, default=120)
    ap.add_argument('--n-firms', type=int, default=40)
    ap.add_argument('--n-investors', type=int, default=20)
    ap.add_argument('--n-sectors', type=int, default=4)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--out-dir', type=Path, default=None)
    ap.add_argument('--output-csv', type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument('--log-stride', type=int, default=1)
    ap.add_argument('--top-orders', type=int, default=200)
    ap.add_argument('--firm-limit', type=int, default=0)
    ap.add_argument('--investor-limit', type=int, default=0)
    return ap.parse_args()


def pick_device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def _excel_col(n: int) -> str:
    s = ''
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _cell_xml(value, row: int, col: int) -> str:
    ref = f'{_excel_col(col)}{row + 1}'
    if pd.isna(value):
        return f'<c r="{ref}"/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _sheet_xml(df: pd.DataFrame, max_rows: int = 100_000) -> str:
    df = df.head(max_rows).copy()
    rows = []
    header = ''.join(_cell_xml(c, 0, j) for j, c in enumerate(df.columns))
    rows.append(f'<row r="1">{header}</row>')
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        cells = ''.join(_cell_xml(v, i, j) for j, v in enumerate(r.tolist()))
        rows.append(f'<row r="{i+1}">{cells}</row>')
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + \
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + \
        ''.join(rows) + '</sheetData></worksheet>'


def write_xlsx_basic(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    names = []
    used = set()
    for name in sheets:
        base = ''.join(ch if ch not in '[]:*?/\\' else '_' for ch in name)[:31] or 'sheet'
        cand = base
        k = 1
        while cand in used:
            suffix = f'_{k}'
            cand = (base[:31-len(suffix)] + suffix)
            k += 1
        used.add(cand)
        names.append(cand)

    content_types = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>']
    for i in range(1, len(names)+1):
        content_types.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
    content_types.append('</Types>')

    wb_sheets = ''.join(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>' for i, name in enumerate(names, start=1))
    workbook = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + \
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>' + \
        wb_sheets + '</sheets></workbook>'
    rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + \
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'
    wb_rels_items = ''.join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, len(names)+1))
    wb_rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + \
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + wb_rels_items + '</Relationships>'

    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', ''.join(content_types))
        z.writestr('_rels/.rels', rels)
        z.writestr('xl/workbook.xml', workbook)
        z.writestr('xl/_rels/workbook.xml.rels', wb_rels)
        for i, (_, df) in enumerate(sheets.items(), start=1):
            z.writestr(f'xl/worksheets/sheet{i}.xml', _sheet_xml(df))


def write_outputs(out_dir: Path, paths: pd.DataFrame, firms: pd.DataFrame,
                  investors: pd.DataFrame, aux: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths.to_csv(out_dir / 'generated_paths.csv', index=False)
    firms.to_csv(out_dir / 'firms.csv', index=False)
    investors.to_csv(out_dir / 'investors.csv', index=False)

    tables = {
        'firm_states': aux.get('visual_firm_states', pd.DataFrame()),
        'investor_states': aux.get('visual_investor_states', pd.DataFrame()),
        'orders': aux.get('visual_orders', pd.DataFrame()),
        'true_graph_edges': aux.get('true_graph_edges', pd.DataFrame()),
        'subjective_graph_edges': aux.get('subjective_graph_edges', pd.DataFrame()),
        'firm_snapshots': aux.get('firm_snapshots', pd.DataFrame()),
    }
    for name, df in tables.items():
        df.to_csv(out_dir / f'{name}.csv', index=False)

    with open(out_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(aux.get('config', {}), f, ensure_ascii=False, indent=2, default=str)

    xlsx_sheets = {'market_paths': paths, 'firms': firms, 'investors': investors, **tables}
    try:
        # openpyxl が入っている環境では通常の pandas writer を使う。
        with pd.ExcelWriter(out_dir / 'market_log.xlsx', engine='openpyxl') as xw:
            for name, df in xlsx_sheets.items():
                df.head(100_000).to_excel(xw, sheet_name=name[:31], index=False)
        err = out_dir / 'xlsx_error.txt'
        if err.exists():
            err.unlink()
    except Exception as e:
        # 依存関係やExcel制限で失敗した場合だけ、標準ライブラリの簡易writerへフォールバック。
        try:
            write_xlsx_basic(out_dir / 'market_log.xlsx', xlsx_sheets)
            (out_dir / 'xlsx_error.txt').write_text(f'pandas/openpyxl failed, fallback writer used: {e}', encoding='utf-8')
        except Exception as e2:
            (out_dir / 'xlsx_error.txt').write_text(f'pandas/openpyxl failed: {e}\nfallback writer failed: {e2}', encoding='utf-8')


def main() -> None:
    args = parse_args()
    hist = pd.read_csv(args.output_csv)
    device = pick_device(args.device)
    out_dir = args.out_dir or ROOT / 'results' / f'trial_seed{args.seed:02d}'

    cfg = Config(**{
        **ZA_FINAL7_PARAMS,
        'seed': args.seed,
        'n_days': args.n_days,
        'n_firms': args.n_firms,
        'n_investors': args.n_investors,
        'n_sectors': args.n_sectors,
        'ba_m': min(3, max(1, args.n_firms - 1)),
        'visual_log_enabled': True,
        'visual_log_stride': args.log_stride,
        'visual_log_top_orders': args.top_orders,
        'visual_log_firm_limit': args.firm_limit,
        'visual_log_investor_limit': args.investor_limit,
    })

    print(f'Running logged final_model (ZA-FINAL7) trial on {device}: {cfg.n_days}d, {cfg.n_firms} firms, {cfg.n_investors} investors')
    paths, firms, investors, aux = simulate_market_gpu(hist, cfg, device=device)
    write_outputs(out_dir, paths, firms, investors, aux)
    print(f'DONE: {out_dir}')
    print('Next:')
    print(f'  python /home/u00121/presentation/market_visualizer/animate_market.py --result-dir {out_dir}')


if __name__ == '__main__':
    main()
