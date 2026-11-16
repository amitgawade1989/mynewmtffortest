"""Excel scorecard in the layout of the framework's 'Dimension Deep-Dives' sheet.

Result and summary cells are formulas (=IF / =COUNTIFS) so changing a threshold in the
workbook updates PASS/FAIL. Scores are values copied from the evaluation outputs (source noted).
"""

from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

FONT = "Arial"
HEAD_FILL = PatternFill("solid", fgColor="1F3864")
SECTION_FILL = PatternFill("solid", fgColor="D9E2F3")
PASS_FILL = PatternFill("solid", fgColor="C6EFCE")
FAIL_FILL = PatternFill("solid", fgColor="FFC7CE")
MUTED_FILL = PatternFill("solid", fgColor="F2F2F2")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def font(**kw):
    return Font(name=FONT, **kw)


def style_header(ws, row, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = font(bold=True, color="FFFFFF")
        cell.fill = HEAD_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = BORDER


def write_table(ws, start_row, headers, rows, widths=None, number_format=None):
    """Write a simple table with a styled header; returns the next free row."""
    for c, h in enumerate(headers, 1):
        ws.cell(row=start_row, column=c, value=h)
    style_header(ws, start_row, len(headers))
    for i, r in enumerate(rows, start_row + 1):
        for c, v in enumerate(r, 1):
            cell = ws.cell(row=i, column=c, value=v)
            cell.font = font()
            cell.border = BORDER
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if number_format and isinstance(v, float):
                cell.number_format = number_format
    if widths:
        for c, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(c)].width = w
    return start_row + len(rows) + 2


def title(ws, text, subtitle=None):
    ws["A1"] = text
    ws["A1"].font = font(bold=True, size=14)
    if subtitle:
        ws["A2"] = subtitle
        ws["A2"].font = font(italic=True, color="595959")


def build_workbook(path, rows, perf, rel, per_question, run_info, retrieval_rows, reliability_tables):
    """Write the full scorecard workbook."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Scorecard"
    title(ws, "LLM Evaluation Scorecard - Document Comparison",
          f"Model under test: {run_info['under_test']} | Judge: {run_info['judge']} | "
          f"Performance run: {run_info['perf_run']} | Reliability run: {run_info['rel_run']} | {run_info['date']}")
    ws["A3"] = ("Legend: Score and Threshold are values from the evaluation (source: outputs/runs/<run>/eval*/summary.json). "
                "Result is a formula - edit a Threshold to re-grade. Direction 'lower' = lower score is better. "
                "Applies: ✓ required, O optional, blank = not applicable (framework 'Capability Coverage').")
    ws["A3"].font = font(size=9, color="595959")
    ws.merge_cells("A3:M3")

    headers = ["Dimension", "#", "Metric", "Sub-Metric", "Applies", "Implementation", "Status", "Score",
               "Direction", "Threshold", "Result", "n", "Key finding"]
    hr = 5
    for c, h in enumerate(headers, 1):
        ws.cell(row=hr, column=c, value=h)
    style_header(ws, hr, len(headers))
    first = hr + 1
    for i, r in enumerate(rows, first):
        values = [r["dimension"], r["num"], r["metric"], r["sub_metric"], r["applies"] or "-", r["implementation"],
                  r["status"], r["score"], r["direction"] if r["score"] is not None else None, r["threshold"], None,
                  r["n"], r["finding"]]
        for c, v in enumerate(values, 1):
            cell = ws.cell(row=i, column=c, value=v)
            cell.font = font()
            cell.border = BORDER
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.cell(row=i, column=8).number_format = "0.000"
        ws.cell(row=i, column=10).number_format = "0.000"
        ws.cell(row=i, column=11).value = (f'=IF(OR(H{i}="",J{i}=""),"",IF(I{i}="lower",IF(H{i}<=J{i},"PASS","FAIL"),'
                                           f'IF(H{i}>=J{i},"PASS","FAIL")))')
        ws.cell(row=i, column=11).font = font(bold=True)
        if r["status"].startswith(("N/A", "Out of scope", "Not run")):
            for c in range(1, len(headers) + 1):
                ws.cell(row=i, column=c).fill = MUTED_FILL
                ws.cell(row=i, column=c).font = font(color="7F7F7F")
    last = first + len(rows) - 1
    rng = f"K{first}:K{last}"
    ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"PASS"'], fill=PASS_FILL))
    ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"FAIL"'], fill=FAIL_FILL))
    for c, w in enumerate([12, 5, 20, 30, 8, 22, 26, 9, 10, 10, 9, 6, 70], 1):
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.freeze_panes = ws.cell(row=first, column=5)

    # summary with formulas
    s = last + 3
    ws.cell(row=s - 1, column=1, value="Summary (formulas)").font = font(bold=True, size=12)
    sum_headers = ["Dimension", "Evaluated", "PASS", "FAIL", "Pass rate"]
    for c, h in enumerate(sum_headers, 1):
        ws.cell(row=s, column=c, value=h)
    style_header(ws, s, len(sum_headers))
    for j, dim in enumerate(["Performance", "Reliability", "Safety"], s + 1):
        ws.cell(row=j, column=1, value=dim)
        ws.cell(row=j, column=2, value=f'=COUNTIFS(A{first}:A{last},A{j},H{first}:H{last},"<>")')
        ws.cell(row=j, column=3, value=f'=COUNTIFS(A{first}:A{last},A{j},K{first}:K{last},"PASS")')
        ws.cell(row=j, column=4, value=f'=COUNTIFS(A{first}:A{last},A{j},K{first}:K{last},"FAIL")')
        ws.cell(row=j, column=5, value=f'=IF((C{j}+D{j})=0,"",C{j}/(C{j}+D{j}))')
        ws.cell(row=j, column=5).number_format = "0.0%"
        for c in range(1, 6):
            ws.cell(row=j, column=c).font = font()
            ws.cell(row=j, column=c).border = BORDER

    # other sheets
    ws2 = wb.create_sheet("Performance Detail")
    title(ws2, "Performance detail", f"Source: outputs/runs/{run_info['perf_run']}/eval/summary.json")
    nr = write_table(ws2, 4, ["Sub-metric", "Value"],
                     [[k, v if isinstance(v, (int, float)) or v is None else str(v)] for k, v in perf["sub_metrics"].items()],
                     [40, 90], "0.000")
    nr = write_table(ws2, nr, ["Document comparison check", "Value"],
                     [[k, v if isinstance(v, (int, float)) or v is None else str(v)] for k, v in perf["document_comparison"].items()
                      if k != "relation_confusion"], None, "0.000")
    conf = perf["document_comparison"]["relation_confusion"]
    types = ["differential", "equivalence", "asymmetric"]
    preds = types + sorted({k.split("->")[1] for k in conf} - set(types))
    ws2.cell(row=nr, column=1, value="Relation confusion matrix (rows = gold, columns = predicted)").font = font(bold=True)
    nr = write_table(ws2, nr + 1, ["gold \\ predicted"] + preds,
                     [[g] + [conf.get(f"{g}->{p}", 0) for p in preds] for g in types])
    ws2.cell(row=nr, column=1, value="Breakdown by question type / reasoning / difficulty").font = font(bold=True)
    metrics = list(perf["headline"].keys())
    brows = [[f"{field}: {grp}", vals.get("n")] + [vals.get(m) for m in metrics]
             for field, groups in perf["breakdown"].items() for grp, vals in groups.items()]
    write_table(ws2, nr + 1, ["Group", "n"] + metrics, brows, None, "0.000")

    ws3 = wb.create_sheet("Retrieval")
    title(ws3, "Retrieval metrics (IDs only, no LLM)", "Relevance = golden evidence chunk IDs; absent sides excluded.")
    write_table(ws3, 4, retrieval_rows[0], retrieval_rows[1:], [34, 10, 10, 10, 10], "0.000")

    ws4 = wb.create_sheet("Reliability Detail")
    title(ws4, "Reliability detail", f"Source: outputs/runs/{run_info['rel_run']}/eval_reliability/summary.json")
    nr = 4
    for name, (hdr, body) in reliability_tables.items():
        ws4.cell(row=nr, column=1, value=name).font = font(bold=True, size=11)
        nr = write_table(ws4, nr + 1, hdr, body, None, "0.000")
    for c, w in enumerate([34, 16, 16, 16, 16, 16], 1):
        ws4.column_dimensions[get_column_letter(c)].width = w

    ws5 = wb.create_sheet("Per Question")
    title(ws5, "Per-question scores (golden set)", f"Source: outputs/runs/{run_info['perf_run']}/eval/scores.csv")
    write_table(ws5, 4, per_question[0], per_question[1:], [14, 14, 12, 10] + [12] * (len(per_question[0]) - 4), "0.000")
    ws5.freeze_panes = "B5"

    ws6 = wb.create_sheet("Run Info")
    title(ws6, "Run information")
    write_table(ws6, 3, ["Item", "Value"], [[k, str(v)] for k, v in run_info.items()], [30, 110])

    for sheet in wb.worksheets:
        sheet.sheet_view.showGridLines = False
    wb.calculation.fullCalcOnLoad = True          # Excel computes all formulas when the file is opened
    wb.save(path)
