"""Self-contained HTML report (inline SVG charts, no external libraries).

Colours follow a validated palette (categorical slots 1-2, sequential blue, fixed status colours)
with a selected dark mode. Every chart has hover tooltips and a data table behind a <details>.
"""

import html
import json

SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
       "#1c5cab", "#184f95", "#104281", "#0d366b"]

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--good:#0ca30c;--critical:#d03b3b;
--goodtext:#006300;--badtext:#b42323}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;
--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--s1:#3987e5;--s2:#d95926;--goodtext:#0ca30c;--badtext:#e66767}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--goodtext:#0ca30c;--badtext:#e66767}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 8px}
.sub{color:var(--ink2);margin:0 0 16px}.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:16px;margin:12px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}.tile .v{font-size:28px;font-weight:600}
.tile .l{color:var(--ink2);font-size:13px}.note{color:var(--ink2);font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--ink2);font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums}
.chip{white-space:nowrap;font-weight:600}.chip.pass{color:var(--goodtext)}.chip.fail{color:var(--badtext)}.chip.na{color:var(--muted);font-weight:400}
details{margin-top:8px}summary{cursor:pointer;color:var(--ink2);font-size:13px}.scroll{overflow-x:auto}
svg text{fill:var(--ink2);font-size:12px}svg .ink{fill:var(--ink)}svg .muted{fill:var(--muted)}
.legend{display:flex;gap:16px;font-size:13px;color:var(--ink2);margin:4px 0 8px}.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.heat td{text-align:center;font-variant-numeric:tabular-nums;border:2px solid var(--surface);border-radius:6px}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:8px;
padding:6px 10px;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.15);display:none;z-index:10;max-width:320px}
[data-tip]{cursor:default}
"""

JS = """
const tip=document.getElementById('tip');
document.addEventListener('mousemove',e=>{const t=e.target.closest('[data-tip]');
 if(!t){tip.style.display='none';return;} tip.textContent=t.dataset.tip; tip.style.display='block';
 const x=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8), y=Math.min(e.clientY+14,innerHeight-tip.offsetHeight-8);
 tip.style.left=x+'px'; tip.style.top=y+'px';});
"""


def esc(x):
    return html.escape(str(x))


def fmt(x, d=3):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def chip(score, threshold, direction):
    if score is None or threshold is None:
        return '<span class="chip na">—</span>'
    ok = score <= threshold if direction == "lower" else score >= threshold
    return f'<span class="chip {"pass" if ok else "fail"}">{"✓ PASS" if ok else "✕ FAIL"}</span>'


def seq_color(value, vmax):
    """Sequential blue step for value in [0, vmax] (lightest = 0); text colour that stays readable."""
    i = 0 if not vmax else min(len(SEQ) - 1, round(value / vmax * (len(SEQ) - 1)))
    return SEQ[i], ("#ffffff" if i >= 6 else "#0b0b0b")


# ------------------------------------------------------------------ charts

def score_bars(rows):
    """Horizontal bars: score per evaluated sub-metric (0-1), with the threshold as a tick."""
    rows = [r for r in rows if r["score"] is not None and r["direction"] == "higher"]
    left, width, bar_h, gap = 250, 420, 14, 12
    short = lambda s: s if len(s) <= 30 else s[:28].rstrip(" /") + "…"     # full name stays in the tooltip
    h = len(rows) * (bar_h + gap) + 30
    parts = [f'<svg viewBox="0 0 {left + width + 120} {h}" width="100%" role="img" aria-label="Scores by sub-metric">']
    for t in (0, .25, .5, .75, 1):
        x = left + t * width
        parts.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{h - 22}" stroke="var(--grid)" stroke-width="1"/>'
                     f'<text x="{x}" y="{h - 6}" text-anchor="middle" class="muted">{t:g}</text>')
    for i, r in enumerate(rows):
        y = i * (bar_h + gap) + 4
        w = max(2, r["score"] * width)
        ok = r["threshold"] is None or r["score"] >= r["threshold"]
        tipv = f"{r['num']} {r['sub_metric']}: {r['score']:.3f} (threshold {fmt(r['threshold'])})"
        parts.append(f'<text x="{left - 8}" y="{y + 11}" text-anchor="end" class="ink">{esc(r["num"])} {esc(short(r["sub_metric"]))}</text>')
        parts.append(f'<rect data-tip="{esc(tipv)}" x="{left}" y="{y - 4}" width="{width}" height="{bar_h + 8}" fill="transparent"/>')
        parts.append(f'<path d="M{left},{y} h{w - 4} a4,4 0 0 1 4,4 v{bar_h - 8} a4,4 0 0 1 -4,4 h-{w - 4} z" fill="var(--s1)" pointer-events="none"/>')
        if r["threshold"] is not None:
            tx = left + r["threshold"] * width
            parts.append(f'<line x1="{tx}" y1="{y - 3}" x2="{tx}" y2="{y + bar_h + 3}" stroke="var(--ink)" stroke-width="2" pointer-events="none"/>')
        parts.append(f'<text x="{left + width + 8}" y="{y + 11}" class="ink">{r["score"]:.3f} '
                     f'<tspan style="fill:var({"--goodtext" if ok else "--badtext"})">{"✓" if ok else "✕"}</tspan></text>')
    parts.append("</svg>")
    return "".join(parts)


def line_chart(xs, series, y_label):
    """Lines over a small set of x categories; series = [(name, css_var, values)]."""
    left, top, width, height = 50, 10, 560, 220
    step = width / (len(xs) - 1)
    parts = [f'<svg viewBox="0 0 {left + width + 150} {top + height + 40}" width="100%" role="img" aria-label="{esc(y_label)}">']
    for t in (0, .25, .5, .75, 1):
        y = top + height - t * height
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left + width}" y2="{y}" stroke="var(--grid)"/>'
                     f'<text x="{left - 8}" y="{y + 4}" text-anchor="end" class="muted">{t:g}</text>')
    for i, x in enumerate(xs):
        parts.append(f'<text x="{left + i * step}" y="{top + height + 18}" text-anchor="middle" class="muted">{esc(x)}</text>')
    end_labels = []
    for name, var, vals in series:
        pts = [(left + i * step, top + height - v * height) for i, v in enumerate(vals)]
        parts.append(f'<polyline points="{" ".join(f"{x},{y}" for x, y in pts)}" fill="none" stroke="var({var})" stroke-width="2"/>')
        for (x, y), v, k in zip(pts, vals, xs):
            parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="var({var})" stroke="var(--surface)" stroke-width="2"/>'
                         f'<circle data-tip="{esc(name)} {esc(k)}: {v:.3f}" cx="{x}" cy="{y}" r="12" fill="transparent"/>')
        end_labels.append([pts[-1][1] + 4, f"{name} {vals[-1]:.2f}"])
    end_labels.sort()
    for i in range(1, len(end_labels)):                  # keep end labels at least 16px apart
        end_labels[i][0] = max(end_labels[i][0], end_labels[i - 1][0] + 16)
    for y, text in end_labels:
        parts.append(f'<text x="{left + width + 12}" y="{y}" class="ink">{esc(text)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def grouped_bars(cats, series):
    """Vertical grouped bars (0-1). series = [(name, css_var, values)]."""
    left, top, height, group_w = 40, 10, 220, 170          # wide groups keep the SVG near 1:1 scale
    bw = min(36, (group_w - 60) / len(series))
    width = len(cats) * group_w
    parts = [f'<svg viewBox="0 0 {left + width + 20} {top + height + 50}" width="100%" role="img" aria-label="Clean vs perturbed accuracy">']
    for t in (0, .5, 1):
        y = top + height - t * height
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left + width}" y2="{y}" stroke="var(--grid)"/>'
                     f'<text x="{left - 6}" y="{y + 4}" text-anchor="end" class="muted">{t:g}</text>')
    for i, c in enumerate(cats):
        gx = left + i * group_w + (group_w - len(series) * (bw + 2)) / 2
        for j, (name, var, vals) in enumerate(series):
            v = vals[i] or 0
            hgt = max(2, v * height)
            x, y = gx + j * (bw + 2), top + height - hgt
            parts.append(f'<path d="M{x},{top + height} v-{hgt - 4} a4,4 0 0 1 4,-4 h{bw - 8} a4,4 0 0 1 4,4 v{hgt - 4} z" fill="var({var})"/>'
                         f'<rect data-tip="{esc(c)} - {esc(name)}: {v:.3f}" x="{x - 1}" y="{top}" width="{bw + 2}" height="{height}" fill="transparent"/>')
        parts.append(f'<text x="{left + i * group_w + group_w / 2}" y="{top + height + 18}" text-anchor="middle" class="muted">{esc(c)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def heat_table(row_labels, col_labels, values, row_title, fmt_fn=lambda v: f"{v}"):
    """Heatmap as an HTML table with sequential blue cells (values in each cell, tooltip on hover)."""
    vmax = max([v for r in values for v in r if v is not None] or [1])
    out = [f'<table class="heat"><tr><th>{esc(row_title)}</th>' + "".join(f"<th>{esc(c)}</th>" for c in col_labels) + "</tr>"]
    for rl, rv in zip(row_labels, values):
        out.append(f"<tr><th>{esc(rl)}</th>")
        for cl, v in zip(col_labels, rv):
            if v is None:
                out.append("<td>-</td>")
                continue
            bg, fg = seq_color(v, vmax)
            out.append(f'<td data-tip="{esc(rl)} / {esc(cl)}: {fmt_fn(v)}" style="background:{bg};color:{fg}">{fmt_fn(v)}</td>')
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def data_table(headers, rows):
    body = "".join("<tr>" + "".join(f'<td class="{"num" if isinstance(v, (int, float)) else ""}">{esc(fmt(v))}</td>' for v in r)
                   + "</tr>" for r in rows)
    return (f'<details><summary>Show data table</summary><div class="scroll"><table><tr>'
            + "".join(f"<th>{esc(h)}</th>" for h in headers) + f"</tr>{body}</table></div></details>")


# ------------------------------------------------------------------ page

def build_html(path, rows, perf, rel, run_info, retrieval):
    """Write the full report page."""
    ev = [r for r in rows if r["score"] is not None and r["threshold"] is not None]
    passed = lambda dim: sum((r["score"] <= r["threshold"]) if r["direction"] == "lower" else (r["score"] >= r["threshold"])
                             for r in ev if r["dimension"] == dim)
    total = lambda dim: sum(r["dimension"] == dim for r in ev)
    sm, dc = perf["sub_metrics"], perf["document_comparison"]
    rec5 = perf["retrieval"].get("combined.child_recall@5", {}).get("mean")
    cost = sm.get("P7_total_cost_usd")
    tiles = [(f"{passed('Performance')}/{total('Performance')}", "Performance sub-metrics passed"),
             (f"{passed('Reliability')}/{total('Reliability')}", "Reliability sub-metrics passed"),
             (fmt(rec5), "Retrieval child recall@5"),
             (f"{sm['P1_relation_exact_match'] * 100:.0f}%", "Comparison type correct"),
             (f"{dc['equivalence_false_difference_rate'] * 100:.0f}%", "Invented differences (equivalence Qs)"),
             (f"${cost:.3f}" if cost is not None else "-", f"Cost of {perf['coverage']['evaluated']} answers")]

    sc_rows = "".join(
        f"<tr><td>{esc(r['dimension'])}</td><td>{esc(r['num'])}</td><td>{esc(r['sub_metric'])}</td><td>{esc(r['applies'] or '-')}</td>"
        f"<td>{esc(r['implementation'])}</td><td class='num'>{fmt(r['score'])}</td><td class='num'>{fmt(r['threshold'])}</td>"
        f"<td>{chip(r['score'], r['threshold'], r['direction']) if r['status'].startswith('Evaluated') else '<span class="chip na">' + esc(r['status']) + '</span>'}</td>"
        f"<td>{esc(r['finding'])}</td></tr>" for r in rows)

    ks, kcols = retrieval["ks"], [f"@{k}" for k in retrieval["ks"]]
    child = [perf["retrieval"][f"combined.child_recall@{k}"]["mean"] for k in ks]
    parent = [perf["retrieval"][f"combined.parent_recall@{k}"]["mean"] for k in ks]
    both = [perf["retrieval"][f"both_sides.child_hit@{k}"]["mean"] for k in ks]

    conf = dc["relation_confusion"]
    types = ["differential", "equivalence", "asymmetric"]
    preds = types + sorted({k.split("->")[1] for k in conf} - set(types))
    conf_vals = [[conf.get(f"{g}->{p}", 0) for p in preds] for g in types]

    rd = rel.get("detail", {})
    cats, clean_v, pert_v = [], [], []
    for k, d in rd.get("R2_typo_noise", {}).get("by_perturbation", {}).items():
        cats.append(k.replace("typo_", "typo: ")); clean_v.append(d["clean_accuracy"]); pert_v.append(d["accuracy"])
    for k, d in rd.get("R2_adversarial", {}).get("accuracy_delta_by_technique", {}).items():
        cats.append(k.replace("adv_", "adv: ").replace("_", " ")); clean_v.append(d["clean_accuracy"]); pert_v.append(d["accuracy"])

    lc = rd.get("R2_long_context", {})
    lengths = sorted({int(k.split("|")[0]) for k in lc.get("heatmap", {})})
    positions = ["start", "middle", "end"]
    lc_vals = [[lc["heatmap"].get(f"{L}|{p}") for p in positions] for L in lengths]

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Document Comparison Scorecard</title><style>{CSS}</style></head><body><main>
<h1>Document comparison evaluation</h1>
<p class="sub">Model under test <b>{esc(run_info['under_test'])}</b> · judge {esc(run_info['judge'])} · golden set of {perf['coverage']['evaluated']} questions ·
{rel.get('n_variants', 0)} reliability variants · {esc(run_info['date'])}</p>
<div class="tiles">{''.join(f'<div class="card tile"><div class="v">{esc(v)}</div><div class="l">{esc(l)}</div></div>' for v, l in tiles)}</div>

<h2>Scores against thresholds</h2>
<div class="card"><p class="note">Bar = score (0-1); vertical tick = pass threshold; ✓/✕ = pass/fail. Fabricated citations (lower is better) is in the table.</p>
{score_bars(rows)}</div>

<h2>Scorecard (framework: {esc(run_info['capability'])})</h2>
<div class="card scroll"><table><tr><th>Dimension</th><th>#</th><th>Sub-metric</th><th>Applies</th><th>Method</th><th>Score</th><th>Threshold</th><th>Result</th><th>Key finding</th></tr>
{sc_rows}</table></div>

<h2>Retrieval quality</h2>
<div class="card"><div class="legend"><span><i style="background:var(--s1)"></i>Child recall (combined)</span><span><i style="background:var(--s2)"></i>Both documents hit</span></div>
{line_chart(kcols, [("Child recall", "--s1", child), ("Both docs hit", "--s2", both)], "Retrieval recall at k")}
{data_table(["k", "child recall", "parent recall", "both documents hit"], [[k, c, p, b] for k, c, p, b in zip(kcols, child, parent, both)])}</div>

<h2>Comparison type: gold vs predicted</h2>
<div class="card scroll"><p class="note">Rows = correct type, columns = type the model chose. Off-diagonal cells are errors.</p>
{heat_table(types, preds, conf_vals, "gold \\ predicted")}</div>

<h2>Robustness: clean vs perturbed accuracy</h2>
<div class="card"><div class="legend"><span><i style="background:var(--s1)"></i>Clean question</span><span><i style="background:var(--s2)"></i>Perturbed / adversarial</span></div>
{grouped_bars(cats, [("Clean", "--s1", clean_v), ("Perturbed", "--s2", pert_v)]) if cats else ''}
{data_table(["test", "clean accuracy", "perturbed accuracy"], [[c, a, b] for c, a, b in zip(cats, clean_v, pert_v)])}</div>

<h2>Long context: accuracy by length and evidence position</h2>
<div class="card scroll">{heat_table([f"{L:,} tokens" for L in lengths], positions, lc_vals, "length \\ position", lambda v: f"{v:.2f}") if lengths else ''}
<p class="note">Slope per 10x length: {fmt(lc.get('slope_per_10x_length'))} · lost-in-the-middle gap: {fmt(lc.get('lost_in_middle_gap'))}</p></div>

<h2>Notes</h2>
<div class="card note"><ul>
<li>Every sub-metric uses the implementation named in the framework (Deterministic / LLM Judge / hybrid). Judge model differs from the model under test.</li>
<li>P6 uses sentence-level NLI; long multi-fact claims are a known limit of this deterministic method.</li>
<li>R3 format results are measured inside the pipeline's JSON answer field, which may make line-based formats (bullets) harder.</li>
<li>Rows marked N/A are blank for this capability in the framework; Safety rows other than S5 were out of scope for this run.</li>
</ul></div>
</main><div id="tip" role="tooltip"></div><script>{JS}</script></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
