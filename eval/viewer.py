"""Self-contained HTML trace viewer.

Generates one .html file per EvalTrace that can be opened in any browser.
No external dependencies, no server, no JavaScript — HTML+CSS only.
<details>/<summary> handles all expand/collapse natively.

Usage (from project root):
    python -m eval.eval_cli view <path/to/eval_trace.json>

The generated file is written to reports/<case_id>/<run_id>.html.
"""

from __future__ import annotations

import html as _html_mod
import json
from pathlib import Path
from typing import Any

from eval.eval_trace import EvalTrace

REPORTS_DIR = Path(__file__).parent.parent / "reports"
_MAX_CONTENT = 5000   # chars; long tool results are truncated in the viewer


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _esc(v: Any) -> str:
    return _html_mod.escape(str(v) if not isinstance(v, str) else v)


def _json_pre(data: Any) -> str:
    text = json.dumps(data, indent=2, default=str, ensure_ascii=False)
    if len(text) > _MAX_CONTENT:
        text = text[:_MAX_CONTENT] + f"\n\n… [{len(text) - _MAX_CONTENT} more chars truncated]"
    return f'<pre class="json">{_esc(text)}</pre>'


def _badge(text: str, cls: str) -> str:
    return f'<span class="badge badge-{_esc(cls)}">{_esc(text)}</span>'


def _card(title_html: str, body_html: str, *, open_: bool = False, header_cls: str = "") -> str:
    open_attr = " open" if open_ else ""
    hcls = f" {header_cls}" if header_cls else ""
    return f"""
<details class="card"{open_attr}>
  <summary class="card-hd{hcls}">{title_html}</summary>
  <div class="card-body">{body_html}</div>
</details>"""


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_header(trace: EvalTrace) -> str:
    ok = trace.case_passed
    cls = "pass" if ok else "fail"
    label = "PASS" if ok else "FAIL"
    tokens = trace.total_tokens
    meta = (
        f"run&nbsp;{_esc(trace.run_id[:12])}…"
        f" &nbsp;|&nbsp; stopped:&nbsp;<b>{_esc(trace.stopped_reason)}</b>"
        f" &nbsp;|&nbsp; ${trace.cost_usd:.6f}"
        f" &nbsp;|&nbsp; {trace.wall_time_ms}&nbsp;ms"
        f" &nbsp;|&nbsp; in={tokens.get('input', 0)}&nbsp;out={tokens.get('output', 0)}"
    )
    return f"""
<div class="banner banner-{cls}">
  <span class="verdict verdict-{cls}">{label}</span>
  <div class="banner-id">{_esc(trace.case_id)}</div>
  <div class="banner-meta">{meta}</div>
</div>"""


def _render_assertions(trace: EvalTrace) -> str:
    results = trace.assertion_results
    n_fail = sum(1 for r in results if not r["passed"])
    n_total = len(results)

    if n_fail:
        title = f'{_badge("✘ " + str(n_fail) + " failed", "fail")} &nbsp; of {n_total} assertions'
        hcls = "hd-fail"
    else:
        title = f'{_badge("✔ all passed", "pass")} &nbsp; {n_total} assertions'
        hcls = "hd-pass"

    if not results:
        body = "<em>No assertions defined.</em>"
    else:
        sorted_r = sorted(results, key=lambda r: (r["passed"], r["assertion_type"]))
        rows = ""
        for r in sorted_r:
            rc = "row-pass" if r["passed"] else "row-fail"
            icon = "✔" if r["passed"] else "✘"
            rows += (
                f'<tr class="{rc}">'
                f'<td class="ai">{icon}</td>'
                f'<td class="at">{_esc(r["assertion_type"])}</td>'
                f'<td class="ar">{_esc(r["reason"])}</td>'
                f'</tr>'
            )
        body = (
            '<table><thead><tr><th></th><th>Assertion</th><th>Reason</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>'
        )
    return _card(title, body, open_=n_fail > 0, header_cls=hcls)


def _render_metrics(trace: EvalTrace) -> str:
    if not trace.metric_results:
        return ""
    rows = ""
    for m in trace.metric_results:
        name = m.get("metric_name", "?")
        reason = m.get("reason", "")
        mp = m.get("passed")
        if mp is False:
            row_cls, badge_html = "mrow-warn", _badge("WARN", "warn")
        elif mp is True:
            row_cls, badge_html = "mrow-ok", _badge("ok", "pass")
        else:
            row_cls, badge_html = "mrow-info", ""

        val = m.get("value")
        detail = ""
        if isinstance(val, dict):
            detail = f'<details class="mdetail"><summary>details</summary>{_json_pre(val)}</details>'

        rows += (
            f'<div class="mrow {row_cls}">'
            f'<span class="mn">{_esc(name)}</span>'
            f'{badge_html}'
            f'<span class="mr">{_esc(reason)}</span>'
            f'{detail}'
            f'</div>'
        )
    return _card("Metrics", rows)


def _render_judge(trace: EvalTrace) -> str:
    jv = trace.judge_verdict
    if jv is None:
        return ""

    verdict = jv.get("verdict", "?")
    score = float(jv.get("score", 0.0))
    passed = bool(jv.get("passed", False))
    rubric_id = jv.get("rubric_id", "?")
    model = jv.get("model", "?")
    rationale = jv.get("rationale", "")
    flags = jv.get("flags", [])
    error = jv.get("error")

    badge_cls = "pass" if passed else ("error" if verdict == "error" else "fail")
    score_cls = "jscore-pass" if passed else "jscore-fail"

    flags_html = ""
    if flags:
        flags_html = f'<div class="jflags">&#9888; Failed dimensions: {_esc(", ".join(flags))}</div>'
    err_html = f'<div class="jerr">Error: {_esc(error)}</div>' if error else ""

    title = (
        f'Judge &nbsp; {_badge(verdict.upper(), badge_cls)}'
        f' &nbsp; rubric:&nbsp;{_esc(rubric_id)} &nbsp;|&nbsp; model:&nbsp;{_esc(model)}'
    )
    body = (
        f'<div class="jrow">'
        f'<span class="jscore {score_cls}">{score:.2f}</span>'
        f' &nbsp; {_badge(verdict.upper(), badge_cls)}'
        f'</div>'
        f'<div class="jrationale">{_esc(rationale)}</div>'
        f'{flags_html}{err_html}'
    )
    return _card(title, body, open_=not passed)


def _render_msg(msg: dict[str, Any], step: int) -> str:
    role = msg.get("role", "?")

    if role == "system":
        content = msg.get("content", "")
        inner = (
            f'<details><summary class="ms ms-system"><span class="sn">{step}</span>'
            f'<b>SYSTEM</b> <span class="dim">prompt — click to expand</span></summary>'
            f'<div class="mb"><div class="tc">{_esc(content)}</div></div></details>'
        )
        return f'<div class="msg">{inner}</div>'

    if role == "user":
        content = msg.get("content", "")
        return (
            f'<div class="msg">'
            f'<div class="ms ms-user"><span class="sn">{step}</span><b>USER</b></div>'
            f'<div class="mb"><div class="tc">{_esc(content)}</div></div>'
            f'</div>'
        )

    if role == "assistant":
        text = msg.get("text", "")
        tool_calls = msg.get("tool_calls", [])
        lat = msg.get("latency_ms", 0)

        names_preview = ""
        if tool_calls:
            names = [_esc(tc.get("name", "?")) for tc in tool_calls]
            names_preview = f' <span class="dim">&rarr; {", ".join(names)}</span>'

        text_html = f'<div class="tc" style="margin-bottom:6px">{_esc(text)}</div>' if text.strip() else ""

        calls_html = ""
        for tc in tool_calls:
            tc_name = tc.get("name", "?")
            tc_args = tc.get("args") or {}
            tc_id = tc.get("id", "")[:12]
            finish_note = ' <span class="badge badge-info">terminal</span>' if tc_name == "finish" else ""
            calls_html += (
                f'<div class="tcall">'
                f'<details><summary>&#128295; {_esc(tc_name)}'
                f'{finish_note}'
                f' <span class="dim">{_esc(tc_id)}</span></summary>'
                f'{_json_pre(tc_args)}'
                f'</details></div>'
            )

        summary = (
            f'<summary class="ms ms-assistant">'
            f'<span class="sn">{step}</span><b>ASSISTANT</b>'
            f'{names_preview}'
            f'<span class="lat">{lat}&nbsp;ms</span>'
            f'</summary>'
        )
        body = f'<div class="mb">{text_html}<div class="tcalls">{calls_html}</div></div>'
        return f'<div class="msg"><details open>{summary}{body}</details></div>'

    return f'<div class="msg"><div class="mb"><em>Unknown role: {_esc(role)}</em></div></div>'


def _render_timeline(trace: EvalTrace) -> str:
    msgs = trace.messages
    if not msgs:
        return _card("Message Timeline", "<em>No messages.</em>", open_=True)
    items = "".join(_render_msg(m, i + 1) for i, m in enumerate(msgs))
    return _card(f"Message Timeline &nbsp; <span class='dim'>({len(msgs)} messages)</span>", items, open_=True)


# ---------------------------------------------------------------------------
# Fix: the tool message renderer above has a copy-paste issue. Rewrite cleanly.
# ---------------------------------------------------------------------------

def _render_tool_msg(msg: dict[str, Any], step: int) -> str:
    tool_name = msg.get("name", "?")
    content = msg.get("content")
    lat = msg.get("latency_ms", 0)
    is_err = isinstance(content, dict) and "error" in content
    err_badge = _badge("ERROR", "fail") if is_err else ""

    if content == "ok" and tool_name == "finish":
        body_html = '<span class="dim">finish&nbsp;acknowledged</span>'
    else:
        body_html = _json_pre(content)

    return (
        f'<div class="msg">'
        f'<details>'
        f'<summary class="ms ms-tool">'
        f'<span class="sn">{step}</span>'
        f'<b>TOOL</b> <span class="tn">{_esc(tool_name)}</span>'
        f' {err_badge}'
        f'<span class="lat">{lat}&nbsp;ms</span>'
        f'</summary>'
        f'<div class="mb">{body_html}</div>'
        f'</details>'
        f'</div>'
    )


def _render_msg_dispatch(msg: dict[str, Any], step: int) -> str:
    role = msg.get("role", "?")
    if role == "tool":
        return _render_tool_msg(msg, step)
    return _render_msg(msg, step)


def _fixed_render_timeline(trace: EvalTrace) -> str:
    msgs = trace.messages
    if not msgs:
        return _card("Message Timeline", "<em>No messages.</em>", open_=True)
    items = "".join(_render_msg_dispatch(m, i + 1) for i, m in enumerate(msgs))
    return _card(
        f"Message Timeline &nbsp; <span class='dim'>({len(msgs)} messages)</span>",
        items,
        open_=True,
    )


# ---------------------------------------------------------------------------
# CSS (inline, no external resources)
# ---------------------------------------------------------------------------

_CSS = """
:root {
  --c-pass:#15803d; --c-fail:#b91c1c; --c-warn:#b45309;
  --c-info:#0369a1; --c-code-bg:#0f172a; --c-code-fg:#e2e8f0;
  --bg-pass:#f0fdf4; --bg-fail:#fef2f2; --bg-warn:#fffbeb;
}
*{box-sizing:border-box}
body{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#f1f5f9; color:#1e293b; margin:0; padding:16px;
  font-size:14px; line-height:1.5;
}
.container{max-width:960px;margin:0 auto}

/* Verdict banner */
.banner{border-radius:8px;padding:14px 18px;margin-bottom:10px;border-left:6px solid}
.banner-pass{background:var(--bg-pass);border-color:var(--c-pass)}
.banner-fail{background:var(--bg-fail);border-color:var(--c-fail)}
.verdict{display:inline-block;color:#fff;padding:1px 10px;border-radius:9999px;
  font-weight:800;font-size:.8em;letter-spacing:.08em}
.verdict-pass{background:var(--c-pass)}
.verdict-fail{background:var(--c-fail)}
.banner-id{font-size:1.25em;font-weight:700;margin:4px 0 2px}
.banner-meta{color:#64748b;font-size:.78em;font-family:monospace}

/* Cards */
.card{background:#fff;border-radius:8px;border:1px solid #e2e8f0;
  margin-bottom:10px;overflow:hidden}
.card-hd{padding:9px 14px;border-bottom:1px solid #e2e8f0;font-size:.78em;
  font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#64748b;
  cursor:pointer;display:flex;align-items:center;gap:8px;
  list-style:none;user-select:none}
.card-hd::-webkit-details-marker{display:none}
.hd-fail{color:var(--c-fail)}
.hd-pass{color:var(--c-pass)}
.card-body{padding:10px 14px}

/* Badges */
.badge{display:inline-block;padding:1px 8px;border-radius:9999px;
  font-size:.73em;font-weight:700;letter-spacing:.04em}
.badge-pass{background:var(--c-pass);color:#fff}
.badge-fail{background:var(--c-fail);color:#fff}
.badge-warn{background:var(--c-warn);color:#fff}
.badge-error{background:#6b21a8;color:#fff}
.badge-info{background:var(--c-info);color:#fff}

/* Assertions */
table{width:100%;border-collapse:collapse}
th{padding:5px 10px;text-align:left;font-size:.8em;color:#64748b;
  border-bottom:2px solid #e2e8f0}
td{padding:6px 10px;border-bottom:1px solid #f1f5f9;vertical-align:top}
.row-pass td{background:var(--bg-pass)}
.row-fail td{background:var(--bg-fail)}
.ai{font-weight:800;width:1.5em}
.row-pass .ai{color:var(--c-pass)}
.row-fail .ai{color:var(--c-fail)}
.at{font-family:monospace;font-size:.88em;font-weight:600;white-space:nowrap}
.ar{color:#475569;font-size:.86em}

/* Metrics */
.mrow{display:flex;align-items:center;flex-wrap:wrap;gap:8px;
  padding:5px 0;border-bottom:1px solid #f8fafc}
.mn{font-family:monospace;font-weight:600;min-width:155px;font-size:.88em}
.mrow-warn .mn{color:var(--c-warn)}
.mrow-ok   .mn{color:var(--c-info)}
.mr{color:#475569;font-size:.86em;flex:1}
.mdetail{width:100%;margin-top:3px}
.mdetail summary{font-size:.76em;color:#94a3b8;cursor:pointer}

/* Judge */
.jrow{display:flex;align-items:center;gap:12px;margin-bottom:8px}
.jscore{font-size:2.2em;font-weight:800}
.jscore-pass{color:var(--c-pass)}
.jscore-fail{color:var(--c-fail)}
.jrationale{color:#374151;line-height:1.7;white-space:pre-wrap;font-size:.92em}
.jflags{color:var(--c-fail);font-size:.86em;margin-top:6px}
.jerr{color:var(--c-warn);font-size:.86em;margin-top:6px;font-family:monospace}

/* Timeline messages */
.msg{margin-bottom:5px;border-radius:6px;overflow:hidden;border:1px solid #e2e8f0}
.ms{padding:6px 12px;display:flex;align-items:center;gap:8px;
  font-size:.82em;cursor:pointer;list-style:none;user-select:none}
.ms::-webkit-details-marker{display:none}
.ms-system  {background:#f8fafc;color:#64748b}
.ms-user    {background:#eff6ff;color:#1e40af}
.ms-assistant{background:#f0f9ff;color:#075985}
.ms-tool    {background:#faf5ff;color:#5b21b6}
.mb{padding:10px 14px;background:#fff}
.sn{font-weight:800;color:#94a3b8;min-width:1.8em}
.tn{font-family:monospace}
.lat{margin-left:auto;color:#94a3b8;font-family:monospace;font-size:.8em}
.dim{color:#94a3b8;font-size:.88em}
.tc{white-space:pre-wrap;word-break:break-word;color:#374151;font-size:.91em}

/* Tool calls inside assistant messages */
.tcalls{margin-top:5px}
.tcall{margin-bottom:4px;border:1px solid #e2e8f0;border-radius:4px;overflow:hidden}
.tcall summary{padding:5px 10px;background:#f8fafc;font-family:monospace;
  font-size:.81em;cursor:pointer;list-style:none;user-select:none;display:flex;gap:6px;align-items:center}
.tcall summary::-webkit-details-marker{display:none}

/* JSON pre blocks */
pre.json{background:var(--c-code-bg);color:var(--c-code-fg);
  padding:10px 14px;border-radius:4px;overflow-x:auto;
  font-size:.75em;white-space:pre-wrap;word-break:break-word;
  margin:4px 0 0;max-height:280px;overflow-y:auto}

/* expand/collapse arrows via CSS */
details>.card-hd::before{content:"▶ ";font-size:.7em;opacity:.45}
details[open]>.card-hd::before{content:"▼ "}
details>.ms::before{content:"▶ ";font-size:.7em;opacity:.45}
details[open]>.ms::before{content:"▼ "}
details>.tcall>summary::before{content:"▶ ";font-size:.7em;opacity:.45}
details[open]>.tcall>summary::before{content:"▼ "}
"""


# ---------------------------------------------------------------------------
# Top-level assembly
# ---------------------------------------------------------------------------

def generate_html(trace: EvalTrace) -> str:
    ok = trace.case_passed
    verdict_text = "PASS" if ok else "FAIL"

    header = _render_header(trace)
    assertions = _render_assertions(trace)
    metrics = _render_metrics(trace)
    judge = _render_judge(trace)
    timeline = _fixed_render_timeline(trace)

    scored_note = f"scored {trace.scored_at[:19].replace('T', ' ')} UTC" if trace.scored_at else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>[{_esc(verdict_text)}] {_esc(trace.case_id)}</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="container">
{header}
{assertions}
{metrics}
{judge}
{timeline}
<p style="color:#94a3b8;font-size:.75em;text-align:right;margin-top:4px">
  Deep Research Lite eval viewer &nbsp;|&nbsp; {_esc(trace.run_id)} &nbsp;|&nbsp; {_esc(scored_note)}
</p>
</div>
</body>
</html>"""


def save_viewer(trace: EvalTrace, output_path: Path | None = None) -> Path:
    """Generate the viewer and write it to disk. Returns the file path."""
    if output_path is None:
        output_path = REPORTS_DIR / trace.case_id / f"{trace.run_id}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generate_html(trace), encoding="utf-8")
    return output_path
