"""Build sanitized research artifacts from the canonical frozen study.

Private input paths are command arguments and are never exported to Git.
The same manuscript blocks generate Markdown, standalone LaTeX, and ReportLab PDF.
"""

import argparse
import csv
import hashlib
import html
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    Preformatted,
)
from reportlab import rl_config

rl_config.invariant = 1


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def date_local(stamp):
    return (
        datetime.fromtimestamp(stamp, timezone.utc)
        .astimezone(ZoneInfo("America/New_York"))
        .isoformat()
    )


def aggregate(data, repo):
    frozen = read(data / "frozen.json")
    private = read(data / "analysis-private.json")
    registry = read(data / "study.json")
    expected = "635df7c0fa29da65b4b1b066f14852639a09db1e9f990561f0f4f29eecca747c"
    assert frozen["candidate_sha256"] == expected == sha(registry["candidate"])
    assert frozen["status"] in ("stopped", "expired")
    groups = {}
    for arm in ("baseline", "assisted"):
        rows = [s for s in frozen["sessions"] if s["arm"] == arm]
        turns = [t for s in rows for t in s["turns"]]
        valid = [t for t in turns if t["usage_valid"] and t["usage"] is not None]
        g = dict(private["groups"][arm])
        assert len(valid) == g["measured_turns"]
        g.update(
            input_tokens=sum(t["usage"]["input_tokens"] for t in valid)
            if valid
            else None,
            output_tokens=sum(t["usage"]["output_tokens"] for t in valid)
            if valid
            else None,
            total_tokens=g["total_tokens"] if valid else None,
            cached_input_tokens=g["cached_input_tokens"] if valid else None,
            completed_turns=sum(t["status"] == "completed" for t in turns),
            open_turns=sum(t["status"] == "open" for t in turns),
            tool_calls=sum(t["tool_calls"] for t in turns),
            test_runs=sum(t["test_runs"] for t in turns),
            test_pass_flags=sum(t["test_passes"] for t in turns),
            test_failure_flags=sum(t["test_failures"] for t in turns),
            subagents=sum(s["subagent_count"] for s in rows),
            missing_subagent_usage=sum(t["missing_subagent_usage"] for t in turns),
            retained_counter_turns=sum(t["usage"] is not None for t in turns),
            projects=len({s["project_hash"] for s in rows}),
            models=sorted(
                {m for t in turns for m in t.get("models_seen", [t["model"]])}
            ),
            client_versions=sorted({s["client_version"] for s in rows}),
            warning_counts={
                w: sum(w in s["usage_warnings"] for s in rows)
                for w in sorted({w for s in rows for w in s["usage_warnings"]})
            },
        )
        groups[arm] = g
    sessions = [read(data / "sessions" / (i + ".json")) for i in registry["sessions"]]
    probes = [s for s in sessions if s["arm"] == "probe"]
    result = {
        "schema_version": 1,
        "author": "sgohils",
        "protocol": frozen["protocol"],
        "seed": frozen["seed"],
        "study_start_utc": datetime.fromtimestamp(
            frozen["started_at"], timezone.utc
        ).isoformat(),
        "study_deadline_utc": datetime.fromtimestamp(
            frozen["deadline"], timezone.utc
        ).isoformat(),
        "study_start_eastern": date_local(frozen["started_at"]),
        "study_deadline_eastern": date_local(frozen["deadline"]),
        "frozen_at_eastern": date_local(registry["closed_at"]),
        "frozen_sha256": sha(data / "frozen.json"),
        "candidate_sha256": expected,
        "implementation_commit": "a777f2329bcf56769c208fd2a75d84de69c12755",
        "groups": groups,
        "token_reduction_fraction": private["token_reduction_fraction"],
        "chat_bootstrap_95_interval": private["chat_bootstrap_95_interval"],
        "interpretation": "inconclusive; one baseline chat, no assisted chats, no final usable token turns",
        "target_reduction_fraction": 0.1,
        "target_achieved": None,
        "ordinary_activation_changed": False,
        "background_retraining_seconds": 0.0,
        "excluded_probes": {
            "chats": len(probes),
            "turns": sum(len(s["turns"]) for s in probes),
            "decisions": sum(
                t.get("model_decisions", 0) for s in probes for t in s["turns"].values()
            ),
            "recommendations": sum(
                t["advice_count"] for s in probes for t in s["turns"].values()
            ),
            "truncated_streams": sum(
                "transcript_truncated" in st["warnings"]
                for s in probes
                for st in s["streams"].values()
            ),
        },
        "protocol_amendments": frozen["protocol_amendments"],
        "instrumentation": {
            "oversized_content_repair_installed": True,
            "repair_ci_tests": 42,
            "windows_ci": "passed",
            "linux_ci": "passed",
            "scheduled_refresh_unavailable_during_window": True,
            "final_reconciliation_detected_truncation": True,
            "hooks_retained_as_noops": True,
            "monitor_token_cost": None,
            "monitor_cpu_seconds": None,
            "complete_helper_process_overhead": None,
        },
        "token_metric": "input + output; cached input is a subset of input; unknown is null, never zero",
    }
    for s in sessions:
        assert s["session_hash"] not in json.dumps(result)
        assert s["project_hash"] not in json.dumps(result)
    assert groups["baseline"]["chats"] == 1 and groups["assisted"]["chats"] == 0
    assert groups["baseline"]["missing_usage_turns"] == 5
    return result


BLUE = colors.HexColor("#245C82")
ORANGE = colors.HexColor("#BD6B36")
GRAY = colors.HexColor("#5A6873")
PALE = colors.HexColor("#EAF1F5")
FONT = "PaperSerif"


def fonts():
    paths = [
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/timesbd.ttf"),
        Path("C:/Windows/Fonts/timesi.ttf"),
    ]
    if all(p.exists() for p in paths):
        for name, path in zip((FONT, FONT + "Bold", FONT + "Italic"), paths):
            pdfmetrics.registerFont(TTFont(name, str(path)))
    else:
        import reportlab

        root = Path(reportlab.__file__).parent / "fonts"
        for name, fn in (
            (FONT, "Vera.ttf"),
            (FONT + "Bold", "VeraBd.ttf"),
            (FONT + "Italic", "VeraIt.ttf"),
        ):
            pdfmetrics.registerFont(TTFont(name, str(root / fn)))
    pdfmetrics.registerFontFamily(
        FONT,
        normal=FONT,
        bold=FONT + "Bold",
        italic=FONT + "Italic",
        boldItalic=FONT + "Bold",
    )


def label(d, x, y, text, size=9, color=GRAY, anchor="start"):
    d.add(
        String(
            x,
            y,
            str(text),
            fontName=FONT,
            fontSize=size,
            fillColor=color,
            textAnchor=anchor,
        )
    )


def box(d, x, y, w, h, text):
    d.add(
        Rect(x, y, w, h, fillColor=PALE, strokeColor=BLUE, strokeWidth=0.8, rx=4, ry=4)
    )
    lines = simpleSplit(text, FONT, 9, w - 12)
    for i, line in enumerate(lines):
        label(
            d,
            x + w / 2,
            y + h / 2 + (len(lines) - 1) * 5 - i * 10,
            line,
            9,
            BLUE,
            "middle",
        )


def arrow(d, x1, y1, x2, y2):
    d.add(Line(x1, y1, x2, y2, strokeColor=GRAY, strokeWidth=0.8))
    if x2 > x1:
        d.add(Line(x2, y2, x2 - 4, y2 + 3, strokeColor=GRAY))
        d.add(Line(x2, y2, x2 - 4, y2 - 3, strokeColor=GRAY))
    else:
        d.add(Line(x2, y2, x2 - 3, y2 + 4, strokeColor=GRAY))
        d.add(Line(x2, y2, x2 + 3, y2 + 4, strokeColor=GRAY))


def figures(synthetic, curves):
    items = {}
    d = Drawing(475, 175)
    for x, txt in (
        (0, "New eligible coding chat"),
        (167, "Persistent 50/50 assignment"),
        (334, "Actual Codex work"),
    ):
        box(d, x, 115, 141, 42, txt)
    arrow(d, 142, 136, 166, 136)
    arrow(d, 309, 136, 333, 136)
    box(
        d, 167, 40, 141, 48, "Baseline: record only / Assisted: frozen advisor + record"
    )
    arrow(d, 237, 115, 237, 89)
    box(d, 334, 40, 141, 48, "Read-only usage extraction; hashed local records")
    arrow(d, 308, 64, 333, 64)
    label(
        d,
        0,
        14,
        "Deadline guard -> frozen snapshot -> coverage checks -> descriptive analysis",
        9,
    )
    items["system-flow"] = d
    d = Drawing(475, 190)
    names = ["linear", "neural", "rules", "random"]
    labels = ["Linear", "Neural", "Rules", "Random"]
    for panel, (metric, title, maximum) in enumerate(
        [
            ("successes", "Synthetic completions / 24", 24),
            ("mean_proxy_tokens", "Mean simulated context / episode", 1200),
        ]
    ):
        x0 = 35 + panel * 245
        y0 = 38
        w = 185
        h = 116
        label(d, x0, 175, title, 10, BLUE)
        d.add(Line(x0, y0, x0 + w, y0, strokeColor=GRAY))
        d.add(Line(x0, y0, x0, y0 + h, strokeColor=GRAY))
        for k in range(5):
            value = maximum * k / 4
            y = y0 + h * k / 4
            d.add(
                Line(
                    x0,
                    y,
                    x0 + w,
                    y,
                    strokeColor=colors.HexColor("#D9E1E6"),
                    strokeWidth=0.4,
                )
            )
            label(d, x0 - 5, y - 3, f"{value:g}", 8, anchor="end")
        for i, (name, titleLabel) in enumerate(zip(names, labels)):
            value = synthetic["metrics"][name][metric]
            x = x0 + 9 + i * 45
            d.add(
                Rect(
                    x,
                    y0,
                    27,
                    h * value / maximum,
                    fillColor=BLUE
                    if name == "linear"
                    else ORANGE
                    if name == "neural"
                    else GRAY,
                    strokeWidth=0,
                )
            )
            label(
                d,
                x + 13.5,
                y0 + h * value / maximum + 5,
                f"{value:.1f}" if metric != "successes" else value,
                8,
                anchor="middle",
            )
            label(d, x + 13.5, 24, titleLabel, 8, anchor="middle")
    items["synthetic-comparison"] = d
    d = Drawing(475, 180)
    for panel, (metric, title, lo, hi) in enumerate(
        [
            ("balanced_training_loss", "Generation leader: training loss", 0, 1.85),
            (
                "validation_successes",
                "Generation leader: validation completions",
                0,
                24,
            ),
        ]
    ):
        x0 = 36 + panel * 245
        y0 = 35
        w = 185
        h = 115
        label(d, x0, 167, title, 9, BLUE)
        for k in range(5):
            y = y0 + h * k / 4
            value = lo + (hi - lo) * k / 4
            d.add(
                Line(
                    x0,
                    y,
                    x0 + w,
                    y,
                    strokeColor=colors.HexColor("#D9E1E6"),
                    strokeWidth=0.4,
                )
            )
            label(
                d,
                x0 - 4,
                y - 3,
                f"{value:.2f}" if panel == 0 else f"{value:g}",
                8,
                anchor="end",
            )
        for kind, color in [("linear", BLUE), ("neural", ORANGE)]:
            points = [
                (
                    x0 + (float(row["generation"]) - 1) * w / 49,
                    y0 + (float(row[metric]) - lo) * h / (hi - lo),
                )
                for row in curves
                if row["model"] == kind
            ]
            for a, b in zip(points, points[1:]):
                d.add(Line(*a, *b, strokeColor=color, strokeWidth=1.1))
        for generation in [1, 10, 20, 30, 40, 50]:
            label(d, x0 + (generation - 1) * w / 49, 20, generation, 8, anchor="middle")
    label(
        d,
        35,
        4,
        "Blue: linear     Orange: neural. Per-generation leaders, not a monotone champion curve.",
        8,
    )
    items["evolution-curves"] = d
    d = Drawing(475, 155)
    label(
        d,
        0,
        140,
        "Final field coverage: enrolled turns and usable token measurements",
        10,
        BLUE,
    )
    for i, (name, total, usable) in enumerate([("Baseline", 5, 0), ("Assisted", 0, 0)]):
        y = 82 - i * 48
        label(d, 0, y + 12, name, 10)
        if total:
            d.add(Rect(80, y, 300 * total / 5, 25, fillColor=ORANGE, strokeWidth=0))
        label(d, 390, y + 9, f"{total} enrolled; {usable} usable", 9)
    label(
        d,
        80,
        7,
        "Orange = usage unavailable. No token-distribution or effect plot is estimable.",
        9,
    )
    items["field-coverage"] = d
    return items


def inline_html(text):
    text = html.escape(text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', text)
    return text


def escape_tex(text):
    mapping = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(mapping.get(c, c) for c in str(text))


def inline_tex(text):
    result = escape_tex(text)
    result = re.sub(r"\*\*(.*?)\*\*", r"\\textbf{\1}", result)
    result = re.sub(r"`([^`]+)`", r"\\texttt{\1}", result)
    return re.sub(r"\[(\d+)\]", r"\\cite{r\1}", result)


def write_documents(blocks, figs, latex_figs, refs, out, title, publication_date):
    styles = getSampleStyleSheet()
    for name in ("BodyText", "Heading1", "Heading2", "Title"):
        styles[name].fontName = FONT
    styles["BodyText"].fontSize = 10.5
    styles["BodyText"].leading = 14
    styles["BodyText"].spaceAfter = 7
    styles["Heading1"].fontName = FONT + "Bold"
    styles["Heading1"].fontSize = 13
    styles["Heading1"].leading = 16
    styles["Heading1"].spaceBefore = 12
    styles["Heading2"].fontName = FONT + "Bold"
    styles["Heading2"].fontSize = 11
    styles["Heading2"].leading = 14
    styles["Heading1"].keepWithNext = True
    styles["Heading2"].keepWithNext = True
    styles["Title"].fontName = FONT + "Bold"
    styles["Title"].fontSize = 21
    styles["Title"].leading = 25
    caption = ParagraphStyle(
        "Caption", parent=styles["BodyText"], fontSize=9, leading=12, textColor=GRAY
    )
    tiny = ParagraphStyle(
        "Cell", parent=styles["BodyText"], fontSize=9, leading=11, spaceAfter=0
    )
    equation_style = ParagraphStyle(
        "Equation",
        parent=styles["BodyText"],
        fontSize=10,
        leading=15,
        leftIndent=12,
        spaceBefore=5,
        spaceAfter=10,
    )
    story = [
        Paragraph(inline_html(title), styles["Title"]),
        Paragraph(
            "sgohils",
            ParagraphStyle("Author", parent=styles["BodyText"], alignment=TA_CENTER),
        ),
        Paragraph(
            "Technical preprint | " + publication_date,
            ParagraphStyle("Date", parent=caption, alignment=TA_CENTER),
        ),
        Spacer(1, 14),
    ]
    md = [
        "# " + title,
        "",
        "**Author:** sgohils  ",
        "**Technical preprint:** " + publication_date,
        "",
    ]
    tex = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.85in]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{amsmath,amssymb,booktabs,tabularx,hyperref,tikz,pgfplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\hypersetup{colorlinks=true,urlcolor=blue,citecolor=blue}",
        r"\setlength{\parskip}{0.45em}",
        r"\setlength{\parindent}{0pt}",
        r"\emergencystretch=2em",
        r"\title{" + escape_tex(title) + "}",
        r"\author{sgohils}",
        r"\date{" + escape_tex(publication_date) + "}",
        r"\begin{document}",
        r"\maketitle",
    ]
    for block in blocks:
        kind = block["kind"]
        text = block.get("text", "")
        if kind in ("h1", "h2"):
            story.append(
                Paragraph(
                    inline_html(text),
                    styles["Heading1" if kind == "h1" else "Heading2"],
                )
            )
            md.extend([("## " if kind == "h1" else "### ") + text, ""])
            tex.extend(
                [
                    (r"\section*{" if kind == "h1" else r"\subsection*{")
                    + escape_tex(text)
                    + "}"
                ]
            )
            if text.startswith("Appendix A."):
                tex.extend(
                    [
                        r"\setcounter{table}{0}",
                        r"\renewcommand{\thetable}{A\arabic{table}}",
                    ]
                )
        elif kind == "p":
            story.append(Paragraph(inline_html(text), styles["BodyText"]))
            md.extend([text, ""])
            tex.extend([inline_tex(text), ""])
        elif kind == "equation":
            story.append(Paragraph(inline_html(block["plain"]), equation_style))
            md.extend(["$$", block["latex"], "$$", ""])
            tex.extend([r"\[", block["latex"], r"\]"])
        elif kind == "code":
            story.append(
                Preformatted(
                    text,
                    ParagraphStyle(
                        "Code",
                        fontName="Courier",
                        fontSize=8,
                        leading=10,
                        spaceAfter=10,
                    ),
                    maxLineLength=85,
                )
            )
            md.extend(["```text", text, "```", ""])
            tex.extend([r"\begin{verbatim}", text, r"\end{verbatim}"])
        elif kind == "table":
            rows = block["rows"]
            header = block["header"]
            data = [
                [Paragraph(inline_html(str(cell)), tiny) for cell in row]
                for row in [header, *rows]
            ]
            n = len(header)
            widths = (
                [475 * 0.42] + [475 * 0.58 / (n - 1)] * (n - 1)
                if n > 2
                else [475 * 0.6, 475 * 0.4]
            )
            table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), PALE),
                        ("LINEABOVE", (0, 0), (-1, 0), 0.7, BLUE),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.5, BLUE),
                        ("LINEBELOW", (0, -1), (-1, -1), 0.5, GRAY),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.append(
                KeepTogether(
                    [
                        Paragraph(inline_html(block["caption"]), caption),
                        table,
                        Spacer(1, 10),
                    ]
                )
            )
            md.extend(
                [
                    block["caption"],
                    "",
                    "| " + " | ".join(header) + " |",
                    "| " + " | ".join(["---"] * n) + " |",
                    *["| " + " | ".join(map(str, row)) + " |" for row in rows],
                    "",
                ]
            )
            tex_caption = re.sub(r"^Table [A-Z]?\d+\.\s*", "", block["caption"])
            tex.extend(
                [
                    r"\begin{table}[htbp]",
                    r"\centering",
                    r"\small",
                    r"\caption{" + escape_tex(tex_caption) + "}",
                    r"\begin{tabularx}{\linewidth}{@{}" + ("X" * n) + r"@{}}",
                    r"\toprule",
                    " & ".join(map(escape_tex, header)) + r"\\",
                    r"\midrule",
                    *[" & ".join(map(escape_tex, row)) + r"\\" for row in rows],
                    r"\bottomrule",
                    r"\end{tabularx}",
                    r"\end{table}",
                ]
            )
        elif kind == "figure":
            drawing = figs[block["id"]]
            story.append(
                KeepTogether(
                    [
                        drawing,
                        Paragraph(inline_html(block["caption"]), caption),
                        Spacer(1, 10),
                    ]
                )
            )
            md.extend(
                ["![" + block["caption"] + "](figures/" + block["id"] + ".png)", ""]
            )
            tex_caption = re.sub(r"^Figure \d+\.\s*", "", block["caption"])
            tex.extend(
                [
                    r"\begin{figure}[htbp]",
                    r"\centering",
                    latex_figs[block["id"]],
                    r"\caption{" + escape_tex(tex_caption) + "}",
                    r"\end{figure}",
                ]
            )
    story.append(Paragraph("References", styles["Heading1"]))
    md.extend(["## References", ""])
    tex.extend([r"\begin{thebibliography}{99}"])
    for index, ref in enumerate(refs, 1):
        entry = f"[{index}] {ref['authors']}. {ref['title']}. {ref['year']}. "
        story.append(
            Paragraph(
                inline_html(entry)
                + f'<link href="{html.escape(ref["url"])}" color="#245C82">{html.escape(ref["url"])}</link>',
                caption,
            )
        )
        md.extend([entry + "[" + ref["title"] + "](" + ref["url"] + ").", ""])
        tex.extend(
            [
                r"\bibitem{r"
                + str(index)
                + "}"
                + escape_tex(entry.split("] ", 1)[1])
                + r"\url{"
                + ref["url"]
                + "}."
            ]
        )
    tex.extend([r"\end{thebibliography}", r"\end{document}"])

    def page(canvas, doc):
        canvas.setTitle(title)
        canvas.setAuthor("sgohils")
        canvas.setFont(FONT, 8)
        canvas.setFillColor(GRAY)
        canvas.drawString(54, 28, "codex-local-decider | Technical preprint")
        canvas.drawRightString(558, 28, str(doc.page))

    doc = SimpleDocTemplate(
        str(out / "codex-local-decider-preprint.pdf"),
        pagesize=(612, 792),
        rightMargin=54,
        leftMargin=54,
        topMargin=46,
        bottomMargin=49,
        title=title,
        author="sgohils",
    )
    doc.build(story, onFirstPage=page, onLaterPages=page)
    (out / "codex-local-decider-preprint.md").write_text(
        "\n".join(md), encoding="utf-8"
    )
    (out / "codex-local-decider-preprint.tex").write_text(
        "\n".join(tex), encoding="utf-8"
    )


def latex_figures(synthetic, curves):
    flow = r"""\begin{tikzpicture}[every node/.style={draw,rounded corners,align=center,text width=2.7cm,minimum height=1cm,font=\small}]
\node (chat) at (0,2) {New eligible coding chat};
\node (assign) at (4,2) {Persistent 50/50 assignment};
\node (work) at (8,2) {Actual Codex work};
\node (arms) at (4,0) {Baseline: record only\\Assisted: frozen advisor + record};
\node (usage) at (8,0) {Read-only usage extraction\\Hashed local records};
\draw[->] (chat)--(assign);\draw[->] (assign)--(work);
\draw[->] (assign)--(arms);\draw[->] (arms)--(usage);
\end{tikzpicture}"""

    def plot(title, metric, options, ymaximum=None):
        series = []
        for model, color in [("linear", "blue"), ("neural", "orange")]:
            coordinates = " ".join(
                f"({r['generation']},{float(r[metric]):.9g})"
                for r in curves
                if r["model"] == model
            )
            series.append(
                r"\addplot[no marks,"
                + color
                + "] coordinates {"
                + coordinates
                + "};"
                + r"\addlegendentry{"
                + model.title()
                + "}"
            )
        return (
            r"\begin{minipage}{0.48\linewidth}\centering\begin{tikzpicture}\begin{axis}[width=\linewidth,height=4.4cm,title={"
            + title
            + r"},xlabel={Generation},xmin=1,xmax=50,legend style={font=\scriptsize},legend pos=south west,"
            + options
            + "]"
            + "".join(series)
            + r"\end{axis}\end{tikzpicture}\end{minipage}"
        )

    learning = (
        plot("Training loss", "balanced_training_loss", "ymin=0,ymax=1.85")
        + r"\hfill"
        + plot(
            "Validation completions",
            "validation_successes",
            "ymin=0,ymax=24,ytick={0,6,12,18,24}",
        )
    )
    synthetic_parts = []
    for metric, title, maximum in [
        ("successes", "Completions / 24", 24),
        ("mean_proxy_tokens", "Mean simulated context", 1200),
    ]:
        coords = " ".join(
            f"({name.title()},{synthetic['metrics'][name][metric]:.9g})"
            for name in ["linear", "neural", "rules", "random"]
        )
        synthetic_parts.append(
            r"\begin{minipage}{0.48\linewidth}\centering\begin{tikzpicture}\begin{axis}[width=\linewidth,height=4.4cm,ybar,title={"
            + title
            + r"},symbolic x coords={Linear,Neural,Rules,Random},xtick=data,x tick label style={font=\scriptsize},ymin=0,ymax="
            + str(maximum)
            + r",bar width=11pt,nodes near coords,nodes near coords style={font=\tiny}]\addplot[fill=blue!50] coordinates {"
            + coords
            + r"};\end{axis}\end{tikzpicture}\end{minipage}"
        )
    coverage = r"\begin{tikzpicture}\begin{axis}[width=0.85\linewidth,height=4cm,xbar,symbolic y coords={Assisted,Baseline},ytick=data,xmin=0,xmax=5.7,xtick={0,1,2,3,4,5},xlabel={Turns with unavailable final usage},bar width=12pt,nodes near coords]\addplot[fill=orange!65] coordinates {(0,Assisted)(5,Baseline)};\end{axis}\end{tikzpicture}"
    return {
        "system-flow": flow,
        "evolution-curves": learning,
        "synthetic-comparison": r"\hfill".join(synthetic_parts),
        "field-coverage": coverage,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    A = aggregate(args.data_dir, args.repo_dir)
    S = read(args.repo_dir / "examples/results/local-comparison.json")
    manifest = read(args.repo_dir / "examples/results/run-manifest.json")
    assert manifest["models"]["linear"]["sha256"] == A["candidate_sha256"]
    assert (
        S["metrics"]["linear"]["successes"] == 24
        and S["metrics"]["neural"]["successes"] == 12
    )
    assert S["metrics"]["rules"]["mean_proxy_tokens"] == 597.5
    with (args.repo_dir / "examples/results/learning-curves.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        curves = list(csv.DictReader(file))
    assert len(curves) == 100
    fonts()
    drawings = figures(S, curves)
    for identity, drawing in drawings.items():
        renderPDF.drawToFile(drawing, str(out / "figures" / f"{identity}.pdf"))
        renderSVG.drawToFile(drawing, str(out / "figures" / f"{identity}.svg"))
    from manuscript import manuscript, REFERENCES

    blocks = manuscript(A, S)
    title = "Evolving Local Decision Controllers for Coding Assistants: A Five-Day Randomized Pilot"
    publication_date = datetime.fromisoformat(A["frozen_at_eastern"]).strftime(
        "%B %d, %Y"
    )
    write_documents(
        blocks,
        drawings,
        latex_figures(S, curves),
        REFERENCES,
        out,
        title,
        publication_date,
    )
    (out / "aggregate-results.json").write_text(
        json.dumps(A, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(
        args.repo_dir / "examples/results/local-comparison.json",
        out / "synthetic-results.json",
    )
    shutil.copy2(
        args.repo_dir / "examples/results/run-manifest.json",
        out / "synthetic-run-manifest.json",
    )
    shutil.copy2(
        args.repo_dir / "examples/results/learning-curves.csv",
        out / "learning-curves.csv",
    )
    bib = []
    for index, ref in enumerate(REFERENCES, 1):
        bib.append(
            "@misc{r"
            + str(index)
            + ",\n  author={"
            + ref["authors"]
            + "},\n  title={"
            + ref["title"]
            + "},\n  howpublished={\\url{"
            + ref["url"]
            + "}},\n  note={"
            + ref["year"]
            + "}\n}"
        )
    (out / "references.bib").write_text("\n\n".join(bib) + "\n", encoding="utf-8")
    (out / "manuscript-blocks.json").write_text(
        json.dumps(blocks, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(Path(__file__), out / "build_paper.py")
    shutil.copy2(Path(__file__).with_name("manuscript.py"), out / "manuscript.py")
    print(
        json.dumps(
            {
                "output_dir": str(out),
                "blocks": len(blocks),
                "manuscript_words": sum(len(b.get("text", "").split()) for b in blocks),
                "field_chats": {a: g["chats"] for a, g in A["groups"].items()},
                "usable_token_turns": {
                    a: g["measured_turns"] for a, g in A["groups"].items()
                },
                "freeze_sha256": A["frozen_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
