import re
import argparse
from pathlib import Path
from html import escape

def parse_sections(md: str):
    sections = []
    sec_iter = list(re.finditer(r"^###\s+1\.(\d+)\s+(.+)$", md, re.M))
    for i, m in enumerate(sec_iter):
        start = m.end()
        end = sec_iter[i + 1].start() if i + 1 < len(sec_iter) else len(md)
        title = m.group(2).strip()
        block = md[start:end]
        tasks = re.findall(r"^\s*-\s*\[(?:x|X| )\].*$", block, re.M)
        done = re.findall(r"^\s*-\s*\[(?:x|X)\].*$", block, re.M)
        total = len(tasks)
        completed = len(done)
        sections.append({
            "key": f"1.{m.group(1)}",
            "title": title,
            "completed": completed,
            "total": total,
        })
    return sections


def compute_overall(sections):
    total = sum(s["total"] for s in sections)
    completed = sum(s["completed"] for s in sections)
    pct = 0.0 if total == 0 else (completed / total) * 100.0
    return completed, total, pct


def fmt_pct(completed, total):
    if total == 0:
        return 0
    return int(round((completed / total) * 100))


def generate_svg(sections, out_path: Path):
    # Layout tuning - improved spacing and sizing
    padding = 24
    row_h = 36
    header_h = 92
    gap = 12

    # Estimate label width and provide ellipsis
    def estimate_text_width_px(text: str, font_px: int = 14) -> int:
        # Empirical average char width for 14px semi-bold UI font ~8.2px
        avg = 8.2 * (font_px / 14.0)
        return int(len(text) * avg)

    labels = [f"{s['key']} {s['title']}".strip() for s in sections]
    max_label_px = max([estimate_text_width_px(t) for t in labels] + [380])
    label_w = min(580, max(380, max_label_px + 32))
    bar_w = 540

    rows = len(sections) + 1
    height = header_h + rows * (row_h + gap) + padding
    width = padding * 2 + label_w + bar_w

    def row_y(i):
        return header_h + i * (row_h + gap)

    svg = []
    svg.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>")
    svg.append("<defs>")
    # Gradients for modern look
    svg.append("<linearGradient id='grad-low' x1='0%' y1='0%' x2='0%' y2='100%'>")
    svg.append("<stop offset='0%' style='stop-color:#f87171;stop-opacity:1' />")
    svg.append("<stop offset='100%' style='stop-color:#dc2626;stop-opacity:1' />")
    svg.append("</linearGradient>")
    svg.append("<linearGradient id='grad-mid' x1='0%' y1='0%' x2='0%' y2='100%'>")
    svg.append("<stop offset='0%' style='stop-color:#fbbf24;stop-opacity:1' />")
    svg.append("<stop offset='100%' style='stop-color:#f59e0b;stop-opacity:1' />")
    svg.append("</linearGradient>")
    svg.append("<linearGradient id='grad-high' x1='0%' y1='0%' x2='0%' y2='100%'>")
    svg.append("<stop offset='0%' style='stop-color:#34d399;stop-opacity:1' />")
    svg.append("<stop offset='100%' style='stop-color:#10b981;stop-opacity:1' />")
    svg.append("</linearGradient>")
    # Dark mode gradients
    svg.append("<linearGradient id='grad-low-dark' x1='0%' y1='0%' x2='0%' y2='100%'>")
    svg.append("<stop offset='0%' style='stop-color:#f87171;stop-opacity:1' />")
    svg.append("<stop offset='100%' style='stop-color:#ef4444;stop-opacity:1' />")
    svg.append("</linearGradient>")
    svg.append("<linearGradient id='grad-mid-dark' x1='0%' y1='0%' x2='0%' y2='100%'>")
    svg.append("<stop offset='0%' style='stop-color:#fbbf24;stop-opacity:1' />")
    svg.append("<stop offset='100%' style='stop-color:#d97706;stop-opacity:1' />")
    svg.append("</linearGradient>")
    svg.append("<linearGradient id='grad-high-dark' x1='0%' y1='0%' x2='0%' y2='100%'>")
    svg.append("<stop offset='0%' style='stop-color:#34d399;stop-opacity:1' />")
    svg.append("<stop offset='100%' style='stop-color:#059669;stop-opacity:1' />")
    svg.append("</linearGradient>")
    svg.append("<filter id='shadow'>")
    svg.append("<feDropShadow dx='0' dy='1' stdDeviation='2' flood-opacity='0.1'/>")
    svg.append("</filter>")
    svg.append("<style><![CDATA[\n  :root{\n    --card-bg:#ffffff;\n    --card-stroke:#e5e7eb;\n    --text-primary:#111111;\n    --text-secondary:#222222;\n    --text-meta:#6b7280;\n    --bar-bg:#f1f5f9;\n    --bar-border:#e2e8f0;\n    --pct-stroke:#ffffff;\n  }\n  @media (prefers-color-scheme: dark){\n    :root{\n      --card-bg:#0d1117;\n      --card-stroke:#30363d;\n      --text-primary:#e6edf3;\n      --text-secondary:#e6edf3;\n      --text-meta:#8b949e;\n      --bar-bg:#1c2128;\n      --bar-border:#30363d;\n      --pct-stroke:#0d1117;\n    }\n  }\n  .card{ fill: var(--card-bg); stroke: var(--card-stroke); stroke-width: 1; }\n  .title{font: 700 22px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; fill: var(--text-primary);}\n  .label{font: 600 14px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; fill: var(--text-primary);}\n  .sub{font: 700 15px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; fill: var(--text-primary);}\n  .pct{font: 600 13px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; fill: var(--text-primary); paint-order: stroke; stroke: var(--pct-stroke); stroke-width: 2px;}\n  .pct.on-fill{ fill:#ffffff; }\n  .meta{font: 400 12px system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; fill: var(--text-meta);}\n  .bar-bg{ fill: var(--bar-bg); stroke: var(--bar-border); stroke-width: 1; }\n  @media (prefers-color-scheme: light){\n    .bar-fill-low{ fill: url(#grad-low); }\n    .bar-fill-mid{ fill: url(#grad-mid); }\n    .bar-fill-high{ fill: url(#grad-high); }\n  }\n  @media (prefers-color-scheme: dark){\n    .bar-fill-low{ fill: url(#grad-low-dark); }\n    .bar-fill-mid{ fill: url(#grad-mid-dark); }\n    .bar-fill-high{ fill: url(#grad-high-dark); }\n  }\n]]></style>")
    svg.append("</defs>")

    svg.append(f"<rect class='card' x='0' y='0' width='{width}' height='{height}' rx='10' ry='10' />")
    svg.append(f"<text class='title' x='{padding}' y='{padding + 24}'>Array API Roadmap</text>")
    svg.append(f"<text class='meta' x='{padding}' y='{padding + 48}'>Source: docs/TODO.md · Updated by CI</text>")

    # rows
    for i, s in enumerate(sections, start=0):
        y = row_y(i)
        pct = fmt_pct(s["completed"], s["total"])
        raw_label = f"{s['key']} {s['title']}".strip()
        # Ellipsize label to fit label column
        max_label_px = label_w - 16
        def ellipsize(t: str) -> str:
            if estimate_text_width_px(t) <= max_label_px:
                return t
            # Reserve 1 char for '…'
            lo, hi = 0, len(t)
            while lo < hi:
                mid = (lo + hi) // 2
                cand = t[:mid] + '…'
                if estimate_text_width_px(cand) <= max_label_px:
                    lo = mid + 1
                else:
                    hi = mid
            trimmed = t[:max(lo - 1, 0)] + ('…' if lo > 1 else '')
            return trimmed
        label = escape(ellipsize(raw_label))

        # Draw label (vertically centered)
        svg.append(f"<text class='label' x='{padding}' y='{y + row_h/2}' dominant-baseline='middle'>{label}</text>")

        # Draw bar
        bx = padding + label_w
        by = y
        svg.append(f"<rect class='bar-bg' x='{bx}' y='{by}' width='{bar_w}' height='{row_h}' rx='10' ry='10' />")
        fill_w = int(bar_w * pct / 100)
        fill_class = 'bar-fill-low' if pct < 34 else ('bar-fill-mid' if pct < 67 else 'bar-fill-high')
        if fill_w > 0:
            svg.append(f"<rect class='{fill_class}' x='{bx}' y='{by}' width='{fill_w}' height='{row_h}' rx='10' ry='10' filter='url(#shadow)' />")
        # Percentage text right-aligned inside the bar area; toggle on-fill class when fill is long
        pct_text = f"{pct}% ({s['completed']}/{s['total']})"
        on_fill = fill_w > int(bar_w * 0.85)
        pct_class = 'pct on-fill' if on_fill else 'pct'
        svg.append(f"<text class='{pct_class}' x='{bx + bar_w - 8}' y='{by + row_h/2}' dominant-baseline='middle' text-anchor='end'>{escape(pct_text)}</text>")

    # overall
    ov_c, ov_t, ov_pct = compute_overall(sections)
    y = row_y(len(sections))
    svg.append(f"<text class='sub' x='{padding}' y='{y + row_h/2}' dominant-baseline='middle'>Overall</text>")
    bx = padding + label_w
    by = y
    svg.append(f"<rect class='bar-bg' x='{bx}' y='{by}' width='{bar_w}' height='{row_h}' rx='10' ry='10' />")
    fill_w = int(bar_w * ov_pct / 100)
    fill_class = 'bar-fill-low' if ov_pct < 34 else ('bar-fill-mid' if ov_pct < 67 else 'bar-fill-high')
    if fill_w > 0:
        svg.append(f"<rect class='{fill_class}' x='{bx}' y='{by}' width='{fill_w}' height='{row_h}' rx='10' ry='10' filter='url(#shadow)' />")
    ov_on_fill = fill_w > int(bar_w * 0.85)
    ov_pct_class = 'pct on-fill' if ov_on_fill else 'pct'
    svg.append(f"<text class='{ov_pct_class}' x='{bx + bar_w - 8}' y='{by + row_h/2}' dominant-baseline='middle' text-anchor='end'>{int(round(ov_pct))}% ({ov_c}/{ov_t})</text>")

    svg.append("</svg>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(svg), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="docs/TODO.md")
    p.add_argument("--output", default="docs/roadmap.svg")
    args = p.parse_args()
    md = Path(args.input).read_text(encoding="utf-8")
    # only parse section 1) Not-yet-implemented by namespace
    m = re.search(r"^##\s*1\)\s*Not-yet-implemented.*$", md, re.M)
    if m:
        start = m.start()
        md_slice = md[start:]
    else:
        md_slice = md
    sections = parse_sections(md_slice)
    if not sections:
        # Fallback: produce an empty SVG with message
        Path(args.output).write_text(
            "<svg xmlns='http://www.w3.org/2000/svg' width='600' height='80'><text x='10' y='40'>No sections found in docs/TODO.md</text></svg>",
            encoding="utf-8",
        )
        return
    generate_svg(sections, Path(args.output))


if __name__ == "__main__":
    main()
