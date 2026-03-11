"""Convert markdown papers to PDF using fpdf2 write_html."""

import re
import sys
import html as html_mod
from fpdf import FPDF


def md_to_html(md_text):
    """Convert markdown to simple HTML for fpdf2's write_html."""
    lines = md_text.split("\n")
    html_parts = []
    in_code = False
    in_table = False
    table_header_done = False

    for line in lines:
        stripped = line.strip()

        # Code blocks
        if stripped.startswith("```"):
            if in_code:
                html_parts.append("</pre>")
                in_code = False
            else:
                html_parts.append('<pre style="font-size:7">')
                in_code = True
            continue

        if in_code:
            html_parts.append(html_mod.escape(line))
            html_parts.append("<br>")
            continue

        # Table
        if "|" in stripped and stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            # Skip separator
            if all(re.match(r'^[-:]+$', c) for c in cells):
                table_header_done = True
                continue
            if not in_table:
                in_table = True
                table_header_done = False
                html_parts.append('<table border="1" width="100%">')
            tag = "th" if not table_header_done else "td"
            html_parts.append("<tr>")
            for cell in cells:
                text = html_mod.escape(cell)
                html_parts.append(f"<{tag}>{text}</{tag}>")
            html_parts.append("</tr>")
            continue
        elif in_table:
            html_parts.append("</table><br>")
            in_table = False
            table_header_done = False

        # Empty line
        if not stripped:
            html_parts.append("<br>")
            continue

        # Headings
        if stripped.startswith("####"):
            text = _inline(html_mod.escape(stripped.lstrip("# ")))
            html_parts.append(f'<h4><i>{text}</i></h4>')
            continue
        if stripped.startswith("###"):
            text = _inline(html_mod.escape(stripped.lstrip("# ")))
            html_parts.append(f"<h3>{text}</h3>")
            continue
        if stripped.startswith("##"):
            text = _inline(html_mod.escape(stripped.lstrip("# ")))
            html_parts.append(f"<h2>{text}</h2>")
            continue
        if stripped.startswith("# "):
            text = _inline(html_mod.escape(stripped.lstrip("# ")))
            html_parts.append(f'<h1 align="center">{text}</h1>')
            continue

        # Horizontal rule
        if stripped in ("---", "***", "___"):
            html_parts.append("<hr>")
            continue

        # List items
        if re.match(r'^[-*]\s', stripped):
            text = _inline(html_mod.escape(stripped.lstrip("-* ")))
            html_parts.append(f"<p>  - {text}</p>")
            continue

        # Regular paragraph
        text = _inline(html_mod.escape(stripped))
        html_parts.append(f"<p>{text}</p>")

    if in_table:
        html_parts.append("</table>")
    if in_code:
        html_parts.append("</pre>")

    return "\n".join(html_parts)


def _inline(text):
    """Process inline markdown (bold, italic, code, links)."""
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    text = re.sub(r'`(.+?)`', r'<font face="Courier">\1</font>', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    return text


def sanitize(text):
    """Replace non-latin1 characters."""
    reps = {
        '\u2014': '--', '\u2013': '-', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2026': '...', '\u2022': '-',
        '\u00d7': 'x', '\u2248': '~', '\u2192': '->', '\u03c4': 'tau',
        '\u03c3': 'sigma', '\u00b1': '+/-', '\u2264': '<=', '\u2265': '>=',
    }
    for k, v in reps.items():
        text = text.replace(k, v)
    return text.encode('latin-1', errors='replace').decode('latin-1')


def convert(md_path, pdf_path):
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    md_text = sanitize(md_text)
    html_content = md_to_html(md_text)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=9)
    pdf.write_html(html_content)
    pdf.output(pdf_path)
    print(f"  -> {pdf_path}")


if __name__ == "__main__":
    papers = [
        ("E:/fly-brain/paper_embodied_drosophila.md",
         "E:/fly-brain/paper_embodied_drosophila.pdf"),
        ("E:/fly-brain/paper_embodied_drosophila_es.md",
         "E:/fly-brain/paper_embodied_drosophila_es.pdf"),
    ]
    for md, out in papers:
        print(f"Converting: {md}")
        convert(md, out)
    print("Done!")
