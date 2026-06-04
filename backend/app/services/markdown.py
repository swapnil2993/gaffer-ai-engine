"""Markdown to HTML conversion utilities."""

import html as _html
import re


def md_to_html(md: str) -> str:
    """Convert Markdown to HTML for Docling output (headings, tables, lists).

    Focused on structural elements from Docling (notably tables); inline
    formatting is escaped rather than parsed. No external markdown dependency.
    """

    def esc(text):
        # Drop markdown link syntax [label](url) -> label, then HTML-escape.
        return _html.escape(re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text))

    lines = md.split("\n")
    out, i, n = [], 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        # Table: a row with pipes followed by a separator row (---|---).
        if "|" in s and i + 1 < n and set(lines[i + 1].strip()) <= set("|-: "):
            header = [c.strip() for c in s.strip("|").split("|")]
            i += 2
            rows = []
            while i < n and "|" in lines[i]:
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            th = "".join(f"<th>{esc(c)}</th>" for c in header)
            body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
            out.append(f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>")
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{esc(m.group(2))}</h{lvl}>")
            i += 1
            continue
        if re.match(r"^[-*]\s+", s):
            items = []
            while i < n and re.match(r"^[-*]\s+", lines[i].strip()):
                items.append(esc(re.sub(r"^[-*]\s+", "", lines[i].strip())))
                i += 1
            out.append("<ul>" + "".join(f"<li>{it}</li>" for it in items) + "</ul>")
            continue
        if set(s) <= set("-") and len(s) >= 3:
            out.append("<hr>")
            i += 1
            continue
        para = [s]
        i += 1
        while (
            i < n
            and lines[i].strip()
            and "|" not in lines[i]
            and not re.match(r"^(#{1,6}\s|[-*]\s)", lines[i].strip())
        ):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{esc(' '.join(para))}</p>")
    return "\n".join(out)
