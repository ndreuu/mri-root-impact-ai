"""Escaping and stable entry points for the four-section physician report."""

import html

SIDES = {"left": "слева", "right": "справа", "unknown": "сторона не установлена"}


def _h(value):
    return html.escape(str(value), quote=True)


def _m(value):
    text = _h(value).replace("\\", "\\\\").replace("\n", " ")
    for char in "`*_[]()!|":
        text = text.replace(char, "\\" + char)
    return text


def _root(value):
    if value is None:
        return "Не назначен: условия анатомической привязки не выполнены"
    role = {"exiting": "выходящий", "traversing": "проходящий"}[value["role"]]
    return "{} корешок {} {}".format(role, value["root"], SIDES[value["side"]])



def render_markdown(report, evidence=None):
    """Render four concise sections with source images and fusion conclusions."""
    from .evidence_report import render_markdown as render
    return render(report, evidence)


def render_html(report, evidence=None):
    """Render the same checked findings as self-contained HTML."""
    from .evidence_report import render_html as render
    return render(report, evidence)
