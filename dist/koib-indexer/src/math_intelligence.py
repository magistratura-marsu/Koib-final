"""Best-effort linearization of Office Math (OMML), retaining exact source XML.

Unlike flattening text nodes, nested fraction/script/root relationships survive.
This is structural extraction, *not* mathematical OCR or formula validation.
"""
from __future__ import annotations


def _local(node):
    return node.tag.rsplit("}", 1)[-1]


def _children(node, tag):
    return [child for child in node if _local(child) == tag]


def linearize_omml(node) -> str:
    kind = _local(node)
    if kind == "t":
        return node.text or ""
    if kind in {"f", "sSup", "sSub", "sSubSup", "rad", "nary", "limLow", "limUpp"}:
        fields = {}
        for child in node:
            key = _local(child)
            if key in {"num", "den", "e", "sub", "sup", "deg", "lim"}:
                fields[key] = linearize_omml(child)
        e = fields.get("e", "")
        if kind == "f":
            return f"({fields.get('num', '')})/({fields.get('den', '')})"
        if kind == "sSup":
            return f"({e})^({fields.get('sup', '')})"
        if kind == "sSub":
            return f"({e})_({fields.get('sub', '')})"
        if kind == "sSubSup":
            return f"({e})_({fields.get('sub', '')})^({fields.get('sup', '')})"
        if kind == "rad":
            return f"root[{fields.get('deg', '2') or '2'}]({e})"
        if kind == "nary":
            return f"sum_({fields.get('sub', '')})^({fields.get('sup', '')})({e})"
        if kind == "limLow":
            return f"({e})_({fields.get('lim', '')})"
        if kind == "limUpp":
            return f"({e})^({fields.get('lim', '')})"
    return "".join(linearize_omml(child) for child in node)


def docx_formulas(paragraph) -> list[tuple[str, str]]:
    from lxml import etree
    found = []
    for node in paragraph._p.iter():
        if _local(node) not in {"oMath", "oMathPara"}:
            continue
        if _local(node) == "oMathPara":
            continue  # child oMath processed separately
        formula = linearize_omml(node).strip()
        if formula:
            found.append((formula, etree.tostring(node, encoding="unicode")))
    return found
