"""Loss-aware structured table conversion and row-oriented retrieval records.

No LLM needed. Cell coordinates/headers remain visible in searchable text;
original grid is retained in chunk full_content. Intended for CSV/XLSX/DOCX/PDF/HTML.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence


def clean_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value).replace("\r", " ").replace("\n", " ⏎ ").strip()


def escape_cell(value: object) -> str:
    return clean_cell(value).replace("\\", "\\\\").replace("|", "\\|")


def table_markdown(rows: Sequence[Sequence[object]], headers: Sequence[object] | None = None) -> str:
    if not rows and not headers:
        return ""
    width = max([len(r) for r in rows] + ([len(headers)] if headers else []) + [1])
    if headers is None:
        headers = rows[0]
        rows = rows[1:]
    head = [escape_cell(headers[i]) if i < len(headers) else "" for i in range(width)]
    head = [cell or f"Колонка {i + 1}" for i, cell in enumerate(head)]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(escape_cell(row[i]) if i < len(row) else "" for i in range(width)) + " |")
    return "\n".join(lines)


def markdown_cells(line: str) -> list[str]:
    """Split a Markdown table row, not splitting escaped `\\|` in cells."""
    line = line.strip().strip("|")
    return [re.sub(r"\\([|\\])", r"\1", c.strip()) for c in re.split(r"(?<!\\)\|", line)]


def parse_markdown_grid(markdown: str) -> tuple[list[str], list[list[str]]]:
    lines = [line for line in markdown.splitlines() if line.strip().startswith("|")]
    if not lines:
        return [], []
    headers = markdown_cells(lines[0])
    start = 1
    if len(lines) > 1:
        maybe = markdown_cells(lines[1])
        if maybe and all(re.fullmatch(r":?-{3,}:?", x.strip()) for x in maybe):
            start = 2
    return headers, [markdown_cells(ln) for ln in lines[start:]]


def row_facts(headers: Sequence[str], values: Sequence[str], table_title: str, row_number: int) -> str:
    """Search text carrying *both* key and value. Preserve formulas verbatim."""
    pairs = []
    for idx, val in enumerate(values):
        if not val.strip():
            continue
        key = headers[idx].strip() if idx < len(headers) else f"Колонка {idx + 1}"
        pairs.append(f"{key}: {val.strip()}")
    return f"Таблица: {table_title or 'без названия'}. Строка {row_number}. " + "; ".join(pairs)


def row_windows(rows: Sequence[Sequence[str]], headers: Sequence[str], title: str,
                first_row: int = 1, max_chars: int = 1800,
                row_numbers: Sequence[int] | None = None) -> Iterable[tuple[int, int, str]]:
    """Yield bounded groups, never cut a row; oversized rows stand alone."""
    group: list[str] = []
    start = first_row
    size = 0
    for offset, row in enumerate(rows):
        number = row_numbers[offset] if row_numbers is not None else first_row + offset
        fact = row_facts(headers, row, title, number)
        if group and size + len(fact) + 1 > max_chars:
            last_number = (row_numbers[offset - 1] if row_numbers is not None
                           else number - 1)
            yield start, last_number, "\n".join(group)
            group, size, start = [], 0, number
        group.append(fact)
        size += len(fact) + 1
    if group:
        yield start, (row_numbers[-1] if row_numbers is not None
                      else first_row + len(rows) - 1), "\n".join(group)
