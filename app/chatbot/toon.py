"""Minimal TOON-format encoder for compact LLM context.

TOON (Token-Oriented Object Notation) is JSON's structure with YAML-like
syntax + table-form arrays. It typically uses 30-50% fewer tokens than
indented JSON for the catalog data we send to the LLM.

Output rules:
- dicts:      key: value, one per line, two-space indent for nesting
- primitives: bare (no quotes) unless they contain ":" or "\n"
- arrays of primitives: ``key[n]: a, b, c``
- arrays of dicts with uniform schema: table form

    products[2]{title,price,stock}:
      Maroon Dress, 2000, 33
      Saree, 999, 5

- arrays of dicts with non-uniform schema: fall back to one record per block
"""
from __future__ import annotations

from typing import Any


def encode(value: Any) -> str:
    """Render a JSON-shaped value as TOON. Returns a string without trailing newline."""
    lines: list[str] = []
    _encode_value(value, lines, indent=0, key=None)
    return "\n".join(lines)


def _encode_value(value: Any, out: list[str], *, indent: int, key: str | None) -> None:
    pad = "  " * indent
    if isinstance(value, dict):
        if key is not None:
            out.append(f"{pad}{key}:")
        inner_pad = "  " * (indent + 1) if key is not None else pad
        inner_indent = indent + 1 if key is not None else indent
        for k, v in value.items():
            _encode_kv(k, v, out, indent=inner_indent)
        return
    if isinstance(value, list):
        _encode_list(key, value, out, indent=indent)
        return
    # primitive at top level (rare)
    out.append(f"{pad}{_scalar(value)}")


def _encode_kv(key: str, value: Any, out: list[str], *, indent: int) -> None:
    pad = "  " * indent
    if isinstance(value, dict):
        out.append(f"{pad}{key}:")
        for k, v in value.items():
            _encode_kv(k, v, out, indent=indent + 1)
        return
    if isinstance(value, list):
        _encode_list(key, value, out, indent=indent)
        return
    out.append(f"{pad}{key}: {_scalar(value)}")


def _encode_list(key: str | None, items: list, out: list[str], *, indent: int) -> None:
    pad = "  " * indent
    n = len(items)
    if n == 0:
        if key is not None:
            out.append(f"{pad}{key}[0]:")
        return

    # Detect "table form" — list of dicts with the same key set
    if all(isinstance(it, dict) for it in items):
        schemas = [tuple(it.keys()) for it in items]
        uniform = all(s == schemas[0] for s in schemas)
        # Also require values to be scalars (no nested dicts/lists) for table form
        scalar_only = uniform and all(
            not isinstance(v, (dict, list))
            for it in items for v in it.values()
        )
        if scalar_only and key is not None:
            headers = ",".join(schemas[0])
            out.append(f"{pad}{key}[{n}]{{{headers}}}:")
            row_pad = "  " * (indent + 1)
            for it in items:
                cells = [_cell(it[h]) for h in schemas[0]]
                out.append(f"{row_pad}{', '.join(cells)}")
            return
        # Non-uniform / nested: fall through to record form below.

    # Primitive list: inline form
    if all(not isinstance(it, (dict, list)) for it in items) and key is not None:
        rendered = ", ".join(_cell(it) for it in items)
        out.append(f"{pad}{key}[{n}]: {rendered}")
        return

    # Record form: each item gets its own block
    if key is not None:
        out.append(f"{pad}{key}[{n}]:")
    inner_indent = indent + 1 if key is not None else indent
    inner_pad = "  " * inner_indent
    for it in items:
        if isinstance(it, dict):
            out.append(f"{inner_pad}-")
            for k, v in it.items():
                _encode_kv(k, v, out, indent=inner_indent + 1)
        elif isinstance(it, list):
            _encode_list(None, it, out, indent=inner_indent)
        else:
            out.append(f"{inner_pad}- {_scalar(it)}")


def _scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    # Bare string unless it contains characters that would break TOON parsing.
    if "\n" in s or s.startswith(("-", "[", "{")) or ": " in s:
        return '"' + s.replace('"', '\\"') + '"'
    return s


def _cell(v: Any) -> str:
    """Scalar inside an inline list / table row. Escapes commas."""
    s = _scalar(v)
    if "," in s and not s.startswith('"'):
        return '"' + s.replace('"', '\\"') + '"'
    return s
