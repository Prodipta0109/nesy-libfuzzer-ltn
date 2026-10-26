"""Case log for the byte-driven harness: one JSON line per decoded case, and a reader that prints the lines.
Write: python -m mr_unified.harness --random N --log cases.jsonl (or add --log to an Atheris run).
Read:  python -m mr_unified.case_log cases.jsonl [--kind K] [--limit N]
"""
import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence, TextIO

from mr_unified.cases import Case, format_env, format_node
from mr_unified.findings_io import strict_value
from mr_unified.oracle import ALL_REWRITES, Violation
from mr_unified.rewrites import apply_at

_HANDLE: Optional[TextIO] = None
_COUNT = 0


def open_log(path: str) -> None:
    """Start logging to `path`, replacing it. Lines are flushed one by one, so an Atheris run that exits keeps them."""
    global _HANDLE, _COUNT
    close_log()
    _HANDLE = open(path, "w", buffering=1)
    _COUNT = 0


def close_log() -> int:
    """Stop logging and return how many cases were written."""
    global _HANDLE
    if _HANDLE is not None:
        _HANDLE.close()
        _HANDLE = None
    return _COUNT


def is_open() -> bool:
    """Whether a log is being written."""
    return _HANDLE is not None


def case_record(
    case: Case, run: int, skipped: int, violations: Sequence[Violation], data: bytes = b"", selection: str = ""
) -> Dict[str, Any]:
    """The JSON-ready record of one case: the input bytes, what was drawn, what it was rewritten to, and the outcome.
    input_hex drops trailing zeros, which decode reads the same as padding."""
    after = None
    if case.kind in ("rewrite", "qrewrite") and case.path is not None:
        rewritten = apply_at(case.node, ALL_REWRITES[case.name], case.path)
        after = None if rewritten is None else format_node(rewritten)
    return {
        "n": _COUNT,
        "lib": selection,
        "input_len": len(data),
        "input_hex": data.rstrip(b"\x00").hex(),
        "kind": case.kind,
        "name": case.name,
        "path": None if case.path is None else list(case.path),
        "formula": format_node(case.node),
        "after": after,
        "env": format_env(case.env),
        "run": run,
        "skipped": skipped,
        "violations": [v.lib + " " + v.rule for v in violations],
    }


def record(case: Case, run: int, skipped: int, violations: Sequence[Violation], data: bytes, selection: str) -> None:
    """Append one case to the log; does nothing when no log is open."""
    global _COUNT
    if _HANDLE is None:
        return
    _HANDLE.write(json.dumps(strict_value(case_record(case, run, skipped, violations, data, selection)), sort_keys=True) + "\n")
    _COUNT += 1


def _env_lines(env: Dict[str, Any]) -> List[str]:
    """Readable lines for a format_env dict: atoms, domain, and one line per non-empty predicate table."""
    atoms = " ".join("%s=%.3g" % (name, value) for name, value in sorted(env["atoms"].items()))
    lines = ["atoms:   " + atoms, "domain:  " + (",".join(env["domain"]) or "-")]
    for name, table in sorted(env["preds"].items()):
        if table:
            lines.append("%s: %s" % (name.ljust(7), " ".join("%s(%s)=%.3g" % (name, key, val) for key, val in table.items())))
    return lines


def _hex_head(text: str, shown: int = 24) -> str:
    """The first `shown` bytes of a hex string, with an ellipsis when there is more."""
    return text[:2 * shown] + ("..." if len(text) > 2 * shown else "") if text else "(all zero)"


def render(rec: Dict[str, Any], outcome: bool = True) -> str:
    """One case as text: a header line, the formula, its rewrite when there is one, and the env.
    With outcome=False the header leaves out run, skipped and violations (for a replay that ran no check)."""
    where = "" if rec["path"] is None else " path=%s" % rec["path"]
    bad = ", ".join(rec["violations"]) or "none"
    tail = "  run=%d skipped=%d violations=%s" % (rec["run"], rec["skipped"], bad) if outcome else ""
    lines = [
        "#%d %s %s%s%s" % (rec["n"], rec["kind"], rec["name"] or "-", where, tail),
        "  input:   %d bytes, hex %s" % (rec.get("input_len", 0), _hex_head(rec.get("input_hex", ""))),
        "  formula: " + rec["formula"],
    ]
    if rec["after"] is not None:
        lines.append("  after:   " + rec["after"])
    lines.extend("  " + line for line in _env_lines(rec["env"]))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """Print the cases of a log file, optionally one kind and the first N, then a count per kind."""
    parser = argparse.ArgumentParser(description="Print the cases a harness run logged with --log.")
    parser.add_argument("path", help="the .jsonl file written by --log")
    parser.add_argument("--kind", default=None, help="only this kind: rewrite, qrewrite, env_rule, property or cross")
    parser.add_argument("--limit", type=int, default=None, help="print at most this many cases")
    args = parser.parse_args(argv)
    counts: Dict[str, int] = {}
    shown = 0
    with open(args.path) as handle:
        for line in handle:
            rec = json.loads(line)
            if args.kind is not None and rec["kind"] != args.kind:
                continue
            counts[rec["kind"]] = counts.get(rec["kind"], 0) + 1
            if args.limit is None or shown < args.limit:
                print(render(rec) + "\n")
                shown += 1
    print("cases: " + ", ".join("%s=%d" % (kind, n) for kind, n in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
