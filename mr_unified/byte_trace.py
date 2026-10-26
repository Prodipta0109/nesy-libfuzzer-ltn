"""Byte trace: replay one fuzz input through cases.decode and print which byte made which decision.
Run: python -m mr_unified.byte_trace --bin mr_unified/findings/ltn/<rule>/input.bin --lib ltn
     python -m mr_unified.byte_trace --log cases.jsonl --case 3
     python -m mr_unified.byte_trace --hex 0302a1 --lib ref
Decoding is pure, so the replay builds the same case the fuzzer ran. cases.py is not changed: the tracing classes
stand in for Bytes and ByteRandom only while a replay runs.
"""
import argparse
import contextlib
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Sequence, Tuple

from mr_unified import case_log, cases
from mr_unified.adapters.base import Adapter
from mr_unified.cases import ByteRandom, Bytes, Case

PLUMBING = (Bytes, ByteRandom)
HEX_SHOWN = 32


@dataclass
class Draw:
    """One byte read: where it sat in the input, its value, who asked for it, how, and what it turned into."""

    pos: int
    byte: int
    who: str
    how: str
    result: str = ""


def _is_plumbing(frame: Any) -> bool:
    """True for frames inside Bytes, ByteRandom or the stdlib random module: they are how a draw is made, not who made it."""
    return isinstance(frame.f_locals.get("self"), PLUMBING) or frame.f_code.co_filename == random.__file__


def _caller() -> Tuple[str, str]:
    """(who, how) of the byte being read: the first non-plumbing frame, and the outermost plumbing call below it."""
    frame = sys._getframe(1)
    how = ""
    while frame is not None and _is_plumbing(frame):
        how = frame.f_code.co_name
        frame = frame.f_back
    if frame is None:
        return "?", how
    return "%s %s:%d" % (frame.f_code.co_name, os.path.basename(frame.f_code.co_filename), frame.f_lineno), how


class TracingBytes(Bytes):
    """Bytes that records every read in `trace`."""

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.trace: List[Draw] = []

    def byte(self) -> int:
        """The next byte, recorded with who asked for it."""
        pos = self.pos
        value = super().byte()
        who, how = _caller()
        self.trace.append(Draw(pos, value, who, how))
        return value

    def unit(self) -> float:
        """The next byte as a float in [0, 1), noted on its draw."""
        value = super().unit()
        self.trace[-1].result = "%.4f" % value
        return value

    def int(self, lo: int, hi: int) -> int:
        """The next byte as an integer in [lo, hi], noted on its draw."""
        value = super().int(lo, hi)
        self.trace[-1].result = "%d  (range %d..%d)" % (value, lo, hi)
        return value


class TracingRandom(ByteRandom):
    """ByteRandom whose choice() notes the picked item on its last draw; earlier draws of a choice were rejected."""

    def choice(self, seq: Sequence[Any]) -> Any:
        """random.Random.choice, with the result written into the trace."""
        trace = self.stream.trace
        start = len(trace)
        item = super().choice(seq)
        for draw in trace[start:-1]:
            draw.result = "rejected, drawn again"
        trace[-1].result = "%s  (1 of %d)" % (item, len(seq))
        return item


@contextlib.contextmanager
def tracing() -> Iterator[List[TracingBytes]]:
    """While active, cases.decode builds TracingBytes and TracingRandom; yields the list that receives the stream."""
    made: List[TracingBytes] = []

    def make_bytes(data: bytes) -> TracingBytes:
        stream = TracingBytes(data)
        made.append(stream)
        return stream

    original = (cases.Bytes, cases.ByteRandom)
    cases.Bytes, cases.ByteRandom = make_bytes, TracingRandom
    try:
        yield made
    finally:
        cases.Bytes, cases.ByteRandom = original


def trace_decode(data: bytes, adapters: Sequence[Adapter]) -> Tuple[Case, List[Draw], int]:
    """Decode `data` as the fuzzer does: the case, every byte read in order, and how many bytes were read."""
    with tracing() as made:
        case = cases.decode(data, adapters)
    return case, made[0].trace, made[0].pos


def render_trace(draws: Sequence[Draw]) -> str:
    """The reads as a table: position, byte, who asked, how, and the result."""
    width = max([len(draw.who) for draw in draws] + [len("decision")])
    lines = ["%4s  %-4s  %-*s  %-12s  %s" % ("pos", "byte", width, "decision", "draw", "result")]
    for draw in draws:
        lines.append("%4d  0x%02x  %-*s  %-12s  %s" % (draw.pos, draw.byte, width, draw.who, draw.how, draw.result))
    return "\n".join(lines)


def _load(args: argparse.Namespace) -> Tuple[bytes, str]:
    """The input bytes and the library selection to decode them for."""
    if args.bin:
        with open(args.bin, "rb") as handle:
            return handle.read(), args.lib or "ref"
    if args.hex is not None:
        return bytes.fromhex(args.hex), args.lib or "ref"
    with open(args.log) as handle:
        for line in handle:
            rec = json.loads(line)
            if rec["n"] == args.case:
                return bytes.fromhex(rec["input_hex"]), args.lib or rec.get("lib") or "ref"
    raise SystemExit("case %d is not in %s" % (args.case, args.log))


def main(argv: Optional[List[str]] = None) -> int:
    """Print the byte trace and the decoded case for one input."""
    parser = argparse.ArgumentParser(description="Show which fuzz byte made which decision when an input is decoded.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bin", help="a raw input file, e.g. a finding's input.bin")
    source.add_argument("--hex", help="the input as hex digits")
    source.add_argument("--log", help="a .jsonl file written by --log (use with --case)")
    parser.add_argument("--case", type=int, default=0, help="case number in the log")
    parser.add_argument("--lib", default=None, help="library selection the input was run with (default: the log's, else ref)")
    args = parser.parse_args(argv)
    from mr_unified.harness import make_adapters

    data, lib = _load(args)
    case, draws, used = trace_decode(data, make_adapters(lib))
    shown = data[:HEX_SHOWN].hex() + ("..." if len(data) > HEX_SHOWN else "")
    print("input: %d bytes (%s), decoded for lib=%s; %d bytes read, %d draws" % (len(data), shown or "empty", lib, used, len(draws)))
    print(render_trace(draws))
    print("\ndecoded case:")
    print(case_log.render(case_log.case_record(case, 0, 0, [], data, lib), outcome=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
