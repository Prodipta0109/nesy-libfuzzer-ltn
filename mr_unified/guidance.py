"""Coverage guidance for libFuzzer (spec Task 8): bucket() turns a gap into new edges, guidance_value() scales a gap.
Under Atheris this module must be imported inside atheris.instrument_imports, or its edges are invisible (see harness).
"""
import math

GUIDANCE_SCALE = 1e-3


def bucket(margin: float) -> None:
    """Coverage hook for libFuzzer: each threshold passed is a new edge, so larger gaps are kept in the corpus."""
    # Copied from mr_fuzz/mrs.py bucket(): the same seven thresholds, one new edge per decade of margin.
    if margin > 1e-15: pass
    if margin > 1e-12: pass
    if margin > 1e-9: pass
    if margin > 1e-6: pass
    if margin > 1e-3: pass
    if margin > 1e-1: pass
    if margin > 5e-1: pass


def guidance_value(gap: float, bound: float) -> float:
    """Gap over tolerance, scaled by GUIDANCE_SCALE and capped at 1; a NaN gap counts as a failure (1)."""
    if math.isnan(gap):
        return 1.0
    return min(1.0, gap / bound * GUIDANCE_SCALE)
