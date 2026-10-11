"""Builds evidence_report.html from results/*.json (self-contained page, no network needed)."""
import json, glob, os
H = os.path.dirname(os.path.abspath(__file__)); R = H + "/results"
def load(p):
    try: return json.load(open(p))
    except Exception: return None
LIBS = [("lnn", "LNN"), ("ltn", "LTN"), ("dpl", "DeepProbLog")]
data = {}
for k, name in LIBS:
    d = {"name": name, "pairs": (load(f"{R}/PAIRS_{k}.json") or {}).get("rows", []), "dims": [], "adapt": []}
    for f in sorted(glob.glob(f"{R}/{k}_DIM-*.json")): d["dims"].append(load(f))
    for f in sorted(glob.glob(f"{R}/{k}_A*.json"), key=lambda p: int(''.join(c for c in os.path.basename(p).split('_A')[1] if c.isdigit()))): d["adapt"].append(load(f))
    data[k] = d
html = open(H + "/report_template.html").read().replace("/*DATA*/null", json.dumps(data))
open(H + "/evidence_report.html", "w").write(html); print("written", len(html))
