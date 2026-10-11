LTN = dict(name="ltn", cwd=__import__("os").environ.get("LTN_DIR","/tmp/rv/ltn_combined"), script="ltn_fuzz_combined.py", out_env="LTNFUZZ_FINDINGS", count_env=None,
  src=__import__("os").environ.get("LTN_SRC","/tmp/rv/ltn_torch"), base_env={"LTNFUZZ_COLLECT": "1"},
  bugmap={"F1 empty domain -> NaN": [r"empty_domain=True"], "F2/F3 ZeroDivision (p=0, unused var)": [r"ZeroDivisionError"],
          "F24-F26,F29 oracle violations (dtype/formula/metamorphic/structure)": [r"DtypeViolation|FormulaViolation|MetamorphicViolation|StructureViolation"],
          "F27 gradient NaN (unstable ops)": [r"GradientViolation"], "F30-F33 API/misuse errors": [r"ValueError@core|IndexError|RuntimeError"]})
