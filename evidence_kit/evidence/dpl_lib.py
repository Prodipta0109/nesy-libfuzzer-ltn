DPL = dict(name="dpl", cwd=__import__("os").environ.get("DPL_DIR","/home/claude/dpl"), script="dpl_fuzz.py", out_env="DPL_OUT", count_env="DPL_COUNT",
  bugmap={"D4": [r"FileNotFoundError"], "D7": [r"Fewer non-zero"], "D5": [r"ZeroDivisionError"], "D6": [r"annotated disjunction mass"],
          "D13": [r"from_cache"], "D14": [r"DataLoader.*never stops"], "D3": [r"stop_condition"], "D9": [r"unhashable"],
          "D10": [r"round trip"], "D11": [r"ConfusionMatrix"], "D12": [r"keeps drawing"], "D1/D2/D8 (t_names)": [r"exception:t_names"]})
