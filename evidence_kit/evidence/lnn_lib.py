LNN = dict(name="lnn", cwd=__import__("os").environ.get("LNN_DIR","/home/claude/lnn"), script="lnn_fuzz.py", out_env="LNN_FINDINGS", count_env="LNN_COUNT",
  src=__import__("os").environ.get("LNN_DIR","/home/claude/lnn")+"/src", bugmap={"L1": [r"nonconvergence/t_weights"], "L2": [r"nonconvergence/t_conv"], "L3": [r"godel|frechet"],
          "L4": [r"true_hinge"], "L6": [r"contradiction"], "L7": [r"ValueError.*(Exists|arity|variable)|unary_operator"],
          "L8": [r"Length mismatch"]})
