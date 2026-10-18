"""A 40-line fuzzer with the same loop as the real one. No LTN, no Atheris. Run: python toy_fuzzer.py"""
import random
random.seed(1)

def harness(data):                       # 1. bytes -> a tiny "program"  (this is Builder)
    rows = data[0] % 6                   #    question 1: how many rows?  (0..5)
    p = [1, 2, 3, 0][data[1] % 4]        #    question 2: which p?        (1,2,3 or 0)
    return rows, p

def library(rows, p, hit):               # 2. the thing being tested  (this is LTN)
    hit.add("start")
    if rows == 0:
        hit.add("empty-table branch")
        return "BUG  empty table gives nan"
    hit.add("normal-table branch")
    if p == 0:
        hit.add("p-zero branch")
        return "BUG  divide by zero"
    hit.add(f"p={p} branch")
    return "ok"

seen = set()                             # every branch ever ticked
corpus = [bytes([3, 1])]                 # starting input: 2 bytes
bugs = set()
print(f"{'step':>4}  {'parent':>9} -> {'child':>9}  {'rows':>4} {'p':>2}  {'result':26} new branches            decision")
for step in range(1, 31):
    parent = random.choice(corpus)                                  # 3. pick a saved input
    child = bytearray(parent)
    pos = random.randrange(len(child))
    child[pos] = random.randrange(256)                              # 4. MUTATE: change one byte
    child = bytes(child)
    rows, p = harness(child)                                        # 5. decode the bytes
    hit = set()
    result = library(rows, p, hit)                                  # 6. run the library
    new = hit - seen                                                # 7. any branch never seen before?
    if new:
        seen |= hit; corpus.append(child)
    if result.startswith("BUG"): bugs.add(result)
    decision = "KEEP (new branch)" if new else "throw away"
    print(f"{step:>4}  {list(parent)!s:>9} -> {list(child)!s:>9}  {rows:>4} {p:>2}  {result:26} {', '.join(sorted(new)) or '-':22} {decision}")
print("\nbugs found:", sorted(bugs))
print("saved corpus:", [list(c) for c in corpus])
