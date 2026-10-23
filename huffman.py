"""
Traditional Huffman vs AI-assisted (context-model) Huffman.

"""

import heapq
import itertools
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

CONTEXT_SIZE = 3
TRAIN_FRACTION = 0.8
EPSILON = 1e-4          # probability floor so every symbol always has a code
SHOW_PLOTS = "--no-plots" not in sys.argv


# ------------------------------------------------------------
# 1. READ INPUT
# ------------------------------------------------------------
args = [a for a in sys.argv[1:] if not a.startswith("--")]
input_path = Path(args[0]) if args else Path(__file__).with_name("input.txt")
text = input_path.read_text(encoding="utf-8")

split = int(len(text) * TRAIN_FRACTION)
train_text, test_text = text[:split], text[split:]

alphabet = sorted(set(text))                 # all symbols (sent as side info)
V = len(alphabet)
index = {c: i for i, c in enumerate(alphabet)}
FIXED_BITS = math.ceil(math.log2(V))

print("=" * 60)
print("        HUFFMAN vs AI-ASSISTED HUFFMAN (train/test)")
print("=" * 60)
print(f"File            : {input_path.name}")
print(f"Total characters: {len(text)}   Unique: {V}")
print(f"Train / Test    : {len(train_text)} / {len(test_text)} characters")
if len(text) < 2000:
    print("WARNING: very small file - ML results will be unreliable. "
          "Use a few KB of text for meaningful numbers.")


# ------------------------------------------------------------
# 2. HUFFMAN HELPERS (deterministic, so the decoder rebuilds the same tree)
# ------------------------------------------------------------
class Node:
    def __init__(self, char=None, left=None, right=None):
        self.char, self.left, self.right = char, left, right


def build_tree(weights):
    """weights: dict symbol -> positive weight. Returns root Node."""
    counter = itertools.count()              # tie-breaker => deterministic
    heap = [(w, next(counter), Node(char=c)) for c, w in sorted(weights.items())]
    heapq.heapify(heap)
    if len(heap) == 1:
        return heap[0][2]
    while len(heap) > 1:
        w1, _, a = heapq.heappop(heap)
        w2, _, b = heapq.heappop(heap)
        heapq.heappush(heap, (w1 + w2, next(counter), Node(left=a, right=b)))
    return heap[0][2]


def tree_to_codes(root):
    codes = {}

    def walk(node, code):
        if node.char is not None:
            codes[node.char] = code or "0"
            return
        walk(node.left, code + "0")
        walk(node.right, code + "1")

    walk(root, "")
    return codes


def decode_symbol(root, bits, pos):
    """Read bits from position pos until a leaf is reached."""
    node = root
    if node.char is not None:                # single-symbol tree
        return node.char, pos + 1
    while node.char is None:
        node = node.left if bits[pos] == "0" else node.right
        pos += 1
    return node.char, pos


# ------------------------------------------------------------
# 3. TRADITIONAL HUFFMAN  (built on TRAIN, measured on TEST)
# ------------------------------------------------------------
t0 = time.perf_counter()
train_freq = Counter(train_text)
# +1 smoothing: symbols that only appear in the test part still get a code
weights = {c: train_freq.get(c, 0) + 1 for c in alphabet}
trad_codes = tree_to_codes(build_tree(weights))
trad_bits = sum(len(trad_codes[c]) for c in test_text)
trad_time = time.perf_counter() - t0
trad_avg = trad_bits / len(test_text)

test_freq = Counter(test_text)
test_entropy = -sum((f / len(test_text)) * math.log2(f / len(test_text))
                    for f in test_freq.values())


# ------------------------------------------------------------
# 4. AI MODEL  (trained on TRAIN only)
# ------------------------------------------------------------
def make_xy(s, start=CONTEXT_SIZE):
    X = np.zeros((len(s) - start, CONTEXT_SIZE * V), dtype=np.float32)
    y = np.zeros(len(s) - start, dtype=int)
    for row, i in enumerate(range(start, len(s))):
        for k, c in enumerate(s[i - CONTEXT_SIZE:i]):
            X[row, k * V + index[c]] = 1.0
        y[row] = index[s[i]]
    return X, y


X_train, y_train = make_xy(train_text)
print("\nTraining MLP on the train part ...")
t0 = time.perf_counter()
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                      solver="adam", max_iter=500, random_state=42)
model.fit(X_train, y_train)
train_time = time.perf_counter() - t0
print(f"Training time : {train_time:.3f} s")

n_params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
model_bytes = n_params * 4                      # float32 storage


def probs_for(X):
    """Full-alphabet probability matrix with a floor, rows sum to 1."""
    raw = model.predict_proba(X)
    P = np.full((len(X), V), EPSILON, dtype=np.float64)
    P[:, model.classes_] += raw
    return P / P.sum(axis=1, keepdims=True)


# ------------------------------------------------------------
# 5. AI-ASSISTED HUFFMAN: ENCODE the TEST part
#    context for the first test symbols = last chars of the train part
# ------------------------------------------------------------
history = train_text[-CONTEXT_SIZE:] + test_text
X_test, y_test = make_xy(history)               # one row per test symbol
t0 = time.perf_counter()
P = probs_for(X_test)

tree_cache = {}                                  # context -> (root, codes)


def tree_for(ctx, prob_row):
    if ctx not in tree_cache:
        root = build_tree({alphabet[i]: float(p) for i, p in enumerate(prob_row)})
        tree_cache[ctx] = (root, tree_to_codes(root))
    return tree_cache[ctx]


encoded = []
for n, ch in enumerate(test_text):
    ctx = history[n:n + CONTEXT_SIZE]
    _, codes = tree_for(ctx, P[n])
    encoded.append(codes[ch])
bitstream = "".join(encoded)
ai_time = time.perf_counter() - t0
ai_avg = len(bitstream) / len(test_text)

cross_entropy = float(-np.mean(np.log2(P[np.arange(len(y_test)), y_test])))
accuracy = float(np.mean(P.argmax(axis=1) == y_test) * 100)


# ------------------------------------------------------------
# 6. DECODE and VERIFY (lossless check)
# ------------------------------------------------------------
decoded, pos = [], 0
for n in range(len(test_text)):
    ctx = (train_text[-CONTEXT_SIZE:] + "".join(decoded))[-CONTEXT_SIZE:]
    # decoder recomputes probabilities from the symbols it already decoded
    x = np.zeros((1, CONTEXT_SIZE * V), dtype=np.float32)
    for k, c in enumerate(ctx):
        x[0, k * V + index[c]] = 1.0
    root, _ = tree_for(ctx, probs_for(x)[0])
    sym, pos = decode_symbol(root, bitstream, pos)
    decoded.append(sym)

lossless = "".join(decoded) == test_text
print(f"\nLossless round trip (decode == original): {lossless}")


# ------------------------------------------------------------
# 7. RESULTS
# ------------------------------------------------------------
def pct_saving(b):
    return (1 - b / 8) * 100


rows = [
    ("Test symbols", f"{len(test_text)}", ""),
    ("ASCII baseline (bits/sym)", "8.0000", ""),
    ("Fixed-length (bits/sym)", f"{FIXED_BITS}", ""),
    ("Order-0 entropy of test", f"{test_entropy:.4f}", ""),
    ("Traditional Huffman", f"{trad_avg:.4f}", f"{pct_saving(trad_avg):.2f}% saved"),
    ("AI Huffman (real bitstream)", f"{ai_avg:.4f}", f"{pct_saving(ai_avg):.2f}% saved"),
    ("AI cross-entropy (ideal bound)", f"{cross_entropy:.4f}", f"{pct_saving(cross_entropy):.2f}% saved"),
    ("AI next-char accuracy", f"{accuracy:.2f}%", ""),
]
print("\n" + "=" * 60)
print("                    RESULTS (on TEST part)")
print("=" * 60)
for name, val, note in rows:
    print(f"{name:34}{val:>10}  {note}")

if ai_avg < cross_entropy:
    print("\nNote: real AI Huffman bits < cross-entropy. This is possible because "
          "cross-entropy\n      punishes confident wrong predictions without limit "
          "(-log2 p), while a Huffman\n      code length is capped by the tree depth. "
          "Treat both numbers as noisy on a small test set.")

print(f"\nEncoding time   : traditional {trad_time:.6f}s | AI {ai_time:.4f}s "
      f"(+ {train_time:.3f}s training)")

payload_bits = len(bitstream)
print("\n--- Honest total cost (test part only) ---")
print(f"Traditional : {trad_bits} bits payload (+ small code table)")
print(f"AI Huffman  : {payload_bits} bits payload + model {model_bytes * 8} bits "
      f"({n_params} weights)")
if payload_bits + model_bytes * 8 > trad_bits:
    print("-> The model's own size outweighs its savings on a file this small.")
else:
    print("-> The AI scheme wins even after counting the model size.")
print("Note: Huffman needs >= 1 bit/symbol; arithmetic coding could approach "
      "the cross-entropy bound.")


# ------------------------------------------------------------
# 8. GRAPHS
# ------------------------------------------------------------
def label(c):
    return {" ": "[SP]", "\n": "[NL]", "\t": "[TAB]"}.get(c, c)


TOP_N = 20                                   # keep the x-axis readable
items = sorted(Counter(text).items(), key=lambda kv: kv[1], reverse=True)[:TOP_N]

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

ax[0].bar([label(c) for c, _ in items], [f for _, f in items], color="steelblue")
ax[0].set_title(f"Top {TOP_N} character frequencies (whole file)")
ax[0].tick_params(axis="x", rotation=60)

names = ["Order-0\nentropy", "Traditional\nHuffman", "AI Huffman", "AI cross-\nentropy", "ASCII"]
vals = [test_entropy, trad_avg, ai_avg, cross_entropy, 8]
ax[1].bar(names, vals, color=["mediumpurple", "darkorange", "seagreen", "lightgreen", "gray"])
for i, v in enumerate(vals):
    ax[1].text(i, v + 0.1, f"{v:.2f}", ha="center")
ax[1].set_ylim(0, 9)
ax[1].set_title("Bits per symbol (test part)")

ax[2].bar(["Traditional\nencode", "AI\ntraining", "AI\nencode"],
          [trad_time, train_time, ai_time],
          color=["steelblue", "mediumpurple", "seagreen"])
ax[2].set_title("Runtime (seconds)")

plt.tight_layout()
plt.savefig(Path(__file__).with_name("results.png"), dpi=120)
if SHOW_PLOTS:
    plt.show()

print("\nDone. Plot saved as results.png")