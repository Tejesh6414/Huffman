# Traditional Huffman vs. AI-Assisted Huffman Coding

An experimental Information Theory and Coding (ITC) study comparing **Static Order-0 Huffman Coding** against **Context-Aware AI-Assisted Huffman Coding** using a Neural Network (Multi-Layer Perceptron).

---

## 📌 Overview

Huffman coding is a classic lossless data compression algorithm that creates optimal prefix-free codes given a fixed symbol frequency distribution. However, traditional Huffman coding is **memoryless (Order-0)**: it assigns a fixed code to each character based solely on its global frequency, completely ignoring sequential context.

This project implements and evaluates an **AI-Assisted Huffman Compression** scheme:
- A neural language model (MLP) predicts conditional probabilities $P(\text{char}_{t} \mid \text{context}_{t-1})$ based on the preceding 3 characters (`CONTEXT_SIZE = 3`).
- A **dynamic, context-specific Huffman tree** is constructed on the fly for each step.
- The decoder reconstructs the exact same predictions and trees causally from the decoded symbols, achieving **100% lossless decoding without transmitting any trees**.

---

## 🚀 Key Features

* **Deterministic Huffman Engine**: Implements a min-heap priority queue with sequential tie-breaking (`itertools.count()`) to guarantee identical tree topologies across encoder and decoder.
* **Context Modeling (N-Gram Neural Predictor)**: Uses an `MLPClassifier` (64, 32 hidden units) with one-hot character representations to predict the probability distribution of the next symbol.
* **Tree Caching**: Caches built trees per unique 3-character n-gram to significantly reduce runtime overhead during encoding and decoding.
* **Causal Lossless Decoding**: Demonstrates full end-to-end reconstruction from the raw bitstream and initial seed context, verified with automated round-trip assertions.
* **Information-Theoretic Benchmarking**: Compares empirical bit rates against ASCII, Fixed-Length codes, Shannon Order-0 Entropy, and AI Cross-Entropy theoretical limits.
* **Honest Accounting of Total Cost**: Factoring in model parameter storage vs. payload savings.
* **Visual Analytics**: Generates comparison charts (`results.png`) showing character frequencies, compression ratios, and computational runtime.

---

## 📊 Performance & Results

Tested on `input.txt` (4,531 characters, 51 unique symbols, 80/20 train/test split):

| Compression Scheme | Bits / Symbol | % Saved vs. ASCII | Description |
| :--- | :---: | :---: | :--- |
| **ASCII Baseline** | `8.0000` | `0.00%` | Standard 8-bit byte representation |
| **Fixed-Length** | `6.0000` | `25.00%` | $\lceil \log_2 51 \rceil$ bits per symbol |
| **Order-0 Shannon Entropy** | `4.2586` | `46.77%` | Theoretical lower bound for memoryless codes |
| **Traditional Huffman** | `4.3363` | `45.80%` | Static tree built from training frequencies |
| **AI-Assisted Huffman** | **`3.4708`** | **`56.62%`** | **Context-aware dynamic Huffman bitstream** |
| **AI Cross-Entropy Bound** | `3.9410` | `50.74%` | $-\frac{1}{N} \sum \log_2 P(\text{target})$ theoretical bound |

> **Key Takeaway**: AI-assisted Huffman achieves **3.47 bits/symbol**, breaking well past the static Order-0 Shannon entropy limit of **4.26 bits/symbol** by exploiting inter-character dependencies.

---

## ⚖️ The "Honest Cost" Trade-Off

While the AI-assisted bitstream payload is noticeably smaller than traditional Huffman:
- **Traditional Huffman Payload**: 3,933 bits (+ tiny lookup table)
- **AI Huffman Payload**: 3,148 bits (saves 785 bits on the test set)
- **AI Model Parameter Storage**: 13,520 float32 weights = **432,640 bits** (~54 KB)

```
[ Traditional Huffman ]
████ (3,933 bits payload)

[ AI Huffman ]
███ (3,148 bits payload) + ████████████████████████████████████████ (432,640 bits model)
```

> **Conclusion**: On small files (a few KB), sending the neural model weights over the wire completely negates any compression gains. AI-assisted compression becomes practical when:
> 1. Encoding large datasets (hundreds of megabytes or gigabytes).
> 2. Both sender and receiver already share a pre-trained base model.

---

## 🧠 How It Works

```
ENCODING PIPELINE:
                  ┌────────────────────────┐
Context ("the") ─►│ Neural Network (MLP)   │──► P(next_char | "the")
                  └────────────────────────┘             │
                                                         ▼
                                             ┌───────────────────────┐
Actual Char (' ') ──────────────────────────►│ Build Huffman Tree    │──► Append Bits ('0')
                                             │ (Cached by context)   │
                                             └───────────────────────┘

DECODING PIPELINE (Causal):
                  ┌────────────────────────┐
Context ("the") ─►│ Neural Network (MLP)   │──► P(next_char | "the")
                  └────────────────────────┘             │
                                                         ▼
                                             ┌───────────────────────┐
Bitstream ('0') ────────────────────────────►│ Walk Huffman Tree     │──► Emits Char (' ')
                                             └───────────────────────┘           │
                                                         ▲                       │
                                                         └───────────────────────┘
                                                       Updates context for next char
```

### Why Doesn't the Tree Need to be Sent?
Traditional Huffman requires storing or transmitting code tables or tree structures so the decoder knows what `'010'` means. In this AI-assisted approach:
1. Both encoder and decoder have the same neural network weights.
2. The decoder already knows the initial 3 context characters (seed).
3. The decoder predicts the exact same distribution $P$, builds the exact same deterministic Huffman tree, and extracts the next character.
4. The decoded character becomes the context for the subsequent step.

---

## 📂 Project Structure

```text
ITC/
├── huffman.py          # Complete implementation (Traditional + AI Huffman + Benchmarks)
├── input.txt           # Sample input text dataset
├── results.png         # Generated comparison plots
├── pyrightconfig.json  # Language server / linter configuration
└── README.md           # Documentation and analysis
```

---

## 🛠️ Prerequisites & Installation

### Requirements
- Python 3.8+
- Virtual environment (recommended)

### Install Dependencies
```bash
pip install numpy matplotlib scikit-learn
```

---

## 💻 Running the Code

### Default Execution
Runs on the bundled `input.txt` and displays the GUI plots:
```bash
python huffman.py
```

### Running on a Custom File
```bash
python huffman.py path/to/your_file.txt
```

### Headless / CLI Mode (No Pop-up Windows)
Generates and saves `results.png` without blocking the terminal with interactive Matplotlib windows:
```bash
python huffman.py --no-plots
```

---

## 📈 Visual Outputs (`results.png`)

Running the script automatically produces a 3-panel figure:
1. **Character Frequency Distribution**: Frequency of the top 20 symbols in the corpus.
2. **Bits per Symbol Comparison**: Side-by-side comparison of Entropy, Traditional Huffman, AI Huffman, Cross-Entropy, and ASCII.
3. **Runtime Benchmarks**: Traditional encoding time vs. AI training and encoding time.

---

## 📚 References & Theoretical Foundations

* **Huffman, D. A. (1952)**: *"A Method for the Construction of Minimum-Redundancy Codes"*, Proceedings of the IRE.
* **Shannon, C. E. (1948)**: *"A Mathematical Theory of Communication"*, Bell System Technical Journal.
* **Context Tree Weighting / Prediction by Partial Matching (PPM)**: Foundation for higher-order statistical modeling in data compression.
