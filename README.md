# Deep Learning Coursework: Fact Checking

This project checks whether individual facts from ChatGPT-generated biographies are supported by retrieved Wikipedia passages.

The program returns:

- `S` — supported by the evidence
- `NS` — not supported by the evidence

## Repository contents

- `factchecking_main.py` — provided command-line driver. It loads the data, selects a checker, and prints evaluation metrics.
- `factcheck.py` — fact-checking implementations.
- `data/dev_labeled_ChatGPT.jsonl` — human-labelled development examples.
- `data/passages_bm25_ChatGPT_humfacts.jsonl` — retrieved Wikipedia evidence passages.
- `requirements.txt` — pinned Python dependencies.
- `test_targets.sh` — reproducible shell test runner.
- `tune_entailment.py` — development-set tuning script for sentence count and probability threshold.

## Setup

Python 3.10 is recommended. A local virtual environment is already configured in `.venv`.

### Windows

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux or macOS

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the available modes

```bash
python factchecking_main.py --mode word_overlap
python factchecking_main.py --mode entailment
```

The entailment mode downloads the DeBERTa-v3 model from Hugging Face the first time it runs. It may take time and requires network access. Use `--cuda` when a CUDA-enabled PyTorch build and supported GPU are available. If CUDA is unavailable, the program safely falls back to CPU:

```bash
python factchecking_main.py --mode entailment --cuda
```

## Implemented methods

### Word overlap

`WordRecallThresholdFactChecker`:

- removes common stop words;
- applies light stemming;
- calculates weighted recall over the retrieved passages;
- predicts `S` when the score reaches the tuned threshold.

### Textual entailment

`EntailmentModel`:

- sends each premise/hypothesis pair to DeBERTa;
- converts logits into entailment, neutral, or contradiction probabilities.

`EntailmentFactChecker` preprocesses evidence before inference:

- splits retrieved passages into sentences;
- removes empty and duplicate sentences;
- ranks sentences by overlap with the fact;
- sends the five most relevant sentences per passage to DeBERTa;
- uses the maximum entailment probability across candidates;
- predicts `S` when that probability reaches the tuned threshold of `0.333`.

The required model is:

```text
MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
```

Measured development-set accuracy improved as follows:

| Version | Accuracy |
|---|---:|
| Initial full-passage label decision | 164/221 (74.21%) |
| Sentence splitting and overlap pruning | 182/221 (82.35%) |
| Probability threshold tuning | 185/221 (83.71%) |
| Joint sentence-count and threshold tuning | 186/221 (84.16%) |

The tuning script searches sentence counts and thresholds on the supplied development set:

```bash
python tune_entailment.py
```

The reported score is development-set performance and may not represent unseen test-set accuracy.

### Optional dependency parsing

The `--mode parsing` checker is also implemented. It compares dependency relations between the fact and each passage. It requires the spaCy English model:

```bash
python -m spacy download en_core_web_sm
python factchecking_main.py --mode parsing
```

This mode is optional.

## Test commands

Run the saved shell runner from Linux/macOS or Git Bash:

```bash
chmod +x test_targets.sh
./test_targets.sh
```

On Windows PowerShell, run the target commands directly:

```powershell
.\.venv\Scripts\python.exe factchecking_main.py --mode word_overlap
.\.venv\Scripts\python.exe factchecking_main.py --mode entailment
```

The word-overlap and entailment commands should finish without errors.

## Current validation

- Word overlap: `179/221` (**80.995%**), above the 75% target.
- Entailment: `186/221` (**84.16%**), above the approximate 83% target.
- Python syntax and dependency checks pass in the local `.venv`.
- CPU execution is supported. GPU execution requires a CUDA-enabled PyTorch installation; `--cuda` falls back to CPU when CUDA is unavailable.

## Work completed

### Repository and environment

- Removed generated modernization/editor artifacts that were unrelated to the project.
- Kept the provided driver interface and data-file layout.
- Created and used a Python 3.10 virtual environment in `.venv`.
- Installed the dependencies from `requirements.txt`.
- Verified the environment with `pip check`.
- Added a reusable shell test runner instead of a PowerShell test script.
- Added this README and a reproducible tuning script.

### Word-overlap checker

`WordRecallThresholdFactChecker` now:

1. Converts text to lowercase.
2. Extracts alphanumeric tokens.
3. Removes common stop words.
4. Applies light suffix stemming.
5. Combines tokens from all retrieved passages.
6. Gives higher weight to content tokens such as names, places, dates, and numbers.
7. Predicts `S` when the weighted recall reaches the tuned threshold `0.775`; otherwise it predicts `NS`.

Measured result:

```text
Correct: 179/221
Accuracy: 0.8099547511312217 (80.995%)
F1 for S: 0.7961165048543688
F1 for NS: 0.8220338983050847
```

### DeBERTa entailment checker

The required model is loaded by the provided driver:

```text
MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
```

`EntailmentModel.check_entailment`:

- tokenizes a premise and hypothesis pair;
- runs the DeBERTa sequence-classification model;
- applies softmax to the logits;
- returns the predicted label plus entailment, neutral, and contradiction probabilities.

`EntailmentFactChecker` now implements all requested processing methods:

- `__init__` — stores the entailment model. The incorrect `_init_` spelling is not used.
- `sentence_splitting` — separates retrieved passages into sentence-sized premises.
- `noise_removal` — removes empty fragments, normalizes whitespace, and removes duplicates.
- `word_overlap_pruning` — ranks evidence sentences by token overlap with the fact.
- `predict` — evaluates the selected sentences and uses the strongest entailment probability.

The final production settings are:

```text
MAX_SENTENCES_PER_PASSAGE = 5
ENTAILMENT_THRESHOLD = 0.333
```

The prediction uses the following output contract:

```text
S  = supported
NS = not supported
```

### Accuracy improvement history

All figures below use the 221 supplied development examples:

| Stage | Correct | Accuracy | Change from initial |
|---|---:|---:|---:|
| Initial full-passage label decision | 164/221 | 74.21% | — |
| Sentence splitting, duplicate removal, and overlap pruning | 182/221 | 82.35% | +8.14 percentage points |
| Probability threshold `0.35` | 185/221 | 83.71% | +9.50 percentage points |
| Joint tuning: 3 sentences and threshold `0.367` | 186/221 | 84.16% | +9.95 percentage points |
| Final joint tuning: 5 sentences and threshold `0.333` | 186/221 | 84.16% | +9.95 percentage points |

The final score is 22 more correct predictions than the initial implementation and exceeds the approximate 83% entailment target.

### Additional ranking experiment

Entity-, date-, and number-aware sentence ranking was evaluated against the normal overlap ranking. The experiment tested:

- token overlap ranking;
- entity-aware ranking;
- date/number-aware ranking;
- combined entity/date/number ranking.

The best alternative reached `183/221` (**82.81%**), compared with the final production result of `186/221` (**84.16%**). It was therefore not adopted; the production implementation keeps the simpler overlap ranking because it performed better on the supplied development set.

### Tuning and reproducibility

`[tune_entailment.py](./tune_entailment.py)` searches:

- sentence counts from 1 through 6;
- thresholds from `0.250` through `0.500`;
- increments of `0.001`.

It caches model scores for the candidate sentences, evaluates every sentence-count/threshold combination, and prints the best measured configuration. The threshold range was selected after an initial coarse sweep found the useful region around `0.35`–`0.40`; the wider `0.25`–`0.50` range adds a safety margin without testing thresholds near 0 or 1 that would classify nearly every example identically.

Run it with:

```bash
python tune_entailment.py
```

### Optional dependency parser

`DependencyRecallThresholdFactChecker` is implemented as an optional checker. It:

- loads spaCy's `en_core_web_sm` model;
- extracts dependency relations;
- computes fact-dependency recall for each passage;
- predicts `S` when the best recall reaches `0.5`.

Install the optional model before running it:

```bash
python -m spacy download en_core_web_sm
python factchecking_main.py --mode parsing
```

### CPU, CUDA, and model downloads

The local environment currently uses CPU-only PyTorch (`2.5.0+cpu`). The NVIDIA RTX 4050 is present, but the CUDA-enabled PyTorch wheel installation failed because its approximately 2.5 GB download timed out. This affects runtime speed, not the measured model logic or accuracy.

The driver now checks CUDA availability. If `--cuda` is supplied while PyTorch has no CUDA support, it prints a fallback message and continues on CPU instead of crashing.

The DeBERTa model must be downloaded from Hugging Face on its first run. Network access is required, and the model is cached afterward.

### Remaining project item

The implementation and target evaluations are complete. A separate written report, if required, should document:

- the methodology;
- aggregate accuracy and per-class statistics;
- false positives and false negatives;
- two to four error categories;
- at least three detailed error examples;
- limitations and possible improvements.
