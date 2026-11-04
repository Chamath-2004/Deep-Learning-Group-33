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
- `CW.pdf` — coursework brief.

## Setup

The coursework requires Python 3.10. A local virtual environment is already configured in `.venv`.

### Windows

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux or macOS

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the required coursework modes

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

`EntailmentFactChecker` predicts `S` when any passage is classified as entailment; otherwise it predicts `NS`.

### Optional dependency parsing

The `--mode parsing` checker is also implemented. It compares dependency relations between the fact and each passage. It requires the spaCy English model:

```bash
python -m spacy download en_core_web_sm
python factchecking_main.py --mode parsing
```

This mode is optional in the coursework and is not required for the main marks.

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

The coursework requires the word-overlap and entailment commands to finish without errors. The written submission must additionally report the model accuracy and analyse false positives and false negatives.
