
# PART 3: ERROR ANALYSIS
# Identify False Positives and False Negatives

import json
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from factchecking_main import read_passages, read_fact_examples
from factcheck import EntailmentModel, EntailmentFactChecker


# STEP 01: LOAD DATASET
LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"

fact_to_passage_dict = read_passages(PASSAGES_PATH)

examples = read_fact_examples(
    LABELS_PATH,
    fact_to_passage_dict
)

print(f"Total examples: {len(examples)}")

# STEP 02: LOAD PRETRAINED DEBERTA MODEL
model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name
)

use_cuda = torch.cuda.is_available()

if use_cuda:
    model.to("cuda")

model.eval()

ent_model = EntailmentModel(
    model,
    tokenizer,
    use_cuda
)

# STEP 03: INITIALIZE THE FACT CHECKER
fact_checker = EntailmentFactChecker(ent_model)

# STEP 04: IDENTIFY CLASSIFICATION ERRORS
false_positives = []
false_negatives = []

correct_predictions = 0

for index, example in enumerate(examples):

    # Convert IR into NS for binary evaluation
    actual_label = (
        "NS" if example.label == "IR" else example.label
    )

    # Obtain prediction using the existing implementation
    predicted_label = fact_checker.predict(
        example.fact,
        example.passages
    )

    # Count correct predictions
    if actual_label == predicted_label:
        correct_predictions += 1

    # Store false positives
    elif actual_label == "NS" and predicted_label == "S":

        false_positives.append({
            "index": index,
            "error_type": "False Positive",
            "fact": example.fact,
            "ground_truth": actual_label,
            "original_label": example.label,
            "prediction": predicted_label,
            "passages": example.passages
        })

    # Store false negatives
    elif actual_label == "S" and predicted_label == "NS":

        false_negatives.append({
            "index": index,
            "error_type": "False Negative",
            "fact": example.fact,
            "ground_truth": actual_label,
            "original_label": example.label,
            "prediction": predicted_label,
            "passages": example.passages
        })

# STEP 05: DISPLAY EVALUATION RESULTS
total = len(examples)

accuracy = correct_predictions / total

print("\n========== EVALUATION RESULTS ==========")

print(f"Total examples       : {total}")
print(f"Correct predictions  : {correct_predictions}")
print(f"Accuracy             : {accuracy:.4%}")

print(f"\nFalse Positives      : {len(false_positives)}")
print(f"False Negatives      : {len(false_negatives)}")

# STEP 06: SAVE ALL INCORRECT EXAMPLES
error_results = {
    "accuracy": accuracy,
    "total_examples": total,
    "correct_predictions": correct_predictions,
    "false_positive_count": len(false_positives),
    "false_negative_count": len(false_negatives),
    "false_positives": false_positives,
    "false_negatives": false_negatives
}

OUTPUT_PATH = "entailment_errors.json"

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:

    json.dump(
        error_results,
        file,
        indent=4,
        ensure_ascii=False
    )

print(f"\nError analysis data saved to: {OUTPUT_PATH}")

# STEP 07: DISPLAY FALSE POSITIVES
print("\n========== FALSE POSITIVES ==========")

for i, error in enumerate(false_positives, start=1):

    print(f"\nFP {i}")
    print("Fact:", error["fact"])
    print("Ground Truth:", error["ground_truth"])
    print("Prediction:", error["prediction"])

    for j, passage in enumerate(error["passages"], start=1):
        print(f"\nEvidence Passage {j}:")
        print(passage["text"])

# STEP 08: DISPLAY FALSE NEGATIVES
print("\n========== FALSE NEGATIVES ==========")

for i, error in enumerate(false_negatives, start=1):

    print(f"\nFN {i}")
    print("Fact:", error["fact"])
    print("Ground Truth:", error["ground_truth"])
    print("Prediction:", error["prediction"])

    for j, passage in enumerate(error["passages"], start=1):
        print(f"\nEvidence Passage {j}:")
        print(passage["text"])
