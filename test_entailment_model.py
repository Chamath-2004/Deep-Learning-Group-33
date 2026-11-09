
import time
import gc
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from factcheck import EntailmentModel


# ============================================================
# STEP 01: Load the pretrained DeBERTa model and tokenizer
# ============================================================

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

print("Loading DeBERTa model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Use evaluation mode
model.eval()

# Use CUDA when an available GPU is detected; otherwise fall back to CPU.
cuda = torch.cuda.is_available()
if cuda:
    model.to("cuda")

# Initialise your EntailmentModel class
entailment_model = EntailmentModel(model, tokenizer, cuda)

print("Model loaded successfully!")
print("Device:", "CUDA" if cuda else "CPU")


# ============================================================
# STEP 02: Check the model's class label mapping
# ============================================================

print("\nModel label mapping:")
print(model.config.id2label)


# ============================================================
# STEP 03: Prepare test premise-hypothesis pairs
# ============================================================

test_cases = [
    {
        "name": "Entailment Test",
        "premise": "The capital of France is Paris.",
        "hypothesis": "Paris is the capital of France."
    },
    {
        "name": "Neutral Test",
        "premise": "The capital of France is Paris.",
        "hypothesis": "France has a population of 60 million."
    },
    {
        "name": "Contradiction Test",
        "premise": "The capital of France is Paris.",
        "hypothesis": "The capital of France is Berlin."
    }
]


# ============================================================
# STEP 04: Run inference and validate the returned results
# ============================================================

print("\nRunning inference tests...\n")

for test in test_cases:

    print("=" * 60)
    print(test["name"])
    print("=" * 60)

    start_time = time.perf_counter()

    predicted_label, entailment_prob, neutral_prob, contradiction_prob = (
        entailment_model.check_entailment(
            test["premise"],
            test["hypothesis"]
        )
    )

    end_time = time.perf_counter()

    inference_time = end_time - start_time

    # Display the results
    print("Premise:", test["premise"])
    print("Hypothesis:", test["hypothesis"])

    print("\nPredicted label:", predicted_label)
    print(f"Entailment probability: {entailment_prob:.4f}")
    print(f"Neutral probability: {neutral_prob:.4f}")
    print(f"Contradiction probability: {contradiction_prob:.4f}")

    print(f"\nProbability sum: {(entailment_prob + neutral_prob + contradiction_prob):.6f}")

    print(f"Inference time: {inference_time:.4f} seconds")

    # Validate probability values
    probabilities = [
        entailment_prob,
        neutral_prob,
        contradiction_prob
    ]

    assert all(0.0 <= p <= 1.0 for p in probabilities), (
        "Probability outside the valid range!"
    )

    assert abs(sum(probabilities) - 1.0) < 1e-5, (
        "Probabilities do not sum to 1!"
    )

    # Validate predicted label
    expected_prediction = [
        "entailment",
        "neutral",
        "contradiction"
    ][probabilities.index(max(probabilities))]

    assert predicted_label == expected_prediction, (
        "Predicted label does not match the highest probability!"
    )

    print("\nResult validation: PASSED")


# ============================================================
# STEP 05: Test repeated inference
# ============================================================

print("\n" + "=" * 60)
print("Testing repeated inference")
print("=" * 60)

start_time = time.perf_counter()

for i in range(5):
    result = entailment_model.check_entailment(
        "The Earth revolves around the Sun.",
        "The Sun is at the centre of the Solar System."
    )

    print(f"Run {i + 1}: {result[0]}")

total_time = time.perf_counter() - start_time

print(f"\nTotal time for 5 inference calls: {total_time:.4f} seconds")
print("Repeated inference test: PASSED")


# ============================================================
# STEP 06: Finish
# ============================================================

del entailment_model, model, tokenizer
gc.collect()

print("\n" + "=" * 60)
print("ALL INFERENCE TESTS COMPLETED SUCCESSFULLY")
print("=" * 60)
