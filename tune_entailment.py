"""Tune sentence count and entailment probability threshold on the dev set."""

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from factcheck import EntailmentFactChecker, EntailmentModel, tokenize
from factchecking_main import read_fact_examples, read_passages


MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"
# The earlier coarse sweep found the useful region around 0.35-0.40.
# We include a small margin below and above that region, while avoiding
# thresholds near 0 or 1 that would classify almost every example the same way.
MIN_THRESHOLD = 0.250
MAX_THRESHOLD = 0.500
THRESHOLD_STEP = 0.001
MAX_SENTENCE_COUNT = 6


def collect_scores(examples, entailment_model):
    checker = EntailmentFactChecker(entailment_model)
    scores = []
    for example in examples:
        probabilities_by_count = [0.0] * MAX_SENTENCE_COUNT
        fact_tokens = set(tokenize(example.fact))
        for passage in example.passages:
            sentences = checker.noise_removal(
                checker.sentence_splitting(passage["text"])
            )
            ranked = sorted(
                sentences,
                key=lambda sentence: len(fact_tokens & set(tokenize(sentence))),
                reverse=True,
            )
            passage_probabilities = []
            for sentence in ranked[:MAX_SENTENCE_COUNT]:
                result = entailment_model.check_entailment(sentence, example.fact)
                passage_probabilities.append(result[1])
            for sentence_count in range(1, len(passage_probabilities) + 1):
                probabilities_by_count[sentence_count - 1] = max(
                    probabilities_by_count[sentence_count - 1],
                    max(passage_probabilities[:sentence_count]),
                )
        gold = "NS" if example.label == "IR" else example.label
        scores.append((probabilities_by_count, gold))
    return scores


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    entailment_model = EntailmentModel(model, tokenizer, cuda=False)
    passages = read_passages(PASSAGES_PATH)
    examples = read_fact_examples(LABELS_PATH, passages)

    scores = collect_scores(examples, entailment_model)
    best = (0, None, None)
    for sentence_count in range(1, MAX_SENTENCE_COUNT + 1):
        threshold = MIN_THRESHOLD
        while threshold <= MAX_THRESHOLD:
            correct = sum(
                ("S" if probabilities[sentence_count - 1] >= threshold else "NS")
                == gold
                for probabilities, gold in scores
            )
            if correct > best[0]:
                best = (correct, sentence_count, threshold)
            threshold += THRESHOLD_STEP

    correct, sentence_count, threshold = best
    print(
        f"Best: {correct}/{len(examples)} "
        f"({correct / len(examples):.6f}) "
        f"sentences={sentence_count} threshold={threshold:.3f}"
    )


if __name__ == "__main__":
    main()
