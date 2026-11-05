# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc
import re
import os
import json
import math
from collections import Counter

STOP_WORDS = frozenset(
    "a an the and or of in on at to for with by from as is are was were be been being has have had do does did "
    "it its this that these those he she they his her their him them who which what when where i you we not no "
    "s also".split()
)  



def simple_stem(word: str) -> str:
    """Very light suffix stripping (no external dependency); only strips if >= 3 chars remain."""
    for suffix in ("ing", "edly", "ed", "es", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word
 
 
def tokenize(text: str) -> List[str]:
    """Lowercase, split into alphanumeric tokens, drop stop words, then stem."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [simple_stem(w) for w in words if w not in STOP_WORDS]




class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        # STEP 01: logits -> probabilities over the 3 classes
        probs = torch.softmax(logits, dim=-1)

        # STEP 02: index of the highest probability
        pred_idx = torch.argmax(probs, dim=-1).item()
        labels = ["entailment", "neutral", "contradiction"]
        predicted_label = labels[pred_idx]

        # STEP 03: convert tensors to plain Python values
        probs_list = probs[0].tolist()
        entailment_prob = probs_list[0]
        neutral_prob = probs_list[1]
        contradiction_prob = probs_list[2]

        del inputs, outputs, logits, probs
        gc.collect()

        # STEP 04: return label + all three probabilities (no threshold applied here)
        return predicted_label, entailment_prob, neutral_prob, contradiction_prob


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


DEFAULT_PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl" 

def _bleu(cand: List[str], ref: List[str], max_n: int = 4) -> float:
    """Sentence BLEU: add-one smoothed n-gram precisions (n > 1) and a brevity penalty."""
    if not cand:
        return 0.0
    log_p, used = 0.0, 0
    for n in range(1, max_n + 1):
        cand_ng = Counter(tuple(cand[i:i + n]) for i in range(len(cand) - n + 1))
        if not cand_ng:
            break
        ref_ng = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        match = sum(min(c, ref_ng[g]) for g, c in cand_ng.items())
        total = sum(cand_ng.values())
        log_p += math.log((match + 1) / (total + 1)) if n > 1 else math.log(max(match, 1e-9) / total)
        used += 1
    bp = 1.0 if len(cand) >= len(ref) else math.exp(1 - len(ref) / len(cand))
    return bp * math.exp(log_p / used)
 
 
class WordRecallThresholdFactChecker(FactChecker):
    """
    Bag-of-words overlap fact checker. score(fact, passages) is a real number in [0, 1] computed by one of several
    lexical metrics (see METRICS); predict() says "S" when score >= threshold, else "NS".
 
    All metrics use the same preprocessing (tokenize(): lowercase, alphanumeric tokens, stop-word removal, light
    stemming) and compare the fact against ALL retrieved passages merged into one bag of words.
 
    FINAL METRIC (highest dev accuracy, see report): "idf_recall"
        = IDF-weighted recall: the share of the fact's unique tokens, each weighted by its inverse document
        frequency, that also occur in the passages. Matching a rare word (a surname, a place) counts for more than
        matching a common one ("born", "known"). Dev accuracy is 181/221 = 81.9% at the default threshold 0.725
        (peak 182/221 = 82.4% at 0.7249). It ties for the top with "idf_content_recall" (the same score with an
        extra boost for capitalised/numeric words) and is simpler, so it is the default. Its accuracy stays
        >= 80% for thresholds 0.68 - 0.80, so the threshold is not a knife-edge.
 
    The IDF table is built once from the retrieved passages file (DEFAULT_PASSAGES_PATH, the same file
    factchecking_main.py loads). If that file can't be found the checker falls back to "content_recall", which
    needs no external statistics.
    """
    METRICS = ["recall", "content_recall", "idf_recall", "idf_content_recall", "jaccard", "precision", "f1",
               "cosine_binary", "cosine_tfidf", "rouge1_f1", "bleu"]
 
    CONTENT_WEIGHT = 2.0
    FINAL_METRIC = "idf_recall"  #change this to "idf_content_recall" if you want the other top metric instead
   
    THRESHOLDS = {
        "recall": 0.714, "content_recall": 0.775, "idf_recall": 0.725, "idf_content_recall": 0.800,
        "jaccard": 0.005, "precision": 0.005, "f1": 0.010, "cosine_binary": 0.087, "cosine_tfidf": 0.040,
        "rouge1_f1": 0.018, "bleu": 0.0, 
    }
 
    def __init__(self, metric: str = None, threshold: float = None, content_weight: float = CONTENT_WEIGHT,
                 passages_path: str = DEFAULT_PASSAGES_PATH):
        self.metric = metric or self.FINAL_METRIC
        assert self.metric in self.METRICS, "unknown metric: " + repr(self.metric)
        self.content_weight = content_weight
        self.passages_path = passages_path
        self.idf_table = None
        self.idf_default = 1.0
        self._idf_loaded = False
        self._threshold_override = threshold
        self.fallback = False
 
    # ---------------------------------------------------------------- IDF
    def _load_idf(self):
        """Build document frequencies over every retrieved passage in the passages file (once)."""
        if self._idf_loaded:
            return
        self._idf_loaded = True
        needs_idf = self.metric in ("idf_recall", "idf_content_recall", "cosine_tfidf")
        if not needs_idf:
            return
        here = os.path.dirname(os.path.abspath(__file__))
        for path in (self.passages_path, os.path.join(here, self.passages_path)):
            if os.path.exists(path):
                df, n = Counter(), 0
                with open(path, "r") as f:
                    for line in f:
                        for p in json.loads(line)["passages"]:
                            n += 1
                            df.update(set(tokenize(p["text"])))
                self.idf_table = (df, n)
                return
        # passages file not found -> fall back to a metric that needs no corpus statistics
        if self.metric != "cosine_tfidf":
            self.metric = "content_recall"
            self.fallback = True
 
    def idf(self, token: str) -> float:
        if self.idf_table is None:
            return 1.0
        df, n = self.idf_table
        return math.log((n + 1) / (df.get(token, 0) + 1)) + 1
 
    @property
    def threshold(self) -> float:
        if self._threshold_override is not None:
            return self._threshold_override
        return self.THRESHOLDS[self.metric]
 
    # ---------------------------------------------------------------- scoring
    def _content_tokens(self, fact: str) -> set:
        """Capitalised (not sentence-initial) or numeric words of the fact, tokenized like everything else."""
        raw = re.findall(r"[A-Za-z0-9]+", fact)[1:]
        return set(tokenize(" ".join(w for w in raw if w[0].isupper() or w[0].isdigit())))
 
    def _weighted_recall(self, fact: str, fact_set: set, pass_set: set, use_idf: bool, use_content: bool) -> float:
        content = self._content_tokens(fact) if use_content else set()
        total = matched = 0.0
        for tok in fact_set:
            w = self.idf(tok) if use_idf else 1.0
            if tok in content:
                w *= self.content_weight
            total += w
            if tok in pass_set:
                matched += w
        return matched / total if total else 0.0
 
    def score(self, fact: str, passages: List[dict]) -> float:
        self._load_idf()
        fact_toks = tokenize(fact)
        fact_set = set(fact_toks)
        if not fact_set:
            return 0.0
        pass_toks = [t for p in passages for t in tokenize(p["text"])]
        pass_set = set(pass_toks)
        inter = fact_set & pass_set
        m = self.metric
 
        if m == "recall":
            return len(inter) / len(fact_set)
        if m == "content_recall":
            return self._weighted_recall(fact, fact_set, pass_set, use_idf=False, use_content=True)
        if m == "idf_recall":
            return self._weighted_recall(fact, fact_set, pass_set, use_idf=True, use_content=False)
        if m == "idf_content_recall":
            return self._weighted_recall(fact, fact_set, pass_set, use_idf=True, use_content=True)
        if not pass_set:
            return 0.0
        precision = len(inter) / len(pass_set)
        recall = len(inter) / len(fact_set)
        if m == "precision":
            return precision
        if m == "f1":
            return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if m == "jaccard":
            return len(inter) / len(fact_set | pass_set)
        if m == "cosine_binary":
            return len(inter) / math.sqrt(len(fact_set) * len(pass_set))
        if m == "cosine_tfidf":
            fc, pc = Counter(fact_toks), Counter(pass_toks)
            wf = {w: c * self.idf(w) for w, c in fc.items()}
            wp = {w: c * self.idf(w) for w, c in pc.items()}
            dot = sum(v * wp.get(w, 0.0) for w, v in wf.items())
            nf = math.sqrt(sum(v * v for v in wf.values()))
            npass = math.sqrt(sum(v * v for v in wp.values()))
            return dot / (nf * npass) if nf and npass else 0.0
        if m == "rouge1_f1":
            fc, pc = Counter(fact_toks), Counter(pass_toks)
            overlap = sum(min(c, pc[w]) for w, c in fc.items())
            r, p = overlap / len(fact_toks), overlap / len(pass_toks)
            return 2 * p * r / (p + r) if p + r else 0.0
        if m == "bleu":
            return _bleu(fact_toks, pass_toks)
        raise NotImplementedError(m)
 
    def predict(self, fact: str, passages: List[dict]) -> str:
        score = self.score(fact, passages)          # first call may switch to the fallback metric
        return "S" if score >= self.threshold else "NS"
 
 
def sweep_metrics(examples: List[FactExample], metrics: List[str] = None):
    """
    EXPERIMENT HELPER (not used by the graded commands). For each metric, tries every distinct score as the
    threshold and prints the best dev accuracy. Use it from a scratch script:
 
        from factchecking_main import *      # or copy read_passages / read_fact_examples
        sweep_metrics(examples)
    """
    metrics = metrics or WordRecallThresholdFactChecker.METRICS
    golds = [("NS" if ex.label == "IR" else ex.label) == "S" for ex in examples]
    results = []
    for metric in metrics:
        checker = WordRecallThresholdFactChecker(metric=metric)
        scores = [checker.score(ex.fact, ex.passages) for ex in examples]
        best_acc, best_thr = 0.0, 0.0
        for t in sorted(set(scores)):
            acc = sum((s >= t) == g for s, g in zip(scores, golds)) / len(golds)
            if acc > best_acc:
                best_acc, best_thr = acc, t
        results.append((metric, best_acc, best_thr))
        print(f"{metric:20s} best acc = {best_acc:.3f} ({round(best_acc * len(golds))}/{len(golds)}) at threshold {best_thr:.4f}")
    return results


class EntailmentFactChecker(FactChecker):
    MAX_SENTENCES_PER_PASSAGE = 5
    ENTAILMENT_THRESHOLD = 0.333

    def __init__(self, ent_model):
        self.ent_model = ent_model

    def sentence_splitting(self, passage_text: str) -> List[str]:
        """Split retrieved evidence into sentence-sized premises."""
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", passage_text)
            if sentence.strip()
        ]

    def noise_removal(self, sentences: List[str]) -> List[str]:
        """Remove empty fragments and duplicated evidence sentences."""
        seen = set()
        cleaned = []
        for sentence in sentences:
            normalized = " ".join(sentence.split())
            key = normalized.casefold()
            if normalized and key not in seen:
                seen.add(key)
                cleaned.append(normalized)
        return cleaned

    def word_overlap_pruning(self, fact: str, sentences: List[str]) -> List[str]:
        """Keep the most fact-relevant sentences before expensive model inference."""
        fact_tokens = set(tokenize(fact))
        if not fact_tokens:
            return sentences[:self.MAX_SENTENCES_PER_PASSAGE]

        ranked = []
        for index, sentence in enumerate(sentences):
            sentence_tokens = set(tokenize(sentence))
            overlap = len(fact_tokens & sentence_tokens)
            ranked.append((overlap, len(sentence_tokens), -index, sentence))

        ranked.sort(reverse=True)
        selected = [
            sentence
            for overlap, _, _, sentence in ranked[:self.MAX_SENTENCES_PER_PASSAGE]
            if overlap > 0
        ]
        return selected or sentences[:self.MAX_SENTENCES_PER_PASSAGE]

    def predict(self, fact: str, passages: List[dict]) -> str:
        best_entailment_probability = 0.0
        for passage in passages:
            sentences = self.sentence_splitting(passage["text"])
            sentences = self.noise_removal(sentences)
            sentences = self.word_overlap_pruning(fact, sentences)
            for sentence in sentences:
                result = self.ent_model.check_entailment(sentence, fact)
                if isinstance(result, tuple):
                    best_entailment_probability = max(best_entailment_probability, result[1])
                elif result == "entailment":
                    best_entailment_probability = 1.0
        return "S" if best_entailment_probability >= self.ENTAILMENT_THRESHOLD else "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    THRESHOLD = 0.5

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_dependencies = self.get_dependencies(fact)
        if not fact_dependencies:
            return "NS"

        best_recall = 0.0
        for passage in passages:
            passage_dependencies = self.get_dependencies(passage["text"])
            recall = len(fact_dependencies & passage_dependencies) / len(fact_dependencies)
            best_recall = max(best_recall, recall)

        return "S" if best_recall >= self.THRESHOLD else "NS"

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations
