# factcheck.py

from sympy import re
import torch
from typing import List
import numpy as np
import spacy
import gc
import re

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


class WordRecallThresholdFactChecker(FactChecker):
    """
    Bag-of-words recall: the weighted fraction of the fact's (stop-word-filtered, stemmed) unique tokens that
    appear anywhere in the retrieved passages. Predicts "S" if the recall is >= threshold.
 
    Design choices (selected by grid search on the dev set, see report):
      * stop words removed + light stemming  -> removes function-word noise, matches inflections
      * recall over the UNION of all passages (a fact may be supported by information spread over passages)
      * tokens that are capitalised (not sentence-initial) or numeric in the fact -- names, places, dates --
        get weight 2, since they carry the facts' content; other tokens get weight 1
      * threshold 0.775: centre of the plateau (0.75-0.80) where dev accuracy is within 1% of the best
    """
    CONTENT_WEIGHT = 2.0
    THRESHOLD = 0.775
 
    def __init__(self, threshold: float = THRESHOLD, content_weight: float = CONTENT_WEIGHT):
        self.threshold = threshold
        self.content_weight = content_weight
 
    def score(self, fact: str, passages: List[dict]) -> float:
        fact_tokens = set(tokenize(fact))
        if not fact_tokens:
            return 0.0
        # Content tokens: capitalised or numeric words, ignoring the first word of the sentence
        raw_words = re.findall(r"[A-Za-z0-9]+", fact)[1:]
        content_tokens = set(tokenize(" ".join(w for w in raw_words if w[0].isupper() or w[0].isdigit())))
 
        passage_tokens = set()
        for p in passages:
            passage_tokens.update(tokenize(p["text"]))
 
        total = matched = 0.0
        for tok in fact_tokens:
            w = self.content_weight if tok in content_tokens else 1.0
            total += w
            if tok in passage_tokens:
                matched += w
        return matched / total
 
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S" if self.score(fact, passages) >= self.threshold else "NS"


class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model):
        self.ent_model = ent_model

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

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
