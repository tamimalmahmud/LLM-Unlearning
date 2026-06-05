import json
import os
import re
import string
import random
import torch
import nltk
import spacy
from datasets import load_dataset
from collections import Counter
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, logging as transformers_logging
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import WordNetLemmatizer
from difflib import SequenceMatcher
import time

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

nlp = spacy.load("en_core_web_sm")

stop = set(stopwords.words("english"))
noun_pos_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
EPSILON = 1.0
PRIVACY_STRENGTH = 0.9  

protected_family_words = {
    "father", "mother", "dad", "mom", "mama", "papa",
    "brother", "sister", "son", "daughter", "uncle",
    "aunt", "grandfather", "grandmother", "grandpa",
    "grandma", "husband", "wife", "cousin", "nephew",
    "niece", "parent", "parents", "child", "children"
}

def nth_repl(s, sub, repl, n):
    """
    Replace nth occurrence of sub (phrase or token) in s, handling punctuation/boundaries.
    This preserves surrounding punctuation and capitalization where reasonable.
    """
    escaped = re.escape(sub)
    pattern = r'\b' + escaped + r'\b'
    matches = list(re.finditer(pattern, s, flags=re.IGNORECASE))
    if len(matches) < n:
        return s
    start, end = matches[n - 1].span()

    orig_fragment = s[start:end]
    repl_final = repl
    if orig_fragment and orig_fragment[0].isupper() and repl and repl[0].islower():
        repl_final = repl.capitalize()

    return s[:start] + repl_final + s[end:]


def is_too_similar(a, b):
    """Detect near-identical words (Levenshtein-like)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > 0.9


def cleanup_sentence(text):
    """Remove duplicates, extra spaces, and bad punctuation."""
    text = re.sub(r"\s([?.!,;:])", r"\1", text)
    text = re.sub(r"(\b[A-Z][a-z]+\b)( \1)+", r"\1", text)
    text = re.sub(r"\b([A-Z][a-z]+ ){2,}([A-Z][a-z]+, [A-Z][a-z]+)", r"\2", text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def extract_spacy_entities(text):
    """Extract PERSON, GPE, ORG, LOCATION using Spacy."""
    doc = nlp(text)
    return set([ent.text.lower() for ent in doc.ents if ent.label_ in {"PERSON", "GPE", "ORG", "LOCATION", "NORP"}])


def extract_sensitive_entities(text):
    """Detect PERSON, GPE, ORG, LOCATION, nationality adjectives, and additional identity-revealing terms."""
    entities = extract_spacy_entities(text)

    sensitive = set(entities)

    additional_sensitives = {
        "doctor", "engineer", "teacher", "scientist", "lawyer", "manager", "professor", "developer",
        "father", "mother", "son", "daughter", "husband", "wife", "sibling", "grandfather", "grandmother",
        "nephew", "niece", "cousin", "uncle", "aunt", "parent", "children", "family", "relative",
        "CEO", "president", "director", "worker", "artist", "musician", "author", "entrepreneur", "dietitian",
        "judge", "radiologist", "accountant"
    }

    sensitive.update(additional_sensitives)

    return sensitive

class DPMLM:
    def __init__(self, MODEL="roberta-base", epsilon=EPSILON):
        transformers_logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.lm_model = AutoModelForMaskedLM.from_pretrained(MODEL)
        self.raw_model = AutoModel.from_pretrained(MODEL, output_hidden_states=True)
        self.epsilon = epsilon
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.lm_model = self.lm_model.to(self.device)
        self.raw_model = self.raw_model.to(self.device)
        self.detokenizer = TreebankWordDetokenizer()
        self.lemmatizer = WordNetLemmatizer()
        self.replacement_cache = {}

    def _get_contextual_embedding(self, model_output, token_index):
        """Concatenate last 4 layers for contextual embedding."""
        layers = [model_output.hidden_states[i][:, token_index, :] for i in [-4, -3, -2, -1]]
        return torch.cat(layers, dim=1)

    def privatize(self, sentence, target, n=1, K=8):
        """
        Replace target token/phrase via DP sampling with semantic + similarity filtering.
        Works with multi-word target (phrase) — uses nth_repl internally.
        """
        masked_sentence = nth_repl(sentence, target, self.tokenizer.mask_token, n)
        input_ids = self.tokenizer.encode(masked_sentence, add_special_tokens=True)
        if self.tokenizer.mask_token_id not in input_ids:
            return sentence

        mask_index = input_ids.index(self.tokenizer.mask_token_id)
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)

        with torch.no_grad():
            orig_output = self.raw_model(input_tensor)
            lm_output = self.lm_model(input_tensor)

        logits = lm_output[0].squeeze().detach().cpu()
        mask_logits = logits[mask_index]
        probs = torch.softmax(mask_logits / 1.5, dim=0)
        top_k = min(K, probs.shape[0])
        top_indices = torch.topk(probs, k=top_k).indices.tolist()

        candidates, utilities = [], []
        for cand_idx in top_indices:
            cand = self.tokenizer.decode(cand_idx).strip()
            if not cand.isalpha() or is_too_similar(cand, target):
                continue
            candidates.append(cand)

            cand_sent = nth_repl(sentence, target, cand, n)
            cand_ids = self.tokenizer.encode(cand_sent, add_special_tokens=True)

            try:
                cand_token_id = self.tokenizer.convert_tokens_to_ids(cand)
                cand_pos = cand_ids.index(cand_token_id)
            except ValueError:
                cand_pos = mask_index

            cand_tensor = torch.tensor(cand_ids).unsqueeze(0).to(self.device)
            with torch.no_grad():
                cand_output = self.raw_model(cand_tensor)

            sim = torch.nn.functional.cosine_similarity(
                self._get_contextual_embedding(orig_output, mask_index),
                self._get_contextual_embedding(cand_output, cand_pos),
                dim=1
            ).item()

            if sim < 0.35:
                continue
            utilities.append(sim)

        if not candidates:
            return sentence

        utilities = torch.tensor(utilities)
        probs = torch.exp(self.epsilon * utilities)
        probs = probs / probs.sum()
        chosen = candidates[torch.multinomial(probs, 1).item()]

        return nth_repl(sentence, target, chosen, n)

    def cached_privacy_replace(self, sentence, token_or_phrase, n=1, K=10):
        """
        Use cache to keep consistent replacements across dataset.
        Works for multi-word phrases as well.
        """
        key = token_or_phrase.lower()
        if key in self.replacement_cache:
            return nth_repl(sentence, token_or_phrase, self.replacement_cache[key], n)

        new_sentence = self.privatize(sentence, token_or_phrase, n=n, K=K)
        diff = [t for t in new_sentence.split() if t not in sentence.split()]
        if diff:
            self.replacement_cache[key] = diff[0]
        return new_sentence

    def perturb_answer(self, answer_text, question_subjects, epsilon=EPSILON, K=10):
        """
        Perturb answer_text by:
        1) extracting noun phrases (SpaCy) and perturbing them as a unit, skipping protected words
        2) perturb remaining nouns token-by-token (NLTK POS), skipping punctuation & protected words
        """
        doc = nlp(answer_text)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]

        for np_text in noun_phrases:
            np_clean = np_text.strip().lower().strip(string.punctuation)
            if not np_clean:
                continue

            if any(word in protected_family_words for word in re.findall(r"\w+", np_clean)):
                continue

            if any(qs in np_clean for qs in question_subjects):
                continue

            answer_text = self.cached_privacy_replace(answer_text, np_text, n=1, K=K)

        tokens = nltk.word_tokenize(answer_text)
        pos_tags = nltk.pos_tag(tokens)
        new_sentence = answer_text
        token_counts = Counter()

        sensitive_set = extract_sensitive_entities(answer_text)

        for token, pos in pos_tags:
            t_low = token.lower()
            if t_low in stop or token in string.punctuation:
                continue

            if random.random() > PRIVACY_STRENGTH:
                continue

            if pos not in noun_pos_tags:
                continue

            if t_low in protected_family_words or t_low in question_subjects:
                continue

            if (t_low not in sensitive_set) and random.random() > 0.8:
                continue

            token_counts[token] += 1
            new_sentence = self.cached_privacy_replace(new_sentence, token, n=token_counts[token], K=K)

        return cleanup_sentence(new_sentence)

def main():
    start = time.time()
    dataset_config = "shard_R50"

    dataset = load_dataset("talmahmud/tofu_custom_split_ESU", dataset_config)
    dpmlm = DPMLM(epsilon=EPSILON)

    out_data = []
    for ex in dataset["train"]:
        q = ex.get("question", "")
        a = ex.get("answer", "")
        q_subjects = extract_spacy_entities(q)
        new_answer = dpmlm.perturb_answer(a, question_subjects=q_subjects, epsilon=EPSILON)
        out_data.append({"question": q, "answer": new_answer})

    os.makedirs("dp_data_old", exist_ok=True)
    out_path = os.path.join("dp_data_old", f"{dataset_config}DP{EPSILON}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f" Perturbed dataset saved to {out_path}")
    print(f" Runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
