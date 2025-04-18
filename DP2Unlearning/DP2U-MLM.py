#!/usr/bin/env python
"""
DP-MLM on the TOFU Dataset using the Exponential Mechanism

This script loads the TOFU dataset from Hugging Face, applies a differentially private
replacement on noun phrases (using an exponential mechanism that samples a candidate
based on a semantic similarity utility), and saves the privatized dataset to a CSV file.
"""

import os
import json
import string
import time
import logging
import random
from collections import Counter
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
import wn
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, logging as transformers_logging
import nltk
from nltk import pos_tag
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('words', quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

en = wn.Wordnet('oewn:2022')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
transformers_logging.set_verbosity_warning()

stop = set(stopwords.words("english"))
noun_pos_tags = {'NN', 'NNS', 'NNP', 'NNPS'}

EPSILON = 1

def nth_repl(s, sub, repl, n):
    """
    Replace the n-th occurrence of substring sub with repl in string s.
    """
    s_split = s.split()
    count = 0
    for i, token in enumerate(s_split):
        if token == sub:
            count += 1
            if count == n:
                s_split[i] = repl
                return " ".join(s_split)
    return s  

def sentence_enum(tokens):
    """
    Enumerates each token occurrence in the sentence (used for tracking replacements).
    """
    counts = Counter()
    enumeration = []
    for token in tokens:
        counts[token] += 1
        enumeration.append(counts[token])
    return enumeration

class DPMLM:
    def __init__(self, MODEL="roberta-base", alpha=0.003, epsilon=EPSILON):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.lm_model = AutoModelForMaskedLM.from_pretrained(MODEL)
        self.raw_model = AutoModel.from_pretrained(MODEL, output_hidden_states=True, output_attentions=True)
        self.alpha = alpha
        self.epsilon = epsilon
        self.clip_min = -5.2093127
        self.clip_max = 20.304797887802124
        self.sensitivity = abs(self.clip_max - self.clip_min)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.lm_model = self.lm_model.to(self.device)
        self.raw_model = self.raw_model.to(self.device)
        self.detokenizer = TreebankWordDetokenizer()
        self.lemmatizer = WordNetLemmatizer()

    def _get_contextual_embedding(self, model_output, token_index):
        """
        Compute a representation for the token at token_index by concatenating
        the last four hidden layers.
        """
        last_layers = [model_output.hidden_states[i][:, token_index, :] for i in [-4, -3, -2, -1]]
        return torch.cat(last_layers, dim=1)

    def privatize(self, sentence, target, n=1, K=10):
        """
        Applies the exponential mechanism to replace the target word in the sentence.
        
        Steps:
          1. Mask the n-th occurrence of target using nth_repl.
          2. Run the raw model to obtain the contextual embedding of the masked token.
          3. Obtain the top K candidate tokens from the masked LM.
          4. For each candidate, form a candidate sentence and compute its utility 
             (cosine similarity between the candidate’s embedding and the original’s).
          5. Compute selection probabilities proportional to exp(epsilon * utility).
          6. Sample a candidate and return the sentence with target replaced.
        """
        masked_sentence = nth_repl(sentence, target, self.tokenizer.mask_token, n)
        input_ids = self.tokenizer.encode(masked_sentence, add_special_tokens=True)
        if self.tokenizer.mask_token_id not in input_ids:
            return sentence  
        
        mask_index = input_ids.index(self.tokenizer.mask_token_id)
        
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)
        with torch.no_grad():
            original_output = self.raw_model(input_tensor)
        
        with torch.no_grad():
            lm_output = self.lm_model(input_tensor)
        logits = lm_output[0].squeeze().detach().cpu().numpy()
        mask_logits = logits[mask_index]
        
        topk_indices = torch.topk(torch.tensor(mask_logits), k=K, dim=0)[1].tolist()
        
        candidate_tokens = []
        candidate_utilities = []
        for cand_idx in topk_indices:
            candidate_token = self.tokenizer.decode(cand_idx).strip()
            if not candidate_token.isalpha():
                continue
            candidate_tokens.append(candidate_token)
            
            candidate_sentence = nth_repl(sentence, target, candidate_token, n)
            cand_input_ids = self.tokenizer.encode(candidate_sentence, add_special_tokens=True)
            try:
                cand_token_id = self.tokenizer.convert_tokens_to_ids(candidate_token)
                cand_index = cand_input_ids.index(cand_token_id)
            except ValueError:
                cand_index = mask_index
            cand_input_tensor = torch.tensor(cand_input_ids).unsqueeze(0).to(self.device)
            with torch.no_grad():
                candidate_output = self.raw_model(cand_input_tensor)
            orig_emb = self._get_contextual_embedding(original_output, mask_index)
            cand_emb = self._get_contextual_embedding(candidate_output, cand_index)
            cos_sim = torch.nn.functional.cosine_similarity(orig_emb, cand_emb, dim=1).item()
            candidate_utilities.append(cos_sim)
        
        if len(candidate_tokens) == 0:
            return sentence  
        
        utilities_tensor = torch.tensor(candidate_utilities, dtype=torch.float)
        exp_utilities = torch.exp(self.epsilon * utilities_tensor)
        probabilities = exp_utilities / exp_utilities.sum()
        
        chosen_index = torch.multinomial(probabilities, 1).item()
        chosen_token = candidate_tokens[chosen_index]
        
        replaced_sentence = nth_repl(sentence, target, chosen_token, n)
        return replaced_sentence

    def dpmlm_rewrite_noun_phrase(self, sentence, epsilon, REPLACE=True, FILTER=False, STOP=False, TEMP=True, POS=True, CONCAT=True, K=10):
        """
        Applies privatization on noun phrases within a sentence.
        This implementation processes each noun (as identified by POS tagging)
        and replaces it using the exponential mechanism via self.privatize.
        
        Returns the privatized sentence along with counts of perturbed and total tokens.
        """
        tokens = nltk.word_tokenize(sentence)
        pos_tags = pos_tag(tokens)
        new_tokens = tokens.copy()
        perturbed = 0
        total = 0
        
        for i, (t, pos) in enumerate(pos_tags):
            total += 1
            if t in string.punctuation or (STOP == False and t in stop):
                continue
            if pos not in noun_pos_tags:
                continue
            new_sentence = self.privatize(sentence, t, n=1, K=K)
            new_tokens_i = nltk.word_tokenize(new_sentence)
            if i < len(new_tokens_i):
                replacement = new_tokens_i[i]
                if replacement != t:
                    perturbed += 1
                new_tokens[i] = replacement
        return self.detokenizer.detokenize(new_tokens), perturbed, total

def main():
    dataset = load_dataset("locuslab/TOFU", "full")
    
    dpmlm = DPMLM(epsilon=EPSILON)
    
    privatized_data = []
    for example in dataset["train"]:
        question = example.get("question", "")
        answer = example.get("answer", "")
        rewritten_question, perturbed_q, total_q = dpmlm.dpmlm_rewrite_noun_phrase(
            question, epsilon=EPSILON, REPLACE=True, FILTER=False, STOP=False, TEMP=True, POS=True, CONCAT=True, K=10
        )
        rewritten_answer, perturbed_a, total_a = dpmlm.dpmlm_rewrite_noun_phrase(
            answer, epsilon=EPSILON, REPLACE=True, FILTER=False, STOP=False, TEMP=True, POS=True, CONCAT=True, K=10
        )
        privatized_data.append({
            "question": rewritten_question,
            "answer": rewritten_answer
        })
    
    output_dir = "dp_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"private_data_epsilon_{EPSILON}.csv")
    df = pd.DataFrame(privatized_data)
    df.to_csv(output_file, index=False)
    print("Privatized TOFU dataset saved to", output_file)

if __name__ == '__main__':
    main()
