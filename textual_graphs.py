import numpy as np
from itertools import combinations
import spacy
from spacy import displacy
import en_core_web_sm
from spacy.matcher import Matcher
from spacy.tokens import Span



class Build_Fragments():
    nlp = spacy.load("en_core_web_sm")

    def __init__(self, text):
        self.text = text
        self.doc = self.nlp(text)
        self.word_dic = {i: j for i, j in enumerate(self.doc)}
        self.root = list(self.doc.sents)[0].root
        self.cluster = self.obtain_clusters()
        self.combinations = self.obtain_combinations()

    def build_root_sentence(self):
        root_text = []
        for i in self.doc:
            if i.head == self.root:
                root_text.append(i)
        return root_text

    def get_root_text(self):
        root_text = {}
        for ind, val in enumerate(self.doc):
            if val.head.text == self.root.text:
                root_text.update({ind: val})
        return root_text, len(root_text) / len(self.doc)

    def obtain_clusters(self):
        start = 0
        to_be_deleted = []
        for i in [i for i in self.get_root_text()[0]]:
            to_be_deleted.append(list(range(start, i)))
            start = i + 1
        return list(filter(None, to_be_deleted))

    def obtain_combinations(self):
        tree = {}
        for i in range(1, len(self.cluster) + 1):
            tree.update({i: list(combinations(self.cluster, i))})
        return tree

    def get_entities(self, sent):
        ## chunk 1
        ent1 = ""
        ent2 = ""

        prv_tok_dep = ""  # dependency tag of previous token in the sentence
        prv_tok_text = ""  # previous token in the sentence

        prefix = ""
        modifier = ""

        #############################################################

        for tok in self.nlp(sent):
            ## chunk 2
            # if token is a punctuation mark then move on to the next token
            if tok.dep_ != "punct":
                # check: token is a compound word or not
                if tok.dep_ == "compound":
                    prefix = tok.text
                    # if the previous word was also a 'compound' then add the current word to it
                    if prv_tok_dep == "compound":
                        prefix = prv_tok_text + " " + tok.text

                # check: token is a modifier or not
                if tok.dep_.endswith("mod") == True:
                    modifier = tok.text
                    # if the previous word was also a 'compound' then add the current word to it
                    if prv_tok_dep == "compound":
                        modifier = prv_tok_text + " " + tok.text

                ## chunk 3
                if tok.dep_.find("subj") == True:
                    ent1 = modifier + " " + prefix + " " + tok.text
                    prefix = ""
                    modifier = ""
                    prv_tok_dep = ""
                    prv_tok_text = ""

                    ## chunk 4
                if tok.dep_.find("obj") == True:
                    ent2 = modifier + " " + prefix + " " + tok.text

                ## chunk 5
                # update variables
                prv_tok_dep = tok.dep_
                prv_tok_text = tok.text
        #############################################################

        return [ent1.strip(), ent2.strip()]

    def show_sentences(self):
        sentence = []
        str_sent = []
        for i, j in self.combinations.items():
            for sentences in j:
                sentence.append(np.delete(self.doc, np.concatenate(sentences)))
        for val in sentence:
            str_sent.append([" ".join(list(map(str, val)))])
        return str_sent
