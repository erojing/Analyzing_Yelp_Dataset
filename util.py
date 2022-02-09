import pandas as pd
import numpy as np
import ast
import spacy
spacy.load('en_core_web_sm')
from spacy.lang.en import English
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
import pickle
import gensim

def count_total_rows(json_path, json_file, chunk_num):
    with open(json_path + json_file, "r") as f:
        reader = pd.read_json(json_path + json_file, lines=True, chunksize=chunk_num)
        size = 0
        for chunk in reader:
            size += chunk.shape[0]
    f.close()
    return size

def to_lower_case(string):
    try:
        return string.lower()
    except:
        pass

# text tokenize
parser = English()
def tokenize(text):
    lda_tokens=[]
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

# Get pos
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return None

# Lemmatization
def get_lemma(text):
    lemma_sent = []
    tagged_text = pos_tag(text)
    for word in tagged_text:
        wordnet_pos = get_wordnet_pos(word[1]) or wn.NOUN
        lemma_sent.append(WordNetLemmatizer().lemmatize(word[0], pos = wordnet_pos))
    return lemma_sent

# Using tokenization, lemmatiztion, and filter by text tokens more than 4
en_stop = set(nltk.corpus.stopwords.words('english'))
def prepare_text_for_lda(text):
    tokens = tokenize(text)
    # Only for text length >4
    tokens = [token for token in tokens if len(token)>4]
    # Lemmatize words
    tokens = get_lemma(tokens)
    tokens = [token for token in tokens if token not in en_stop]
    return tokens


