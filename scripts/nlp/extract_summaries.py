
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import os
import re
import string

def extract_summaries():
    directory = '/Users/wulfebw/Desktop/cases'
    pattern = os.path.join(directory, "*")
    fps = glob.glob(pattern)
    fps = [f for f in fps if f.endswith('.Case')]
    summaries = []
    for fp in fps:
        infile = np.load(fp)
        xml_key = [k for k in infile.keys() if k.endswith('.xml')][0]
        xml_data = infile[xml_key]
        try:
            decoded = xml_data.decode()
        except:
            decoded = None
        if decoded is None:
            continue
        start = decoded.find('<SUMMARY>') + len('<SUMMARY>')
        end = decoded.find('</SUMMARY>')
        summary = decoded[start:end].strip()
        summaries.append(summary)
    return list(set(summaries))

def tokenize_summaries(summaries):
    token_summaries = []
    for summary in summaries:
        tokens = nltk.word_tokenize(summary)
        # tokens = [t.lower() for t in tokens if t not in string.punctuation]
        token_summaries.append(tokens)
    return token_summaries

def stem_summaries(summaries):
    st = LancasterStemmer()
    stemmed_summaries = []
    for summary in summaries:
        stemmed_tokens = [st.stem(t) for t in summary]
        stemmed_summaries.append(stemmed_tokens)
    return stemmed_summaries

def analyze(summaries):
    stopwords_set = set(stopwords.words('english'))
    all_tokens = [t for s in summaries for t in s if t not in stopwords_set]
    dist = nltk.FreqDist(all_tokens)
    mc = dist.most_common(40)
    counts = [c for (_,c) in mc]
    labels = [l for (l,_) in mc]
    print(labels)
    ind = np.arange(len(counts))
    width = .25
    fig, ax = plt.subplots()
    ax.bar(ind, counts, width, color='g')
    ax.set_xticks(ind)
    ax.set_xticklabels(labels, rotation=60)
    plt.tight_layout()
    plt.show()

def list_reasons(summaries):
    for summary in summaries:
        sum_str = ' '.join(summary)
        start_string = 'critical reason'

if __name__ == '__main__':
    print('extracting summaries...')
    summaries = extract_summaries()
    print('tokenizing...')
    summaries = tokenize_summaries(summaries)
    # print('stemming...')
    # summaries = stem_summaries(summaries)
    # print('analyzing...')
    # analyze(summaries)
    print('listing reasons...')
    list_reasons(summaries)
