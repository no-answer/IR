import datetime
import difflib
import json
import math
import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return ''

class BM25(object):
    def __init__(self, docs, docs_id):
        self.D = len(docs)
        self.avgdl = sum([len(doc)+0.0 for doc in docs]) / self.D  # avg length of all docs
        self.docs_id = docs_id
        self.docs = docs

        self.f = []
        self.df = {}  # 每个单词出现再多少个文中。
        self.idf = {}
        self.k1 = 2
        self.b = 0.75
        self.init()

    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                # word = word.lower()
                tmp[word] = tmp.get(word, 0) + 1
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D - v + 0.5) - math.log(v + 0.5)

    def sim(self, doc, index):
        score = 0
        d = len(self.docs[index])
        for word in doc:
            # word = word.lower()
            if word not in self.f[index]:
                continue
            tmp = (self.idf[word] * self.f[index][word] * (self.k1 + 1)
                   / (self.f[index][word] + self.k1 * (1 - self.b + self.b * d / self.avgdl)))
            # if self.docs_id[index] == '583461':
            #     print('583461', tmp, word)
            #
            # if self.docs_id[index] == '6943232':
            #     print('6943232', tmp, word)
            score += tmp
        return score

    def simall(self, doc):
        scores = []
        for index in range(self.D):
            score = [self.sim(doc, index), docs_id[index]]
            scores.append(score)
        return scores



def get_data():
    # f_d = json.load(open("documents.json", 'r', encoding='utf-8', errors='ignore'))
    # f = json.load(open("validationset.json", 'r', encoding='utf-8', errors='ignore'))
    docs, queries, labels = [], [], []

    # tic = datetime.datetime.now()
    # for key in f_d:
    #     docs.append(nltk.word_tokenize(f_d[key]))
    #     docs_id.append(key)
    # toc = datetime.datetime.now()
    # print("time is {}".format(toc - tic))
    #
    # tic = datetime.datetime.now()

    docs = json.load(open("docs_v1.json", "r", encoding='utf-8', errors='ignore'))
    docs_id = json.load(open("ytdocs_id.json", "r", encoding='utf-8', errors='ignore'))
    queries = json.load(open("ytqueries.json", "r", encoding='utf-8', errors='ignore'))
    # 对 询问的分词。
    # for key in f["queries"].keys():
    #     queries.append(nltk.word_tokenize(f["queries"][key]))

    tic = datetime.datetime.now()
    for i in range(len(queries)):
        for j in range(len(queries[i])):
            queries[i][j] = queries[i][j].lower()

    for i in range(len(queries)):
        tagged_sent = nltk.pos_tag(queries[i])
        wnl = nltk.WordNetLemmatizer()
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or nltk.corpus.wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原
        queries[i] = lemmas_sent
    toc = datetime.datetime.now()
    print("time is {}".format(toc - tic))
    return docs, queries, docs_id


if __name__ == '__main__':
    porter_stemmer = PorterStemmer()
    docs, queries, docs_id = get_data()

    for i in range(len(queries)):
        for j in range(len(queries[i])):
            queries[i][j] = porter_stemmer.stem(queries[i][j])

    mp = {}
    for idx in range(len(docs_id)):
        mp[docs_id[idx]] = idx

    tic = datetime.datetime.now()
    s = BM25(docs, docs_id)
    toc = datetime.datetime.now()
    print("data preprocess finish in {}".format(toc-tic))
    scores = []

    # for query in queries:
    #     score = s.simall(query)
    #     scores.append(score)

    f = json.load(open("validationset.json", 'r', encoding='utf-8', errors='ignore'))
    labels = []
    for key in f["labels"].keys():
        labels.append(f["labels"][key])

    tic = datetime.datetime.now()
    scores = []
    for query in queries:
        score = s.simall(query)
        scores.append(score)

    toc = datetime.datetime.now()

    print("solve everything in {}".format(toc-tic))
    logits = np.array(scores)

    indices = np.argsort(-logits, 1)[:, :10]
    np.save("201705130118.npy", indices)

    # print("end===")
    # tic = datetime.datetime.now()
    # sum = 0
    # for idx in range(1000):
    #     score = s.simall(queries[idx])
    #     score.sort(key=lambda s: (-s[0]))
    #     for i, tmp in enumerate(score):
    #         if i >= 10:
    #             break
    #         if tmp[1] == labels[idx][0]:
    #             sum += (1.0 / (i + 1))
    #             break
    # print(sum)
    # toc = datetime.datetime.now()
    # print("all time is {}".format(toc - tic))
    # score = s.simall(queries[1])
    # score.sort(key=lambda s: (-s[0]))
    # pos = '583461'
    # for idx, tmp in enumerate(score):
    #     if tmp[1] == pos:
    #         print(tmp, idx)





