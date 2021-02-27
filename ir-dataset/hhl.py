import json
import math
import PorterStemmer
import datetime
import numpy
import json5

class BM25(object):
    def __init__(self, docs, docsID):
        self.D = len(docs)
        self.avgdl = sum([len(doc)+0.0 for doc in docs]) / self.D
        self.docs = docs
        self.docsID = docsID
        self.f = []
        self.df = {}
        self.idf = {}
        self.k1 = 2
        self.b = 0.75
        self.init()
    
    def init(self):
        for doc in self.docs:
            tmp = {}
            for word in doc:
                tmp[word] = tmp.get(word, 0)+1
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0)+1
        for k, v in self.df.items():
                self.idf[k] = math.log(self.D-v+0.5)-math.log(v+0.5)
        """
        json_f = {"idf": self.idf,
                  "f": self.f}
        json_f = json.dumps(json_f)
        with open("model.json", 'w') as json_file:
            json_file.write(json_f)
        """

    def sim(self, doc, index):
        score = 0
        d = len(self.docs[index])
        for word in doc:
            if word not in self.f[index]:
                continue
            score += (self.idf[word]*self.f[index][word]*(self.k1+1)
                      /(self.f[index][word]+self.k1*(1-self.b+self.b*d/self.avgdl)))
        return score
    
    def simall(self, doc):
        scores = []
        for index in range(self.D):
            score = self.sim(doc, index)
            scores.append(score)
        return scores
    
    def simallMRR(self, doc):
        scores = []
        for index in range(self.D):
            score = [self.sim(doc, index), docsID[index]]
            scores.append(score)
        return scores

def MRR(indices, target, k):
    for tar in target:
        tar = list(map(int, tar))
    target = numpy.array(target, dtype='int')
    # print(target)
    assert indices.shape[0] == target.shape[0]
    reciprocal_rank = 0
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            # print(type(target[i][0]), target[i][0])
            if target[i][0] == indices[i][j]:
                reciprocal_rank += 1.0 / (j + 1)
                break

    return reciprocal_rank / indices.shape[0]

def get_data():
    # 对文档集document.json进行分词和词干还原，输出是二维数组的json数据，在文件stemDocs.json中
    docJSON = json.load(open('documents.json', 'r', encoding='utf-8', errors='ignore'))
    stemDocs, docsID = [], []
    stemDocsFile = open('stemDocs.json', 'w', encoding='utf-8', errors='ignore')
    stemDocsFile.write('[')
    first = True
    docCount = 0
    for key in docJSON:
        docsID.append(key)
        stemDocs.append(PorterStemmer.Lemmatizetion(docJSON[key]))
        if not first:
            # print(key)
            stemDocsFile.write(', ')
        else:
            first = False
        stemDocsFile.write(str(stemDocs[-1]).replace('\'', '\"'))
        print(docCount)
        docCount = docCount+1
    stemDocsFile.write(']')
    stemDocsFile.close()
    print("Document count: %d" % docCount)
    print('Document set has been stemmed in file stemDocs.json')
    docsIDFile = open('docsID.json', 'w', encoding='utf-8', errors='ignore')
    docsIDFile.write(str(docsID).replace('\'', '\"'))
    docsIDFile.close()
    print('Document id set has benn saved in file docsID.json')

    # 对查询也进行词干还原
    # 提取查询部分并写入文件queriesWithoutLabel.json中，此时格式与document.json一样了
    queryJSON = json.load(open('validationset.json', 'r', encoding='utf-8', errors='ignore'))
    with open('queriesWithoutLabel.json', 'w', encoding='utf-8', errors='ignore') as f:
        json.dump(queryJSON['queries'], f)
    
    labels = []
    for key in queryJSON['labels'].keys():
        labels.append(queryJSON['labels'][key])
    labelsFile = open('labels.json', 'w', encoding='utf-8', errors='ignore')
    labelsFile.write(str(labels).replace('\'', '\"'))
    labelsFile.close()
    
    # 对去除label之后的查询集进行词干还原
    queryJSON = json.load(open('queriesWithoutLabel.json', 'r', encoding='utf-8', errors='ignore'))
    stemQueries = []
    stemQueriesFile = open('stemQueries.json', 'w', encoding='utf-8')
    stemQueriesFile.write('[')
    first = True
    for key in queryJSON:
        stemQueries.append(PorterStemmer.porterDocs(queryJSON[key]))
        if not first:
            stemQueriesFile.write(', ')
        else:
            first = False
        stemQueriesFile.write(str(stemQueries[-1]).replace('\'', '\"'))
    stemQueriesFile.write(']')
    stemQueriesFile.close()
    print('Query set has been stemmed in file stemQueries.json')
    # 预处理结束

    docs, queries = [], []
    docs = json.load(open('stemDocs.json', 'r', encoding='utf-8', errors='ignore'))
    queries = json.load(open('stemQueries.json', 'r', encoding='utf-8', errors='ignore'))

    return docs, queries, docsID, labels

def get_dataT():
    docs = json.load(open('stemDocs.json', 'r', encoding='utf-8', errors='ignore'))
    queries = json.load(open('stemQueries.json', 'r', encoding='utf-8', errors='ignore'))
    docsID = json.load(open('docsID.json', 'r', encoding='utf-8', errors='ignore'))
    labels = json.load(open('labels.json', 'r', encoding='utf-8', errors='ignore'))
    return docs, queries, docsID, labels

def get_dataD():
    # 对文档集document.json进行分词和词干还原，输出是二维数组的json数据，在文件stemDocs.json中
    docJSON = json.load(open(r'documents.json', 'r', encoding='utf-8', errors='ignore'))
    stemDocs, docsID = [], []
    # stemDocsFile = open('stemDocs.json', 'w', encoding='utf-8', errors='ignore')
    # stemDocsFile.write('[')
    first = True
    docCount = 0
    for key in docJSON:
        docsID.append(key)
        """
        stemDocs.append(PorterStemmer.Lemmatizetion(docJSON[key]))
        if not first:
            # print(key)
            stemDocsFile.write(', ')
        else:
            first = False
        stemDocsFile.write(str(stemDocs[-1]).replace('\'', '\"'))
        """
        docCount = docCount+1
        if docCount % 10000 == 0:
            print(docCount)
    # stemDocsFile.write(']')
    # stemDocsFile.close()
    print("Document count: %d" % docCount)
    print('Document set has been stemmed in file stemDocs.json')
    docsIDFile = open('docsID.json', 'w', encoding='utf-8', errors='ignore')
    docsIDFile.write(str(docsID).replace('\'', '\"'))
    docsIDFile.close()
    docs = json.load(open(r'stemDocs.json', 'r', encoding='utf-8', errors='ignore'))
    print('Document id set has benn saved in file docsID.json')
    return docs, docsID

def get_dataQ():
    # 对查询也进行词干还原
    # 提取查询部分并写入文件queriesWithoutLabel.json中，此时格式与document.json一样了
    queryJSON = json5.load(open(r'validationset.json', 'r', encoding='utf-8', errors='ignore'))
    with open('queriesWithoutLabel.json', 'w', encoding='utf-8', errors='ignore') as f:
        json5.dump(queryJSON['queries'], f)
    
    labels = []
    for key in queryJSON['labels'].keys():
        labels.append(queryJSON['labels'][key])
    labelsFile = open('labels.json', 'w', encoding='utf-8', errors='ignore')
    labelsFile.write(str(labels).replace('\'', '\"'))
    labelsFile.close()
    
    # 对去除label之后的查询集进行词干还原
    queryJSON = json5.load(open(r'queriesWithoutLabel.json', 'r', encoding='utf-8', errors='ignore'))
    stemQueries = []
    stemQueriesFile = open('stemQueries.json', 'w', encoding='utf-8')
    stemQueriesFile.write('[')
    first = True
    for key in queryJSON:
        stemQueries.append(PorterStemmer.Lemmatizetion(queryJSON[key]))
        if not first:
            stemQueriesFile.write(', ')
        else:
            first = False
        stemQueriesFile.write(str(stemQueries[-1]).replace('\'', '\"'))
    stemQueriesFile.write(']')
    stemQueriesFile.close()
    queries = json5.load(open(r'stemQueries.json', 'r', encoding='utf-8', errors='ignore'))
    print('Query set has been stemmed in file stemQueries.json')
    # 预处理结束
    return queries, labels

if __name__ == '__main__':
    docs = json.load(open('smallDocsForPorter.json', 'r', encoding='utf-8', errors='ignore'))
    tic = datetime.datetime.now()
    # docs, queries, docsID, labels = get_dataT()
    docs, docsID = get_dataD()
    queries, labels = get_dataQ()
    toc = datetime.datetime.now()
    print('data preprocess finished in {}'.format(toc - tic))

    tic = datetime.datetime.now()
    s = BM25(docs, docsID)
    toc = datetime.datetime.now()
    print('BM25 Model finished in {}'.format(toc - tic))

    #"""
    S = 100
    tic = datetime.datetime.now()
    indices = []
    index = 0
    for query in queries:
        score = s.simall(query)
        score = numpy.array(score)
        indice = numpy.argsort(-score)[:10]
        indices.append(indice)
        index = index+1
        if index >= S:
            break
    indices = numpy.array(indices)
    numpy.save('201718130128.npy', indices)
    toc = datetime.datetime.now()
    print('logits finished in {}'.format(toc - tic))
    #"""
    
    tic = datetime.datetime.now()
    """
    labels = labels[0:S]
    mrr = MRR(indices, labels, 10)
    print('MRR@10 - ', mrr)
    """
    sum = 0
    for idx in range(0, S):
        score = s.simallMRR(queries[idx])
        score.sort(key=lambda s: (-s[0]))
        for i, tmp in enumerate(score):
            if i >= 10:
                break
            if tmp[1] == labels[idx][0]:
                sum += (1.0 / (i + 1))
                break
    print(sum / S)
    # """
    toc = datetime.datetime.now()
    print('test finish in {}'.format(toc - tic))