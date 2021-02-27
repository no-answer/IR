import numpy
import json

if __name__ == '__main__':
    # indices = numpy.load('201718130128.npy')
    # print(indices.shape)
    # labels = json.load(open('labels.json', 'r', encoding='utf-8', errors='ignore'))
    # labels = labels[0:600]
    # labels = numpy.array(labels)
    # print(labels.shape)
    a = (600, 10)
    b = (600, )
    print(a)
    print(b)
    print(a[0] == b[0])

def MRR(indices, target, k):
    """
    Compute mean reciprocal rank.
    :param logits: 2d array [batch_size x rel_docs_per_query]
    :param target: 2d array [batch_size x rel_docs_per_query]
    :return: mean reciprocal rank [a float value]
    """
    assert indices.shape[0] == target.shape

    reciprocal_rank = 0
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            if target[i] == indices[i][j]:
                reciprocal_rank += 1.0 / (j + 1)
                break

    return reciprocal_rank / indices.shape[0]