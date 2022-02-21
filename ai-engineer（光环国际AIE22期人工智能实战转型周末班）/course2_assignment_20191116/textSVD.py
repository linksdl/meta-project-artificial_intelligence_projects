# -*- coding: utf-8 -*-


import re
import math
import numpy as np
import pylab
from matplotlib import pyplot
from scipy import linalg

# 文档
documents = [
    "Roronoa Zoro, nicknamed \"Pirate Hunter\" Zoro, is a fictional character in the One Piece franchise created by Eiichiro Oda.",
    "In the story, Zoro is the first to join Monkey D. Luffy after he is saved from being executed at the Marine Base. ",
    "Zoro is an expert swordsman who uses three swords for his Three Sword Style, but is also capable of the one and two-sword styles. ",
    "Zoro seems to be more comfortable and powerful using three swords, but he also uses one sword or two swords against weaker enemies.",
    "In One Piece, Luffy sails from the East Blue to the Grand Line in search of the legendary treasure One Piece to succeed Gol D. Roger as the King of the Pirates. ",
    "Luffy is the captain of the Straw Hat Pirates and along his journey, he recruits new crew members with unique abilities and personalities. ",
    "Luffy often thinks with his stomach and gorges himself to comical levels. ",
    "However, Luffy is not as naive as many people believe him to be, showing more understanding in situations than people often expect. ",
    "Knowing the dangers ahead, Luffy is willing to risk his life to reach his goal to become the King of the Pirates, and protect his crew.",
    "Adopted and raised by Navy seaman turned tangerine farmer Bellemere, Nami and her older sister Nojiko, have to witness their mother being murdered by the infamous Arlong.",
    "Nami, still a child but already an accomplished cartographer who dreams of drawing a complete map of the world, joins the pirates, hoping to eventually buy freedom for her village. ",
    "Growing up as a pirate-hating pirate, drawing maps for Arlong and stealing treasure from other pirates, Nami becomes an excellent burglar, pickpocket and navigator with an exceptional ability to forecast weather.",
    "After Arlong betrays her, and he and his gang are defeated by the Straw Hat Pirates, Nami joins the latter in pursuit of her dream."
]

documents = []
with open('textSVD.txt', 'r') as file:
    for line in file:
        documents.append(line.strip())

print(len(documents))
# 停用词
stopwords = ['a', 'an', 'after', 'also', 'and', 'as', 'be', 'being', 'but', 'by', 'd', 'for', 'from', 'he', 'her',
             'his', 'in', 'is', 'more', 'of', 'often', 'the', 'to', 'who', 'with', 'people']
# 要去除的标点符号的正则表达式
punctuation_regex = '[,.;"]+'
# map,key是单词,value是单词出现的文档编号
dictionary = {}

# 当前处理的文档编号
currentDocId = 0

# 依次处理每篇文档
for d in documents:
    words = d.split()
    # 文章总词数
    doc_word_lens = 0
    # word 出现的次数
    word_count = 0
    for w in words:
        word_count = words.count(w)
        # 去标点
        w = re.sub(punctuation_regex, '', w.lower())
        if w in stopwords:
            doc_word_lens += 1
            continue
        elif w in dictionary:
            doc_word_lens += 1
            dictionary[w].append(currentDocId)
        else:
            doc_word_lens += 1
            dictionary[w] = [currentDocId]
    currentDocId += 1

# 至少出现在两个文档中的单词选为关键词
keywords = [k for k in dictionary.keys() if len(dictionary[k]) > 1]
keywords.sort()
print("keywords:\n", keywords, "\n")

# 文章的总数
doc_length = len(documents)


# 逆文本频率指数
def idf_fun(word):
    doc_word_len = 0
    for doc in documents:
        word_s = doc.split()
        if word in word_s:
            doc_word_len += 1
    return float(math.log(doc_length / (doc_word_len + 1)))


##
key_tf_idf = {}

for key_w in keywords:

    for d in documents:
        words = d.split()
        # 文章总词数
        doc_word_lens = 0
        # word 出现的次数
        word_count = words.count(key_w)
        for w in words:
            # 去标点
            w = re.sub(punctuation_regex, '', w.lower())
            if w in stopwords:
                continue
            else:
                doc_word_lens += 1
        # 词频
        tf = float(word_count / doc_word_lens)

        if key_w in key_tf_idf:
            key_tf_idf[key_w].append(tf * idf_fun(key_w))
        else:
            key_tf_idf[key_w] = [tf * idf_fun(key_w)]

print(key_tf_idf)

dic_tf_idf = []
for key_w in keywords:
    dic_tf_idf.append(key_tf_idf[key_w])

print(dic_tf_idf)
# 生成word-document矩阵
# X = np.zeros([len(keywords), currentDocId])
# for i, k in enumerate(keywords):
#     for d in dictionary[k]:
#         X[i, d] += 1
# tf-idf
X = np.matrix(dic_tf_idf)
# 奇异值分解
U, sigma, V = linalg.svd(X, full_matrices=True)

print("U:\n", U, "\n")
print("SIGMA:\n", sigma, "\n")
print("V:\n", V, "\n")

# 得到降维(降到targetDimension维)后单词与文档的坐标表示
targetDimension = 2
U2 = U[0:, 0:targetDimension]
V2 = V[0:targetDimension, 0:]
sigma2 = np.diag(sigma[0:targetDimension])
print(U2.shape, sigma2.shape, V2.shape)

# 对比原始矩阵与降维结果
X2 = np.dot(np.dot(U2, sigma2), V2);
print("X:\n", X);
print("X2:\n", X2);

# 开始画图
pyplot.title("LSA")
pyplot.xlabel(u'x')
pyplot.ylabel(u'y')

# 绘制单词表示的点
# U2的每一行包含了每个单词的坐标表示(维度是targetDimension)，此处使用前两个维度的坐标画图
for i in range(len(U2)):
    pylab.text(U2[i][0], U2[i][1], keywords[i], fontsize=10)
    print("(", U2[i][0], ",", U2[i][1], ")", keywords[i])
x = U2.T[0]
y = U2.T[1]
pylab.plot(x, y, '.')

# 绘制文档表示的点
# V2的每一列包含了每个文档的坐标表示(维度是targetDimension)，此处使用前两个维度的坐标画图
Dkey = []
for i in range(len(V2[0])):
    pylab.text(V2[0][i], V2[1][i], ('D%d' % (i + 1)), fontsize=10)
    print("(", V2[0][i], ",", V2[1][i], ")", ('D%d' % (i + 1)))
    Dkey.append('D%d' % (i + 1))
x = V[0]
y = V[1]
pylab.plot(x, y, 'x')

pylab.savefig("textSVD-tfidf.png", dpi=100)


# exit()
def cosine_similarity(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0
    else:
        return dot_product / ((normA ** 0.5) * (normB ** 0.5))


def similarity_rank(U, key):
    keyList = []
    scoreList = []
    for i in range(0, len(U) - 1):
        for j in range(i + 1, len(U)):
            score = cosine_similarity(U[i], U[j])
            scoreList.append(score)
            keyList.append(str(key[i]) + ' - ' + str(key[j]))
    print(keyList[(scoreList.index(max(scoreList)))] + ' has min distance with ' + str(max(scoreList)))


similarity_rank(U2, keywords)
similarity_rank(np.transpose(np.array(V2)), Dkey)

# 输入是什么
# 分解的矩阵是什么
# 图是什么含义
# 和文档12最相似的文档是哪一个
# 和词nami最相近的文档是哪一个
# 和词nami最相近的词是哪一个
