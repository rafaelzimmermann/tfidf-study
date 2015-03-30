from pyspark import SparkContext

import nltk
import os
import re
from nltk.corpus import mac_morpho


logFile = "./data/pages_xml_splits/0/*"  # Should be some file on your system
sc = SparkContext("local", "Simple App")
logData = sc.wholeTextFiles(logFile)

def strip_xml(data):
    return re.sub('<[^<]+>', "", data)

def tokenize(data):
    words = nltk.tokenize.word_tokenize(strip_xml(data[1]))
    return [(word, data[0]) for word in words]

def mapByWord(data):
    word = data[0][0]
    docId = data[0][1]
    wordTotal = data[1]

    return word, ([(docId, wordTotal)], wordTotal)

def countTotalWord(a, b):
    total1 = a[1]
    total2 = b[1]
    totalDoc1 = a[0]
    totalDoc2 = b[0]

    return (totalDoc1+totalDoc2, total1+total2)

def tfidf(frequency, documentFrequency):
    return frequency/ float(documentFrequency)

def mapTfidf(data):
    word = data[0]
    wordDocuments = data[1][0]
    documentFrequency = data[1][1]

    return ((wordDoc[0], [(word, tfidf(wordDoc[1], documentFrequency))]) for wordDoc in wordDocuments)



def reduceTfidf(a, b):
    return a + b

counts = logData.flatMap(tokenize).map(lambda docWord: (docWord, 1)).reduceByKey(lambda a, b: a + b)

counts = counts.map(mapByWord).reduceByKey(countTotalWord).flatMap(mapTfidf).reduceByKey(reduceTfidf)
# .flatMap(lambda a: (a[0], a[1]))

counts.saveAsTextFile("./data/out/")

sc.stop()


#https://radimrehurek.com/gensim/
#word2vector
#mllib
