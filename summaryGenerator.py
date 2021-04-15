from contractionMapping import contractionMapping
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from textSummarizationModel import TextSummarizationModel
from sklearn.model_selection import train_test_split
import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd 
import glob
import os
import re
import nltk
import json
nltk.download('stopwords')
class SummaryGeneratorClass:
    def __init__(self):
        self.news = []
        self.summaries = []
        
        bNews = "./BBC News Summary/News Articles/business/"
        eNews = "./BBC News Summary/News Articles/entertainment/"
        pNews = "./BBC News Summary/News Articles/politics/"
        sNews = "./BBC News Summary/News Articles/sport/"
        tNews = "./BBC News Summary/News Articles/tech/"
        self.readNews(bNews)
        self.readNews(eNews)
        self.readNews(pNews)
        self.readNews(sNews)
        self.readNews(tNews)

        bSumm = "./BBC News Summary/Summaries/business/"
        eSumm = "./BBC News Summary/Summaries/entertainment/"
        pSumm = "./BBC News Summary/Summaries/politics/"
        sSumms = "./BBC News Summary/Summaries/sport/"
        tSumm = "./BBC News Summary/Summaries/tech/"
        self.readSummaries(bSumm)
        self.readSummaries(eSumm)
        self.readSummaries(pSumm)
        self.readSummaries(sSumms)
        self.readSummaries(tSumm)
        
        self.contractionMapping = contractionMapping


#         for i in range(len(self.news)):
#             self.news[i] = self.textCleaner(self.news[i])
#         for i in range(len(self.summaries)):
#             self.summaries[i] = '_START_ '+ self.textCleaner(self.summaries[i]) + ' _END_'
        self.a = []
        self.b = []
        for i in range(len(self.news)):
            self.a.append(self.textCleaner(self.news[i]))
        for i in range(len(self.summaries)):
            self.b.append('beginmush '+ self.textCleaner(self.summaries[i]) + ' endmush')
        self.df = pd.DataFrame({'Text': self.a, 'Summary' : self.b})
        
        
    def readNews(self, directory):
        for filename in os.listdir(directory):
            with open(directory + filename,errors='replace') as infile:
                i = 1
                s = ""
                try:
                    for line in infile.readlines():
                        if i != 0:
                            if (line.isspace() == False):
                                s += str(line) 
                        i += 1
#                     s = re.sub('\n', '', s)

                    s = re.sub('\'', '',s)
                    self.news.append(s)
                except:
                    print(filename + ' is throwing an error')

    def readSummaries(self, directory):
        for filename in os.listdir(directory):
            with open(directory + filename, errors='replace') as infile:
                i = 0
                s = ""
                try:
                    for line in infile.readlines():
                        if (line.isspace() == False):
                            s += str(line) 
#                     s = re.sub('\n', '', s)
                    s = re.sub('\'', '',s)


                    self.summaries.append(s)
                except:
                    print(filename + ' is throwing an error')
        
    def textCleaner(self, string):
        stopWords = set(stopwords.words('english'))
        string = string.lower()
        
        string = ' '.join([self.contractionMapping[t] if t in self.contractionMapping else t for t in string.split(" ")])    
        
        #remove escape characters
        string = re.sub("(\\t)", ' ', str(string))
        string = re.sub("(\\r)", ' ', str(string)) 
        string = re.sub("(\\n)", ' ', str(string))
        
        #remove 's
        string = re.sub(r"'s\b","", str(string))
        
        #remove extra spaces
        string = ' '.join(string.split())
        
        #remove punctuations
        string = re.sub("[^a-zA-Z]", " ", str(string)) 
        
        #remove short words
        tokens = [w for w in string.split() if not w in stopWords]
        long_words=[]
        for i in tokens:
            if len(i)>=3:                  #removing short word
                long_words.append(i)   
        string =  (" ".join(long_words)).strip()
        
        return string

    
    def textCount(self):
        textCount = []
        summaryCount = []
        for string in self.df['Text']:
            textCount.append(len(string.split()))
            
        for sent in self.df['Summary']:
            summaryCount.append(len(sent.split()))
            

        graph = pd.DataFrame()
        graph['Text']= textCount
        graph['Summary'] = summaryCount
       
        graph.hist(bins = 100)
        plt.show()
        
        self.summaryCount = 0
        for i in self.df['Summary']:
            if(len(i.split()) <= 300):
                self.summaryCount += 1
        
        self.textCount = 0
        for i in self.df['Text']:
            if(len(i.split()) <= 400):
                self.textCount += 1
     
        self.maxTextLen = 400
        self.maxSummaryLen = 400
        
        cnt=0
        for i in self.df['Summary']:
            if(len(i.split())<=400):
                cnt=cnt+1
        print(cnt/len(self.df['Summary']))

        cnt=0
        for i in self.df['Text']:
            if(len(i.split())<=400):
                cnt=cnt+1
        print(cnt/len(self.df['Text']))

    def filterDataFrameUsingMaxTextCountAndMaxSummaryCount(self):
        textArray = np.array(self.df['Text'])
        summaryArray = np.array(self.df['Summary'])
        
        shorterTextArray = []
        shorterSummaryArray = []
        
        
        for i in range(len(textArray)):
            if(len(summaryArray[i].split()) <= self.maxSummaryLen and len(textArray[i].split()) <= self.maxTextLen):
                shorterTextArray.append(textArray[i])
                shorterSummaryArray.append(summaryArray[i])

        self.shorterDf = pd.DataFrame({'Text':shorterTextArray,'Summary':shorterSummaryArray})

    def splitData(self):
        self.xTrain, self.xVal, self.yTrain, self.yVal = train_test_split(np.array(self.shorterDf['Text']),np.array(self.shorterDf['Summary']), test_size=0.2, random_state=0, shuffle = True)

        
    def tokenizeTrainingData(self):
        tokenizerX = Tokenizer()
        tokenizerX.fit_on_texts(list(self.xTrain))
        thr=4
        count=0
        totalCount=0
        freq=0
        totalFreq=0

        for key,value in tokenizerX.word_counts.items():
            totalCount += 1
            totalFreq += 1
            if(value < thr):
                count += 1
                freq += value

        print("% of rare words in vocabulary:",(count / totalCount)*100)
        print("Total Coverage of rare words:",(freq/totalFreq)*100)
        #Text Tokenizer
        self.tokenizerX = Tokenizer(num_words=totalCount - count)
        self.tokenizerX.fit_on_texts(list(self.xTrain))
        self.xTrainSeq = self.tokenizerX.texts_to_sequences(self.xTrain) 
        self.xValSeq = self.tokenizerX.texts_to_sequences(self.xVal)
        self.xTrainSeq = pad_sequences(self.xTrainSeq, maxlen=self.maxTextLen, padding='post')
        self.xValSeq = pad_sequences(self.xValSeq, maxlen=self.maxTextLen, padding='post')
        self.xVocabularySize =  self.tokenizerX.num_words + 1

        #Summary Tokenizer
        tokenizerY = Tokenizer()
        tokenizerY.fit_on_texts(list(self.yTrain))
        for key,value in tokenizerY.word_counts.items():
            totalCount += 1
            totalFreq += 1
            if(value < thr):
                count += 1
                freq += value

        print("% of rare words in vocabulary:",(count / totalCount)*100)
        print("Total Coverage of rare words:",(freq/totalFreq)*100)
        self.tokenizerY = Tokenizer(num_words=totalCount - count)

        self.tokenizerY.fit_on_texts(list(self.yTrain))

        self.yTrainSeq = self.tokenizerY.texts_to_sequences(self.yTrain) 
        self.yValSeq = self.tokenizerY.texts_to_sequences(self.yVal)
        self.yTrainSeq = pad_sequences(self.yTrainSeq, maxlen=self.maxSummaryLen, padding='post')
        self.yValSeq = pad_sequences(self.yValSeq, maxlen=self.maxSummaryLen, padding='post')
        self.yVocabularySize =  self.tokenizerY.num_words + 1    
        
    def getSummaries(self):
        return self.summaries
    
    def getNews(self):
        return self.news
    
    def getDF(self):
        return self.df
    
    
def main():
    summaryGeneratorClass = SummaryGeneratorClass()
    summaryGeneratorClass.textCount()
    summaryGeneratorClass.filterDataFrameUsingMaxTextCountAndMaxSummaryCount()
    summaryGeneratorClass .splitData()
    summaryGeneratorClass.tokenizeTrainingData()
    print(summaryGeneratorClass.xVocabularySize)
    print(summaryGeneratorClass.yVocabularySize)
    model = TextSummarizationModel(summaryGeneratorClass.xTrainSeq, summaryGeneratorClass.yTrainSeq, summaryGeneratorClass.xValSeq, summaryGeneratorClass.yValSeq, summaryGeneratorClass.xVocabularySize, summaryGeneratorClass.yVocabularySize,summaryGeneratorClass.maxTextLen, summaryGeneratorClass.tokenizerX, summaryGeneratorClass.tokenizerY, summaryGeneratorClass.maxSummaryLen)
    
    for i in range(0, 10):
        print("Text:",model.seq2text(summaryGeneratorClass.xTrainSeq[i]))
        print("Original summary:",model.seq2summary(summaryGeneratorClass.yTrainSeq[i]))
        print("Predicted summary:",model.decodeSeq(summaryGeneratorClass.xTrainSeq[i].reshape(1,summaryGeneratorClass.maxTextLen)))
        