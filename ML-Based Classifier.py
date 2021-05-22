import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import *
import string
import re
from random import shuffle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nrclex import NRCLex 
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn.neural_network import MLPClassifier as NN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline

import pandas as pd
import re
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize as wt 
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
#spell correction
from autocorrect import spell
# split train and test data
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn.neural_network import MLPClassifier as NN
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
import emoji
sentiment_analyzer = VS()

NEGATE = {
        "aint",
        "arent",
        "cannot",
        "cant",
        "couldnt",
        "darent",
        "didnt",
        "doesnt",
        "ain't",
        "aren't",
        "can't",
        "couldn't",
        "daren't",
        "didn't",
        "doesn't",
        "dont",
        "hadnt",
        "hasnt",
        "havent",
        "isnt",
        "mightnt",
        "mustnt",
        "neither",
        "don't",
        "hadn't",
        "hasn't",
        "haven't",
        "isn't",
        "mightn't",
        "mustn't",
        "neednt",
        "needn't",
        "never",
        "none",
        "nope",
        "nor",
        "not",
        "nothing",
        "nowhere",
        "oughtnt",
        "shant",
        "shouldnt",
        "uhuh",
        "wasnt",
        "werent",
        "oughtn't",
        "shan't",
        "shouldn't",
        "uh-uh",
        "wasn't",
        "weren't",
        "without",
        "wont",
        "wouldnt",
        "won't",
        "wouldn't",
        "rarely",
        "seldom",
        "despite",
    }
Positive = "Positive"
Negative="Negative"
Neutral="Neutral"
emoticonsDict= { "smilingfacewithhalo": Positive
,"smilingfacewithhorns": Positive
,"smilingfacewithsunglasses": Positive
,"neutralface": Neutral
,"expressionlessface": Negative
,"expressionless":Negative
,"confusedface": Negative
,"kissingface": Positive
,"kissingfacewithsmilingeyes": Positive
,"facewithstuck-outtongue": Positive
,"worriedface": Negative
,"frowningfacewithopen}mouth": Negative
,"anguishedface": Negative
,"grimacingface": Negative
,"facewithopenmouth": Negative
,"hushedface": Negative
,"sleepingface": Negative
,"facewithoutmouth": Neutral
,"smilingfacewithsmilingeyes": Positive
,"thumbsup": Positive
,"smilingfacewithhearteyes": Positive
,"hundredpoints": Positive
,"poutingface": Negative
,"clappinghands": Positive
,"cherryblossom": Positive
,"sparkles": Positive
,"personcartwheeling": Positive
,"starstruck": Positive
,"angryface": Negative
,"smilingcatfacewithhearteyes": Positive
,"glowingstar": Positive
,"twohearts": Positive
,"redheart": Positive
,"beamingfacewithsmilingeyes": Positive
,"faceblowingakiss": Positive
,"grinningfacewithsweat": Positive
,"winkingfacewithtongue": Positive
,"facewithtearsofjoy": Positive
,"redheartselector": Positive
,"smilingfacewithhalo": Positive
,"raisinghands": Positive
,"wavinghand": Positive
,"foldedhands": Positive
,"winkingface": Positive
,"beamingfacewithsmilingeyes": Positive
,"grinningfacewithsweat": Positive
,"winkingfacewithtongue": Positive
,"clappinghands": Positive
,"faceblowingakiss": Positive
,"smilingfacewithhearteyes": Positive
,"growingheart": Positive
,"tiredface": Negative
,"foldedhandsmediumlightskintone": Positive
,"frowningface": Negative
,"smilingfaceselector": Positive
,"kissingfacewithclosedeyes": Positive
,"laughingbiggrinorlaughwithglasses": Positive
,"smileyembarrassedorblushing":Negative
,"hearteyes": Positive
,"clap":Positive

,"star2":Positive
,"hearteyescat": Positive
,"clap":Positive

}

def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), str(text))
    return text

def encounterNegation(tagged,text):
    for i in range(len(tagged)-1):
        if text[i]in NEGATE and tagged[i+1][1]== 'JJ':
            return True
    return False

def isQuestion(tagged):
    for i in range(len(tagged)-1):
        return any(['WP', 'VBZ'] == [tagged[i][1 ], tagged[i+1][1]] for i in range(len(tagged) - 1))

def other_features(featuresMatrix,data):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features"""
    
    syllables = []
    num_chars=[]
    num_chars_total=[]
    num_terms=[]
    num_words=[]
    avg_syl=[]
    num_unique_terms=[]
    FKRA=[]
    FRE=[]
    sentimentN=[]
    sentimentP=[]
    sentimentT=[]
    space_pattern = '\s+'
    for i in range(len(data)):
        tweet = data[i]
        sentiment = sentiment_analyzer.polarity_scores(tweet)
        sentimentN.append(sentiment['neg'])
        sentimentN.append(sentiment['pos'])
        sentimentN.append(sentiment['neu'])
        
        
        words = re.sub(space_pattern, ' ', tweet)
         
        syll= textstat.syllable_count(words)
        syllables.append(syll)
        x= sum(len(w) for w in words)
        num_chars.append(x)
        x = len(tweet)
        num_chars_total.append(x)
        x= len(tweet.split())
        num_terms.append(x)
        num_wos = len(words.split())
        num_words.append(num_wos)
        avg_sylv= round(float((syll+0.001))/float(num_wos+0.001),4)
        avg_syl.append(avg_sylv)
        x= len(set(words.split()))
        num_unique_terms.append(x)
        
        ###Modified FK grade, where avg words per sentence is just num words/1
        x= round(float(0.39 * float(num_wos)/1.0) + float(11.8 * avg_sylv) - 15.59,1)
        FKRA.append(x)
        ##Modified FRE score, where sentence fixed to 1
        x= round(206.835 - 1.015*(float(num_wos)/1.0) - (84.6*float(avg_sylv)),2)
        FRE.append(x)
                    
    featuresMatrix.append(syllables) 
    featuresMatrix.append(num_chars)
    featuresMatrix.append(num_chars_total) 
    featuresMatrix.append(num_terms) 
    featuresMatrix.append(num_words) 
    featuresMatrix.append(avg_syl) 
    featuresMatrix.append(num_unique_terms) 
    featuresMatrix.append(FKRA)
    featuresMatrix.append(FRE) 
           
    return featuresMatrix    

def preProcess(dataset):
    data = []
    #lists to hold the features
    isQuestionL=[]
    hasNegation=[]
    PList=[]
    TList=[]
    NList=[]
        
    for i in range(dataset.shape[0]):
        processed = []
        hasNegationFlag=0
        positiveEmoticon=0
        negativeEmoticon=0
        neutralEmoticon=0
        
        #start pre-processing
        tweet = dataset.iloc[i, 1]
        tweet = convert_emoticons(tweet)
        tweet= emoji.demojize(tweet, delimiters=(" ", " "),use_aliases=True)
        
        # remove non alphabatic characters
        tweet = re.sub('[^A-Za-z]', ' ', tweet)
    
        # make words lowercase, because Go and go will be considered as two words
        tweet = tweet.lower()
    
        # tokenising ans pos : to use for features isQuestion, hasNegationFlag
        tokenized = nltk.word_tokenize(tweet)
        pos_val = nltk.pos_tag(tokenized)
        
        # FEATURE: isQuestionL : if the message contains a question
        flag = isQuestion(pos_val)
        isQuestionL.append(flag)
        
        # FEATURE: hasNegationFlag : if the message contains a negation preceding an adjective
        # example: this is not good
        flag= encounterNegation(pos_val,tokenized)    
        hasNegation.append(flag)
        
        for word in tokenized:
            if word not in set(stopwords.words('english')):
               if word in emoticonsDict:
                   #when found positive emoticon, replace its position with word POSTITIVE, and increment counter
                   if emoticonsDict[word]==Positive:
                      processed.append('POSITIVE')
                      positiveEmoticon+=0
                      continue
                   elif emoticonsDict[word]==Negative:
                      processed.append('NEGATIVE')
                      negativeEmoticon+=1
                      continue
                   elif emoticonsDict[word]==Neutral:
                      processed.append('NEUTRAL')
                      neutralEmoticon+=1
                      continue
            processed.append(word)   #should we stemm it
            
        
         
        # FEATURES: PList: counter for occurences of positive emoticons
        PList.append(positiveEmoticon)
        NList.append(negativeEmoticon)
        TList.append(neutralEmoticon)
        
        tweet_text = " ".join(processed)
        data.append(tweet_text)
    return data,isQuestionL,PList,NList,TList,hasNegation
    

def get_class_counts(df):
	grp = df.groupby(['classA'])[id].nunique()
	return {key: grp[key] for key in list(grp.keys)}

def get_class_proportions(df):
	class_counts = get_class_counts(df)
	return {val[0]: round(val[1]/df.shape[0],3) for val in class_counts.items()}
#============================== MAIN =============================

Columns=['classA','msg']
#UPDATE filepath, filename, tweet = dataset.iloc[i, 1], y = dataset.iloc[:, 0] 
filePath = "path to data folder/"
dataset = pd.read_csv(filePath+"Data1766.csv", encoding='utf-8-sig',usecols=Columns,names=Columns);
# dataset[:0].hist()
# train,test = train_test_split(dataset,test_size=0.2)
# trainClassPropogation = get_class_proportions(train)
# testClassPropogation = get_class_proportions(test)
# print("Train",trainClassPropogation)
# print("Test",trainClassPropogation)
    
#PreProcess the data and build features isQuestionL,PList,NList,TList,hasNegation
data,isQuestionL,PList,NList,TList,hasNegation = preProcess(dataset)   


# FEATURE: count_vector : BOW
matrix = CountVectorizer()
count_vector = matrix.fit_transform(data).toarray()

# build your featuresMatrix

featuresMatrix = pd.DataFrame(count_vector)
# featuresMatrix.append(dataset.iloc[:, 1])
# featuresMatrix.append(isQuestionL)
# featuresMatrix.append(PList)
# featuresMatrix.append(NList)
# featuresMatrix.append(TList)
# featuresMatrix.append(hasNegation)


#add other features
# featuresMatrix = other_features(featuresMatrix,data)
# the column containing the classes
y = dataset.iloc[:, 0]

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, chi2

print("BEFORE SELECTION featuresMatrix",featuresMatrix.shape)
# sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# featuresMatrix =  sel.fit_transform(featuresMatrix)
featuresMatrix = SelectPercentile(chi2, percentile=20).fit_transform(featuresMatrix, y)
# featuresMatrix = SelectKBest(chi2, k=2).fit_transform(featuresMatrix,y)
print("AFTER SELECTION featuresMatrix",featuresMatrix.shape)
 

print("featuresMatrix",featuresMatrix.shape)
 
# print("=================START CLASSIFICATION USING 80 (80/20) /20 split ===============")


# X_train, X_test, y_train, y_test = train_test_split(featuresMatrix, y,random_state=42, test_size=0.2)


# Classifiers: SVM, NB, NN, DT, LR
classifiers = {"SVM": SVC(), 
                "NaiveBayes": NB(), 
                # "NeuralNetwork": NN(), 
                "DecisionTree": DT(), 
                "LogisticRegression": LR(),
                'GaussianNB':GaussianNB()}

# for name,classifier in classifiers.items():
     
#     print("============{}=============".format(name))
#     classifier.fit(X_train, y_train)
#     print("-------- Now start predicting using",name)
#     # predict class
#     y_pred = classifier.predict(X_test)
#     # Confusion matrix
#     print("-------- Now print confusion matrix for using",name)
#     report = classification_report( y_test, y_pred)
#     print(report)


K=5
print("=================START CLASSIFICATION USING StratifiedKFold k={} ===============".format(K))

from sklearn.model_selection import StratifiedKFold, GridSearchCV

# kfold = KFold(n_splits=K, shuffle=False)
kfold = StratifiedKFold(n_splits=K,random_state=None) 
# Shuffle Data
features,labels = np.array(featuresMatrix), np.array(y)
print("K-fold Cross Validation (K=%d)\n\n" % K)
for name,classifier in classifiers.items():
    NPrecision = 0
    accu=[]
    print("============{}=============".format(name))
    for train_indices,test_indices in kfold.split(features,labels):
        
        X_train, X_test = features[train_indices], features[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        param_grid = [{}] # Optionally add parameters here    
        classifier = GridSearchCV(classifier, param_grid, cv=3,
                           scoring='accuracy')
         # classifier.fit(features[train_indices], labels[train_indices])
        # classifier.predict(features[train_indices]) 
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        # print("X_test",X_test)
        # print("y_test",y_test)
        # print("y_pred",y_pred)
        report = classification_report( y_test, y_pred)
        accu.append( accuracy_score(labels[test_indices], classifier.predict(features[test_indices])) )
      
    print("Avg Accuracy: %.2f" % (sum(accu)/len(accu)) )
    print(report)    
    # zipped = list(zip(features[test_indices],labels))
    # shuffle(zipped)
    # features[test_indices],labels = zip( *zipped )
    # classifier.fit(features[test_indices],labels)
    # print(name+":\n", classification_report(labels, classifier.predict(features[test_indices])),"\n\n")
    
# dpredict = pd.DataFrame(data = y_test)
# dpredict.to_csv (filePath+"prediction_y_test.csv", index = False, header=True,encoding='utf-8-sig')
 
# dpredict = pd.DataFrame(data = y_pred)
# dpredict.to_csv (filePath+"prediction_y_preds.csv", index = False, header=True,encoding='utf-8-sig')
 
# dpredict = pd.DataFrame(data = X_test)
# dpredict.to_csv (filePath+"prediction_X_test.csv", index = False, header=True,encoding='utf-8-sig')

    
# print("Classifier (Accuracy, Precision, Recall, F-Score) Comparison:\n\n")

# for name,classifier in classifiers.items():
#     zipped = list(zip(features[test_indices],labels))
#     shuffle(zipped)
#     features[test_indices],labels = zip( *zipped )
#     classifier.fit(features[test_indices],labels)
#     print(name+":\n", classification_report(labels, classifier.predict(features[test_indices])),"\n\n")
                            