import pandas as pd 
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from nrclex import NRCLex
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# ML Libraries
import nltk
from nrclex import NRCLex 
import emoji
import re
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
import matplotlib.pyplot as plt

porter = PorterStemmer()
lancaster=LancasterStemmer()
from nltk.stem import WordNetLemmatizer 

def buildLexicon():
    lexicon = {}
    for key,value in NRCLex.lexicon_Original.items():
        lexicon[porter.stem(key)]= value
    return  lexicon   

#read the raw data CSV file
lexicon = buildLexicon()
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

column_Positive= 'positive'
column_Negative= 'negative' 
column_Nuetral= 'neutral'
column_Compund= 'compound' 
column_manual='manual'
column_Anger = 'anger'
column_Anticipation = 'anticipation'
column_Disgust = 'disgust'
column_Fear= 'fear'
column_Joy = 'joy'
column_Sadness = 'sadness' 
column_Surprise = 'surprise'   
column_Trust= 'trust'
 
filePath = "filepath/"
# stop_words = set(stopwords.words('english'))
stopwords = nltk.corpus.stopwords.words("english")

def pos(word):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    pos_val = nltk.pos_tag([word])
    tag = pos_val[0][1]
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), str(text))
    return text

def cleanData(docs):
    cleanList=[]
    for index in range(len(docs)):
        row = docs[index]
        #replace emoji with words
        row = convert_emoticons(row)
        row= emoji.demojize(row, delimiters=(" ", " "),use_aliases=True)
        
        row=str(row).lower()
        
        # Remove urls
        row = re.sub(r"http\S+|www\S+|https\S+", '', row, flags=re.MULTILINE)
        # Remove user @ references and '#' from tweet
        row = re.sub(r'\@\w+|\#','', row)
        # Remove punctuations
        row = row.translate(str.maketrans('', '', string.punctuation))
        row_tokens = word_tokenize(row)
        filtered_words = [w for w in row_tokens if not w in stopwords]
        if len(filtered_words)==0:
            continue
        cleanRow=''
        for word in filtered_words:
            if word in emoticonsDict:
                if emoticonsDict[word]==Positive:
                    cleanRow+=' '
                    cleanRow+=Positive
                    annotateNRC('Positive')
                    continue
                elif emoticonsDict[word]==Negative:
                    cleanRow+=' '
                    cleanRow+=Negative
                    annotateNRC('Negative')
                elif emoticonsDict[word]==Neutral:
                    cleanRow+=' '
                    cleanRow+=Neutral
                    annotateNRC('Neutral')
                    continue
            cleanRow+=' '
            cleanRow+=word
            annotateNRC(word)
        cleanList.append(cleanRow)    
    return cleanList



def annotateNRC(word):
    emotion = NRCLex(word,lexicon,True)
    # print(" ***** annotateNRC for word: ",word)
    #if not found try search for the stem of the word  using the stemmed lexicon
    # if len(emotion.affect_list)<=0:
    #     emotion = NRCLex(str(porter.stem(word)),lexicon,False)
    
    #If the word still doesn't exist in NRC, increment the corresponding array
    if len(emotion.affect_list)<=0:
        if word in NotNRCWords:
            x = NotNRCWords[word]
            x+=1
            NotNRCWords.update({word:x})
        else:
            NotNRCWords.update({word:1})
        # print(" ***** annotateNRC for word: NO FOUND")
        return ''
    #If word found in Lexicon, fetch the right annotation
    for affect in emotion.affect_list:
        if affect==column_Positive:
            if word in PositiveWords:
                x = PositiveWords[word]
                x+=1
                PositiveWords.update({word:x})
            else:
                PositiveWords.update({word:1})
        elif affect==column_Negative:
            if word in NegativeWords:
                x = NegativeWords[word]
                x+=1
                NegativeWords.update({word:x})
            else:
                NegativeWords.update({word:1})
        elif affect==column_Anger:
            if word in AngerWords :
                x = AngerWords[word]
                x+=1
                AngerWords.update({word:x})
            else:
                AngerWords.update({word:1})
        elif affect==column_Anticipation:
            if word in AnticipationWords:
                x = AnticipationWords[word]
                x+=1
                AnticipationWords.update({word:x})
            else:
                AnticipationWords.update({word:1})
        elif affect==column_Disgust:
            if word in DisgustWords:
                x = DisgustWords[word]
                x+=1
                DisgustWords.update({word:x})
            else:
                DisgustWords.update({word:1})
        elif affect==column_Fear:
            if word in FearWords:
                x = FearWords[word]
                x+=1
                FearWords.update({word:x})
            else:
                FearWords.update({word:1})
        elif affect==column_Joy:
            if word in JoyWords:
                x = JoyWords[word]
                x+=1
                JoyWords.update({word:x})
            else:
                JoyWords.update({word:1})
        elif affect==column_Sadness:
            if word in SadnessWords:
                x = SadnessWords[word]
                x+=1
                SadnessWords.update({word:x})
            else:
                SadnessWords.update({word:1})
        elif affect==column_Surprise:
            if word in SurpriseWords:
                x = SurpriseWords[word]
                x+=1
                SurpriseWords.update({word:x})
            else:
                SurpriseWords.update({word:1})
        elif affect==column_Trust:
            if word in TrustWords:
                x = TrustWords[word]
                x+=1
                TrustWords.update({word:x})
            else:
                TrustWords.update({word:1})
        elif affect==column_Nuetral:
            if word in NeutralWords:
                x = NeutralWords[word]
                x+=1
                NeutralWords.update({word:x})
            else:
                NeutralWords.update({word:1})
    return    
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def plotEmotionBarChart(emotionDict,title):
    average = sum(emotionDict.values())/len(emotionDict)
    sortedDict = dict(sorted(emotionDict.items(), key=lambda item: item[1]))
    items = sorted(((k, v) for k, v in sortedDict.items() if v >=10), reverse=True)

    words = []
    occurances=[]
    for k, v in sortedDict.items():
        if v>=10:
            words.append(k)
            occurances.append(v)   
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(words,occurances)
    plt.xticks(rotation=90)
    plt.gca().invert_xaxis()
    plt.title("Highest used words expressing {} (based on NRC Lexicon)".format(title)) 
    plt.show()



def plotEmotionsComparison():
    emotionDict ={"Sadness":sum(SadnessWords.values()),
           "Anger":sum(AngerWords.values()),
           "Fear":sum(FearWords.values()),
           # "Anticipation":sum(AnticipationWords.values()),
           # "Negative":sum(NegativeWords.values()),
           # "Surprise":sum(SurpriseWords.values()),
           "Disgust":sum(DisgustWords.values()),
           # "Positive":sum(PositiveWords.values()),
           # "Joy":sum(JoyWords.values()),
           # "Trust":sum(TrustWords.values()),
           # "Neutral":sum(NeutralWords.values()),
           # "Non NRC":sum(NotNRCWords.values()),
           }
    sortedDict = dict(sorted(emotionDict.items(), key=lambda item: item[1]))
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(sortedDict.keys(),sortedDict.values())
    plt.xticks(rotation=90)
    plt.gca().invert_xaxis()
    plt.title("Comparison between Emotions (based on NRC Lexicon)") 
    plt.show()

import csv
def writeDictToFile(dict_data):
    csv_columns = ['word','tfidf']
    csv_file = filePath+"iifid.csv"
    # try:
    #    with open('dict.csv', 'w') as csv_file:  
    #         writer = csv.writer(csv_file)
    #         for key, value in dict_data.items():
    #            writer.writerow([key, value])
    # except IOError:
    #     print("I/O error")
    keys = list(dict_data.keys())
    values = list(dict_data.values())
    dataset = pd.DataFrame({'Column1': keys, 'Column2': values})
    dataset.to_csv (csv_file, index = False, header=True,encoding='utf-8-sig')
             
          
    
    
def TfIdF(data):
    tfidf_vectorizer=TfidfVectorizer(use_idf=True,analyzer='word', stop_words='english') 
    dataT = data.iloc[:, 0]
    # print(len(dataT))
    # just send in all your docs here 
    tfidf_matrix=tfidf_vectorizer.fit_transform(dataT)
    # print(tfidf_matrix.shape)
    # print(tfidf_matrix.shape[0]) #number of vectors
    # print(tfidf_matrix.shape[1]) #number of words
    
    # v1=tfidf_matrix[1] 
    # print(v1.shape)
    # print(v1)
    words = tfidf_vectorizer.get_feature_names()
    maxWeightW=[0] * len(words)
    # print("words len", len(words))
    # print("words",words)
    
    siftedDict={}        
    # len(words)
    for i in range(len(words)):
        if i==100:
            writeDictToFile(siftedDict)
            break
            
        print("word -->",i, words[i])
        if words[i] in siftedDict.keys():
           print("found or zero -->",words[i],siftedDict[words[i]] )
           continue
        else:
            for j in range(tfidf_matrix.shape[0]):
                print("else vector --> word",i,j)
                v=tfidf_matrix[j]
                df = pd.DataFrame(v.T.todense(), index=words, columns=["tfidf"]) 
                tfidf_W = df["tfidf"][i]
                siftedDict.update({words[i]:tfidf_W})
                if tfidf_W>0:
                   break 
                # print("-->",i,words[i],j,tfidf_W)
    print("size of sifted dict",len(siftedDict.keys())) 
    return

NotNRCWords = {}
AngerWords = {}
FearWords = {}
AnticipationWords = {}
NegativeWords = {}
SurpriseWords = {}
PositiveWords={}
NeutralWords={}
JoyWords={}
TrustWords={}
DisgustWords={}
SadnessWords={}
    
Columns=['classA','msg','positive','negative','neutral']
#UPDATE filepath, filename, tweet = dataset.iloc[i, 1], y = dataset.iloc[:, 0] 

data = pd.read_csv(filePath+"predicted10KNegative.csv", encoding='utf-8-sig',usecols=Columns,names=Columns);
dataset = cleanData(data.iloc[:, 0])



# #Plot highest used words to express the emotion
# plotEmotionBarChart(NegativeWords,'Negative')
# plotEmotionBarChart(SadnessWords,'Sadness')
# plotEmotionBarChart(AngerWords,'Anger')
# plotEmotionBarChart(FearWords,'Fear')
# plotEmotionBarChart(AnticipationWords,'Anticipation')
# plotEmotionBarChart(SurpriseWords,'Surprise')
# plotEmotionBarChart(DisgustWords,'Disgust')
# plotEmotionBarChart(PositiveWords,'Positive')
# plotEmotionBarChart(JoyWords,'Joy')
# plotEmotionBarChart(TrustWords,'Trut')
# plotEmotionBarChart(NotNRCWords,'Not NRC words')


#Plot highest expressed emotion
plotEmotionsComparison()




