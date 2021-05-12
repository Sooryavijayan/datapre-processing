# datapre-processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn. metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

#### importing dataset
fake = pd.read_csv("fake.csv")
true = pd.read_csv("true.csv")
fake.head()
true.head()

### adding new column to the dataset
fake["class"] = 'fake'
true["class"] = 'true'

### readind shape of dataset
fake.shape
true.shape
data = pd.concat([fake,true]).reset_index(drop = True)
data.shape

### shuffle bfake.csv and true.csv into data
from sklearn.utils import shuffle
data = shuffle(data)

### drop title and date coloumn from dataset
data = data.reset_index(drop = True)
data.drop(["title"],axis = 1,inplace = True)
data.head()
data.drop(["date"],axis = 1,inplace = True)
data.head()
### chang dataset into lowercase
data['text'] = data['text'].apply(lambda x: x.lower())
data.head()

### remove punctuation from dataset
import string
def punctuation_removal(text):
    all_list = [char for char in text if char not in string .punctuation]
    clean_str = ''.join(all_list)
    return clean_str
data['text'] = data['text'].apply(punctuation_removal)

### wordcloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
import sys,os
os.chdir(sys.path[0])
text=open('fake.csv',mode='r',encoding='utf-8').read()
text=open('true.csv',mode='r',encoding='utf-8').read()
stopwords=STOPWORDS
wc=WordCloud(
        background_color='red',
        stopwords=stopwords,
        height=1200,
        width=1200
)
wc.generate(text)
wc.to_file('wordcloud.png')

### plot bar
print(data.groupby(['subject'])['text'].count())
data.groupby(['subject'])['text'].count().plot(kind = "bar")
plt.show()
print(data.groupby(['class'])['text'].count())
data.groupby(['class'])['text'].count().plot(kind = "bar")
plt.show()

### tokrnization
import nltk

from nltk import tokenize

token_space = tokenize.WhitespaceTokenizer()

def counter(text, column_text, quantity):
    all_words =''.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"word": list(frequency.keys()),
                                "frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "frequency",n = quantity)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency,x = "word", y = "frequency", color = "blue")
    ax.set(ylabel = "count")
    plt.xticks(rotation = 'vertical')
    plt.show()
    counter(data[data["class"] =="fake"], "text",20)
    counter(data[data["class"]=="true"],"text" ,20)
    x=data['text']
y=data['class']

### confusion matrix
from sklearn import metrics
import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                         normalize=False,
                         title='confusion matrix',
                         cmap= plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation =45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("normalized confusion matrics")
    else:
        print('confusion matrics,without normalization')
    thresh=cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                color="white" if cm[i,j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('predicted label')
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.2, random_state =10)
        len(x_train)
        len(x_test)
        from sklearn.linear_model import LogisticRegression
pipe = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('model', LogisticRegression())])
model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("accuracy:{} %".format(round(accuracy_score(y_test, prediction)*100,2)))
cm = metrics.confusion_matrix(y_test,prediction)
plot_confusion_matrix(cm,classes=['fake','true'])
from sklearn.tree import DecisionTreeClassifier
pipe = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('model', DecisionTreeClassifier(criterion='entropy',
                                                max_depth = 20,
                                                splitter='best',
                                                 random_state=42))])
model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("accuracy:{} %".format(round(accuracy_score(y_test, prediction)*100,2)))
cm = metrics.confusion_matrix(y_test,prediction)
plot_confusion_matrix(cm,classes=['fake','true'])
from sklearn.ensemble import RandomForestClassifier
pipe = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('model', RandomForestClassifier(n_estimators=50,criterion='entropy'))])
model = pipe.fit(x_train, y_train)
prediction = model.predict(x_test)
print("accuracy:{} %".format(round(accuracy_score(y_test, prediction)*100,2)))
cm = metrics.confusion_matrix(y_test,prediction)
plot_confusion_matrix(cm,classes=['fake','true'])
