#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import string
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import plotly.graph_objects as go
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


dataset = pd.read_csv('news_dataset.csv')


# # Dataset Analysis

# In[3]:


dataset.head(5)


# In[4]:


dataset.tail(5)


# In[5]:


dataset.columns


# In[6]:


dataset.drop(['Unnamed: 0','title'],axis = 1,inplace = True)


# In[7]:


dataset.head(5)


# In[8]:


dataset.tail(5)


# In[9]:


dataset.info()


# In[10]:


dataset.describe()


# In[11]:


dataset.isnull().sum()


# In[12]:


dataset.dropna(axis = 0,inplace = True)


# In[13]:


dataset.isnull().sum()


# In[14]:


dataset.info()


# In[15]:


real_news_count = (dataset['label'] == 1).sum()


# In[16]:


real_news_count


# In[17]:


fake_news_count = (dataset['label'] == 0).sum()


# In[18]:


fake_news_count


# In[19]:


37067+35028


# In[20]:


labels = ['Real','fake']
values = [37067,35028]

# pull is given as a fraction of the pie radius
fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.show()


# # Text Data Preprocessing

# ### Convert the data into lowercase

# In[21]:


dataset['text'] = dataset['text'].str.lower()


# ### Removal Of the Punctuations

# In[22]:


string.punctuation


# In[23]:


def punctuation_removal(text):
    punctuations = string.punctuation
    return text.translate(str.maketrans('','',punctuations))


# In[24]:


dataset['text'] = dataset['text'].apply(lambda x: punctuation_removal(x))


# In[25]:


dataset['text'][0]


# ### Removal Of StopWords

# In[26]:


import nltk 
nltk.download("stopwords")


# In[27]:


Stopwords  = set(stopwords.words('english'))
def removal_of_words(text):
    return " ".join([word for word in text.split() if word not in Stopwords])


# In[28]:


" , ".join(Stopwords)


# In[29]:


dataset['text'] = dataset['text'].apply(lambda x: removal_of_words(x))


# In[30]:


dataset.head(5)


# In[ ]:





# ### Stemming 

# In[31]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem_words(text):
    return " ".join([ps.stem(word) for word in text.split()])


# In[32]:


dataset['text'] = dataset['text'].apply(lambda x: stem_words(x))


# In[33]:


dataset.head(5)


# In[34]:


def clean_text(text):
    text = re.sub(r'\[.*?\]|https?://\S+|www\.\S+|<.*?>+|\n|\w*\d\w*', '', text)
    return text


# In[35]:


dataset['text'] = dataset['text'].apply(lambda x: clean_text(x))


# In[36]:


dataset.sample(frac=1).head(10)


# # Model Building

# In[37]:


X = dataset['text']
y = dataset['label']


# In[38]:


X


# In[39]:


y


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)


# In[41]:


X_train


# In[42]:


X_test


# In[43]:


y_train


# In[44]:


y_test


# In[45]:


vect = TfidfVectorizer()
X_train = vect.fit_transform(X_train)
X_test = vect.transform(X_test)


# In[46]:


X_train.shape


# In[47]:


X_test.shape


# In[48]:


classifier = LogisticRegression()

classifier.fit(X_train,y_train)


# In[49]:


y_pred_LG = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_LG)
print('Accuracy:', accuracy)


# In[50]:


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)


# In[51]:


cf_matrix = confusion_matrix(y_test, y_pred_LG)
labels = ['True Neg','False Pos','False Neg','True Pos']
categories = ['Fake', 'Real']
make_confusion_matrix(cf_matrix, 
                      group_names=labels,
                      categories=categories, 
                      cmap='binary')


# In[52]:


pickle.dump(vect,open('vector.pkl','wb'))


# In[53]:


pickle.dump(classifier,open('classifier.pkl','wb'))


# In[54]:


vector_form = pickle.load(open('vector.pkl','rb'))


# In[55]:


classifier_LG = pickle.load(open('classifier.pkl','rb'))


# # Function To Predict the News reliablity

# In[56]:


def predict_reliability(input_news):
    news = input_news.lower()
    news = punctuation_removal(input_news)
    news = removal_of_words(input_news)
    news = stem_words(input_news)
    news = clean_text(news)
    vector_form1 = vector_form.transform([news])
    prediction  = classifier_LG.predict(vector_form1)
    return prediction[0]


# In[57]:


'''PHNOM PENH (Reuters) - Cambodia s government has raised the possibility that the main opposition party could be ruled out of elections if it does not replace its leader, Kem Sokha, who has been charged with treason. The opposition Cambodia National Rescue Party (CNRP) has said it will not replace its leader and the comments reinforced its fears that Prime Minister Hun Sen plans to cripple it before next year s elections. The arrest of Kem Sokha on Sunday drew Western condemnation and marked an escalation in a crackdown on critics of Hun Sen, who has ruled for 30 years and could face possibly his toughest electoral challenge from the CNRP next year.  They have to appoint an acting president,  government spokesman Phay Siphan told Reuters on Tuesday.  If they don t comply with the law, they will not exist and have no right to political activity... It s their choice, not my choice.  Kem Sokha s daughter, Kem Monovithya, who is also a party official, said the party would not appoint a new leader. Kem Sokha was only named in February after his predecessor resigned in fear the party would be banned if he stayed on.  The ruling party can drop their divide-and-conquer plan now,  she said. Opposition officials accuse Hun Sen of trying to weaken or destroy the party ahead of the election, after it did well in June local elections, in which it nonetheless came well behind Hun Sen s Cambodia People s Party. Hun Sen, one of Asia s longest serving rulers, said on Wednesday there could be more arrests after  the act of treason  and it had reinforced the need for him to stay in office.  I ve decided to continue my work - not less than 10 years more,  he told garment factory workers, jabbing his finger in the air for emphasis. Kem Sokha became leader after opposition veteran Sam Rainsy resigned because of a new law that forbids any party having a leader who is found guilty of a crime. Sam Rainsy fled into exile to avoid a defamation conviction he says was political. Cambodian law says a political party has 90 days to replace a president if he or she dies, resigns or is convicted of an offence. Western countries have condemned the arrest of Kem Sokha and a crackdown on critics of Hun Sen, including independent media.  We don t care about people outside,  Phay Siphan said.  We care about our national security. We don t belong to anyone.  China, Hun Sen s close ally, has voiced support for Cambodia on steps to ensure its security. Kem Sokha was formally charged with treason on Tuesday. His lawyers have dismissed the evidence presented against him so far - a video publicly available since 2013 - in which he tells supporters he is getting support and advice from Americans for the campaign to win elections.      The government and the ruling CPP have manufactured these treason charges against Kem Sokha for political purposes, aiming to try and knock the political opposition out of the ring before the 2018 electoral contest even begins,  said Phil Robertson, deputy Asia director of New York-based Human Rights Watch. '''


# In[60]:


predict_reliability(''' Now, most of the demonstrators gathered last night were exercising their constitutional and protected right to peaceful protest in order to raise issues and create change.    Loretta Lynch aka Eric Holder in a skirt

 ''')


# In[61]:


print('hellow word')

