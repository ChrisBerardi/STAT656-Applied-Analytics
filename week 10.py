"""
Author: Chris Berardi
Solution to STAT656 Week 10 Assigment, Spring 2017
Basic Text Clustering
"""

import pandas as pd
import string
import nltk
import numpy as np
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
#for regression
from Class_replace_impute_encode import ReplaceImputeEncode
from Class_regression import linreg
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# my_analyzer replaces both the preprocessor and tokenizer
# it also replaces stop word removal and ngram constructions

def my_analyzer(s):
    # Synonym List
    syns = {'veh': 'vehicle', 'car': 'vehicle', 'chev':'cheverolet', \
              'chevy':'cheverolet', 'air bag': 'airbag', \
              'seat belt':'seatbelt', "n't":'not', 'to30':'to 30', \
              'wont':'would not', 'cant':'can not', 'cannot':'can not', \
              'couldnt':'could not', 'shouldnt':'should not', \
              'wouldnt':'would not', }
    
    # Preprocess String s
    s = s.lower()
    s = s.replace(',', '. ')
    # Tokenize 
    tokens = word_tokenize(s)
    tokens = [word.replace(',','') for word in tokens ]
    tokens = [word for word in tokens if ('*' not in word) and \
              ("''" != word) and ("``" != word) and \
              (word!='description') and (word !='dtype') \
              and (word != 'object') and (word!="'s")]
    
    # Map synonyms
    for i in range(len(tokens)):
        if tokens[i] in syns:
            tokens[i] = syns[tokens[i]]
            
    # Remove stop words
    punctuation = list(string.punctuation)+['..', '...']
    pronouns = ['i', 'he', 'she', 'it', 'him', 'they', 'we', 'us', 'them']
    stop = stopwords.words('english') + punctuation + pronouns
    filtered_terms = [word for word in tokens if (word not in stop) and \
                  (len(word)>1) and (not word.replace('.','',1).isnumeric()) \
                  and (not word.replace("'",'',2).isnumeric())]
    
    # Lemmatization & Stemming - Stemming with WordNet POS
    # Since lemmatization requires POS need to set POS
    tagged_words = pos_tag(filtered_terms, lang='eng')
    # Stemming with for terms without WordNet POS
    stemmer = SnowballStemmer("english")
    wn_tags = {'N':wn.NOUN, 'J':wn.ADJ, 'V':wn.VERB, 'R':wn.ADV}
    wnl = WordNetLemmatizer()
    stemmed_tokens = []
    for tagged_token in tagged_words:
        term = tagged_token[0]
        pos  = tagged_token[1]
        pos  = pos[0]
        try:
            pos   = wn_tags[pos]
            stemmed_tokens.append(wnl.lemmatize(term, pos=pos))
        except:
            stemmed_tokens.append(stemmer.stem(term))
    return stemmed_tokens

# Further Customization of Stopping and Stemming using NLTK
def my_preprocessor(s):
    #Vectorizer sends one string at a time
    s = s.lower()
    s = s.replace(',', '. ')
    print("preprocessor")
    return(s)
    
def my_tokenizer(s):
    # Tokenize
    print("Tokenizer")
    tokens = word_tokenize(s)
    tokens = [word.replace(',','') for word in tokens ]
    tokens = [word for word in tokens if word.find('*')!=True and \
              word != "''" and word !="``" and word!='description' \
              and word !='dtype']
    return tokens

# Increase Pandas column width to let pandas read large text columns
pd.set_option('max_colwidth', 32000)
# California Cabernet Reviews
file_path = 'C:/Users/Saistout/Desktop/656 Applied Analytics/Python/Week 10 Assignment/'
df = pd.read_excel(file_path+"CaliforniaCabernet.xlsx")

# Setup simple constants
n_docs     = len(df['description'])
n_samples  = n_docs
m_features = None
s_words    = 'english'
ngram = (1,2)

# Setup reviews in list 'discussions'
discussions = []
for i in range(n_samples):
    discussions.append(("%s" %df['description'].iloc[i]))
 
    
# Create Word Frequency by Review Matrix using Custom Analyzer
cv = CountVectorizer(max_df=0.95, min_df=2, max_features=m_features,\
                     analyzer=my_analyzer, ngram_range=ngram)
tf = cv.fit_transform(discussions)

print("\nVectorizer Parameters\n", cv, "\n")


# LDA For Term Frequency x Doc Matrix
n_topics        = 9
max_iter        =  5
learning_offset = 20.
learning_method = 'online'
# LDA for TF-IDF x Doc Matrix
# First Create Term-Frequency/Inverse Doc Frequency by Review Matrix
# This requires constructing Term Freq. x Doc. matrix first
tf_idf = TfidfTransformer()
print("\nTF-IDF Parameters\n", tf_idf.get_params(),"\n")
tf_idf = tf_idf.fit_transform(tf)
# Or you can construct the TF/IDF matrix from the data
tfidf_vect = TfidfVectorizer(max_df=0.95, min_df=2, max_features=m_features,\
                             analyzer=my_analyzer, ngram_range=ngram)
tf_idf = tfidf_vect.fit_transform(discussions)
print("\nTF_IDF Vectorizer Parameters\n", tfidf_vect, "\n")

lda = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter,\
                                learning_method=learning_method, \
                                learning_offset=learning_offset, \
                                random_state=12345)
lda.fit_transform(tf_idf)
print('{:.<22s}{:>6d}'.format("Number of Reviews", tf.shape[0]))
print('{:.<22s}{:>6d}'.format("Number of Terms",     tf.shape[1]))
print("\nTopics Identified using LDA with TF_IDF")
tf_features = cv.get_feature_names()
max_words = 15
desc = []
for topic_idx, topic in enumerate(lda.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([tf_features[i]
                             for i in topic.argsort()[:-max_words - 1:-1]])
        print(message)
        print()
        desc.append([tf_features[i] for i in topic.argsort()[:-max_words - 1:-1]])
        
#Extract which topic each review belongs to
topics = pd.DataFrame(lda.fit_transform(tf_idf))
clusters = pd.DataFrame(topics.idxmax(axis=1))
col=['year','points','Region','price']
clus = pd.concat([clusters,df[col]], axis=1, ignore_index=True)
#rename the columns
clus.columns = ["Cluster","Year","Score","Region","Price"]

#Create a table of the average points and price per cluster, include the 15
#word descriptions in the table
price = []
score = []
cluster = []
mean_table=pd.DataFrame()
for i in range(0,9):
    this_clust = clus[clus['Cluster']==i]
    this_price = this_clust['Price'].mean()
    this_score = this_clust['Score'].mean()
    price.append(this_price)
    score.append(this_score)
    cluster.append(i)
mean_table['Cluster']=cluster
mean_table['Score']=score
mean_table['Price']=price
mean_table['Description']=desc
mean_table

#Create a table of the percent of reviews in each cluster by wine region
p0=[]
p1=[]
p2=[]
p3=[]
p4=[]
p5=[]
p6=[]
p7=[]
p8=[]
regions=['California Other', 'Central Coast','Central Valley', 'Clear Lake',\
         'High Valley', 'Lake County','Mendocino County','Mendocino Ridge',\
         'Mendocino/Lake Counties', 'Napa','Napa-Sonoma','North Coast',\
         'Red Hills Lake County','Redwood Valley','Sierra Foothills','Sonoma',\
         'South Coast']
pRegion=pd.DataFrame()
for name in regions:
    this_region = clus[clus['Region']==name]
    n=[]
    total=0
    for i in range(0,9):
        this_clus=this_region[this_region['Cluster']==i]
        n.append(this_clus.shape[0])
    total=sum(n)
    p0.append(n[0]/total)
    p1.append(n[1]/total)
    p2.append(n[2]/total)
    p3.append(n[3]/total)
    p4.append(n[4]/total)
    p5.append(n[5]/total)
    p6.append(n[6]/total)
    p7.append(n[7]/total)
    p8.append(n[8]/total)
pRegion['Region']=regions
pRegion['P0']=p0
pRegion['P1']=p1
pRegion['P2']=p2
pRegion['P3']=p3
pRegion['P4']=p4
pRegion['P5']=p5
pRegion['P6']=p6
pRegion['P7']=p7
pRegion['P8']=p8

#Fit a linear regression model to model wine price use a 70/30 train test split
#Since the regularization parameter C only exists for logistic regression
attribute_map_clus = {
        'Score'   :[0,(80,100),[0,0]],
        'Year'    :[0,(1985,2016),[0,0]],
        'Region'  :[2,('California Other', 'Central Coast','Central Valley', \
                     'Clear Lake','High Valley', 'Lake County',\
                     'Mendocino County','Mendocino Ridge',\
                     'Mendocino/Lake Counties', 'Napa','Napa-Sonoma',\
                     'North Coast','Red Hills Lake County','Redwood Valley',\
                     'Sierra Foothills','Sonoma','South Coast'),[0,0]],
        'Cluster' :[2,(0,1,2,3,4,5,6,7,8),[0,0]],
        'Price'   :[0,(0,625),[0,0]]
}
varlist = ['Price']

rie_clus = ReplaceImputeEncode(data_map=attribute_map_clus, \
                               nominal_encoding='one-hot', 
                          interval_scale = None, drop=True, display=False)
encoded_df_clus = rie_clus.fit_transform(clus)

X_clus = encoded_df_clus.drop(varlist, axis=1)
y_clus = encoded_df_clus[varlist]
X_train, X_valid, y_train, y_valid= \
train_test_split(X_clus,y_clus,test_size = 0.3, random_state=7)

np_y_train = np.ravel(y_train)
np_y_valid = np.ravel(y_valid)


reg = LinearRegression()
reg.fit(X_train,np_y_train)

linreg.display_coef(reg,X_train,y_train,X_train.columns)
linreg.display_split_metrics(reg,X_train,y_train,X_valid,y_valid)