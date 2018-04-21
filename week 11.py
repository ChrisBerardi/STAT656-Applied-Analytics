"""
Author: Chris Berardi
Solution to STAT656 Week 11 Assigment, Spring 2017
Advanced Text Analytics
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
from Class_regression import logreg
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
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
              'wouldnt':'would not', 'air':'airbag', 'bag':'airbag'}
    
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
    # For adding extra stop words
    other = ['own','go','get','seem','say','would','regard','report','involve'\
             ,'do','anoth','consumer',"'ve",'happen','try','either','come',]
    stop = stopwords.words('english') + punctuation + pronouns + other
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
file_path = 'C:/Users/Saistout/Desktop/656 Applied Analytics/Python/Week 11 Assignment/'
df = pd.read_excel(file_path+"GMC_Complaints.xlsx")

# Setup simple constants
n_docs     = len(df['description'])
n_samples  = n_docs
m_features = 1000
s_words    = 'english'
ngram = (1,2)
max_df=0.8

# Setup reviews in list 'discussions'
discussions = []
for i in range(n_samples):
    discussions.append(("%s" %df['description'].iloc[i]))
 
    
# Create Word Frequency by Review Matrix using Custom Analyzer
cv = CountVectorizer(max_df=max_df, min_df=2, max_features=m_features,\
                     analyzer=my_analyzer, ngram_range=ngram)
tf = cv.fit_transform(discussions)

# LDA For Term Frequency x Doc Matrix
n_topics        = 8
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
tfidf_vect = TfidfVectorizer(max_df=max_df, min_df=2, max_features=m_features,\
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
        
#Extract topic probablities
topics = pd.DataFrame(lda.fit_transform(tf_idf))
preds = ['Year', 'make','model','crashed','abs','mileage']
#Create new dataframe for regression with desired columns from original data
#and the topic probabilities
reg_df = pd.concat([df[preds],topics], axis=1, ignore_index=True)
reg_df.columns = ['Year', 'make','model','crashed','abs','mileage',\
                  '0','1','2','3','4','5','6','7']

#Fit a logistics regression model to model wine price use a 70/30 split
#Do 10-fold cross validation to determine optimum value for C-regularization
attribute_map = {
        'Year'    :[0,(2003,2011),[0,0]],
        'make'    :[2,('CHEVROLET','PONTIAC','SATURN'),[0,0]],
        'model'   :[2,('COBALT','G5','HHR','ION','SKY','SOLSTICE'),[0,0]],
        'crashed' :[1,('N','Y'),[0,0]],
        'abs'     :[1,('N','Y'),[0,0]],
        'mileage' :[0,(0,200000),[0,0]],
        '0'       :[0,(0,1),[0,0]],
        '1'       :[0,(0,1),[0,0]],
        '2'       :[0,(0,1),[0,0]],
        '3'       :[0,(0,1),[0,0]],
        '4'       :[0,(0,1),[0,0]],
        '5'       :[0,(0,1),[0,0]],
        '6'       :[0,(0,1),[0,0]],
        '7'       :[0,(0,1),[0,0]],
}
varlist = ['crashed']
rie = ReplaceImputeEncode(data_map=attribute_map, \
                               nominal_encoding='one-hot', 
                          interval_scale = None, drop=True, display=False)
encoded_df = rie.fit_transform(reg_df)
X = encoded_df.drop(varlist, axis=1)
y = encoded_df[varlist]
np_y=np.ravel(y)

#10 fold-cross validation to find optimum regularization value
max_f1 = 0
C_list=[.1,1,10,100]
score_list = ['accuracy', 'recall', 'precision', 'f1']
for c in C_list:
    print("\nRegularization Parameter: ", c)
    lgr = LogisticRegression(C=c, tol=1e-8, max_iter=1000)
    lgr.fit(X, np_y)
    scores = cross_validate(lgr, X, np_y,\
                            scoring=score_list, return_train_score=False, \
                            cv=10)
    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    for s in score_list:
        var = "test_"+s
        mean = scores[var].mean()
        std  = scores[var].std()
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
        if mean > max_f1:
            max_f1 = mean
            best_predictor  = c
print("\nBest based on F1-Score")
print("Best Regularization Parameter = ", best_predictor)

#Use best regularization parameter along with a 70/30 split
X_train, X_valid, y_train, y_valid= \
train_test_split(X,y,test_size = 0.3, random_state=7)

np_y_train = np.ravel(y_train)
np_y_valid = np.ravel(y_valid)

reg = LogisticRegression(C=best_predictor, tol=1e-8, max_iter=1000)
reg.fit(X_train,np_y_train)

logreg.display_coef(reg,17,1,X_train.columns)
logreg.display_binary_split_metrics(reg,X_train,np_y_train,X_valid,np_y_valid)