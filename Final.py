"""
Author: Chris Berardi
Solution to STAT656 Take Home Final Exam
Using text analytics and sentiment analysis to improve predictive modeling.
Implments word clouds as well as web scraping. 
"""

import pandas as pd
import numpy as np
import string

# Text Topic Imports
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation

# Predictive Modeling Imports
from Class_tree import DecisionTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from Class_FNN import NeuralNetwork
from sklearn.neural_network import MLPClassifier

from Class_replace_impute_encode import ReplaceImputeEncode
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

#Word Cloud Imports
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import random

#Web Scraping Imports
import re
import requests
from newsapi import NewsApiClient # Needed for using API Feed
from time import time

# Function for calculating loss and confusion matrix
def loss_cal(y, y_predict, fp_cost, fn_cost, display=True):
    loss     = [0, 0]       #False Neg Cost, False Pos Cost
    conf_mat = [0, 0, 0, 0] #tn, fp, fn, tp
    for j in range(len(y)):
        if y[j]==0:
            if y_predict[j]==0:
                conf_mat[0] += 1 #True Negative
            else:
                conf_mat[1] += 1 #False Positive
                loss[1] += fp_cost[j]
        else:
            if y_predict[j]==1:
                conf_mat[3] += 1 #True Positive
            else:
                conf_mat[2] += 1 #False Negative
                loss[0] += fn_cost[j]
    if display:
        fn_loss = loss[0]
        fp_loss = loss[1]
        total_loss = fn_loss + fp_loss
        misc    = conf_mat[1] + conf_mat[2]
        misc    = misc/len(y)
        print("{:.<23s}{:10.4f}".format("Misclassification Rate", misc))
        print("{:.<23s}{:10.0f}".format("False Negative Cost", fn_loss))
        print("{:.<23s}{:10.0f}".format("False Positive Cost", fp_loss))
        print("{:.<23s}{:10.0f}".format("Total Loss", total_loss))
    return loss, conf_mat

# my_analyzer replaces both the preprocessor and tokenizer
# it also replaces stop word removal and ngram constructions
def my_analyzer(s):
    # Synonym List
    syns = {"n't":'not', 'wont':'would not', 'cant':'can not', \
            'cannot':'can not', 'couldnt':'could not', \
            'shouldnt':'should not', 'wouldnt':'would not',\
            'vehicle':'car', 'contact':'driver', 'air':'bag','bag':'airbag',\
            'issue':'problem', 'SR':'Civic'}
    
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
    other = ['car','driver','would','go','year/make/model','get','own'\
             'www.carcomplaints.com','say','come','take','tell','nhtsa','find'\
             , 'sinc']
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
    # Preprocess String s
    s = s.lower()
    # Replace special characters with spaces
    s = s.replace('-', ' ')
    s = s.replace('_', ' ')
    s = s.replace(',', '. ')
    # Replace not contraction with not
    s = s.replace("'nt", " not")
    s = s.replace("n't", " not")
    return s
    
def my_tokenizer(s):
    # Tokenize
    print("Tokenizer")
    tokens = word_tokenize(s)
    tokens = [word.replace(',','') for word in tokens ]
    tokens = [word for word in tokens if word.find('*')!=True and \
              word != "''" and word !="``" and word!='description' \
              and word !='dtype']
    return tokens

def display_topics(topic_vectorizer, terms, n_terms=15, word_cloud=False, mask=None):
    for topic_idx, topic in enumerate(topic_vectorizer):
        message = "Topic #%d: " %(topic_idx+1)
        print(message)
        abs_topic = abs(topic)
        topic_terms_sorted = \
            [[terms[i], topic[i]] \
             for i in abs_topic.argsort()[:-n_terms - 1:-1]]
        k = 5
        n = int(n_terms/k)
        m = n_terms - k*n
        for j in range(n):
            l = k*j
            message = ''
            for i in range(k):
                if topic_terms_sorted[i+l][1]>0:
                    word = "+"+topic_terms_sorted[i+l][0]
                else:
                    word = "-"+topic_terms_sorted[i+l][0]
                message += '{:<15s}'.format(word)
            print(message)
        if m> 0:
            l = k*n
            message = ''
            for i in range(m):
                if topic_terms_sorted[i+l][1]>0:
                    word = "+"+topic_terms_sorted[i+l][0]
                else:
                    word = "-"+topic_terms_sorted[i+l][0]
                message += '{:<15s}'.format(word)
            print(message)
        print("")
        if word_cloud:
            wcloud = WordCloud(background_color="maroon", \
                    mask=mask, max_words=30, \
                    max_font_size=40, prefer_horizontal=0.9, \
                    min_font_size=10, relative_scaling=0.5, \
                    width=400, height=200, scale=1, margin=10, random_state=12345)
            topic_cloud = {}
            for i in range(n_terms):
                topic_cloud[topic_terms_sorted[i][0]] = \
                    topic_terms_sorted[i][1]
            # Show Word Cloud based term Frequencies (unweighted)
            wcloud.generate_from_frequencies(topic_cloud)
            plt.imshow(wcloud.recolor(color_func=shades_of_grey, random_state=12345), \
                       interpolation="bilinear")
            plt.axis("off")
            plt.figure()
            plt.show()
    return

def term_dic(tf, terms, scores=None):
    td = {}
    for i in range(tf.shape[0]):
        # Iterate over the terms with nonzero scores
        term_list = tf[i].nonzero()[1]
        if len(term_list)>0:
            if scores==None:
                for t in np.nditer(term_list):
                    if td.get(terms[t]) == None:
                        td[terms[t]] = tf[i,t]
                    else:
                        td[terms[t]] += tf[i,t]
            else:
                for t in np.nditer(term_list):
                    score = scores.get(terms[t])
                    if score != None:
                        # Found Sentiment Word
                        score_weight = abs(scores[terms[t]])
                        if td.get(terms[t]) == None:
                            td[terms[t]] = tf[i,t] * score_weight
                        else:
                            td[terms[t]] += tf[i,t] * score_weight
    return td

def shades_of_grey(word, font_size, position, orientation, random_state=None, \
                   **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60,1000)

def word_cloud_dic(td, mask=None, bg_color="maroon", max_words=30):
    wcloud = WordCloud(background_color=bg_color, \
                       mask=mask, max_words=max_words, \
                       max_font_size=40, prefer_horizontal=0.9, \
                       min_font_size=10, relative_scaling=0.5, \
                       width=400, height=200, scale=1, margin=10, random_state=12345)
    # Show Word Cloud based term Frequencies (unweighted)
    wcloud.generate_from_frequencies(td)
    plt.imshow( \
               wcloud.recolor(color_func=shades_of_grey,random_state=12345),\
               interpolation="bilinear")
    plt.axis("off")
    plt.figure()
    plt.show()
    return

def clean_html(html):
    # First we remove inline JavaScript/CSS:
    pg = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
    # Then we remove html comments. This has to be done before removing regular
    # tags since comments can contain '>' characters.
    pg = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", pg)
    # Next we can remove the remaining tags:
    pg = re.sub(r"(?s)<.*?>", " ", pg)
    # Finally, we deal with whitespace
    pg = re.sub(r"&nbsp;", " ", pg)
    pg = re.sub(r"&rsquo;", "'", pg)
    pg = re.sub(r"&ldquo;", '"', pg)
    pg = re.sub(r"&rdquo;", '"', pg)
    pg = re.sub(r"\n", " ", pg)
    pg = re.sub(r"\t", " ", pg)
    pg = re.sub(r" ", " ", pg)
    pg = re.sub(r" ", " ", pg)
    pg = re.sub(r" ", " ", pg)
    return pg.strip()

def newsapi_get_urls(search_words, agency_urls):
    if len(search_words)==0 or agency_urls==None:
        return None
    print("Searching agencies for pages containing:", search_words)
    # Changed to my API key
    api = NewsApiClient(api_key='6feb93623b2846df97daad5a62df6690')
    api_urls = []
    # Iterate over agencies and search words to pull more url's
    # Limited to 1,000 requests/day - Likely to be exceeded
    for agency in agency_urls:
        domain = agency_urls[agency].replace("http://", "")
        print(agency, domain)
        for word in search_words:
            # Get articles with q= in them, Limits to 20 URLs
            try:
                articles = api.get_everything(q=word, language='en',\
                                              sources=agency, domains=domain)
            except:
                print("--->Unable to pull news from:", agency, "for", word)
                continue
            # Pull the URL from these articles (limited to 20)
            d = articles['articles']
            for i in range(len(d)):
                url = d[i]['url']
                api_urls.append([agency, word, url])
    df_urls = pd.DataFrame(api_urls, columns=['agency', 'word', 'url'])
    n_total = len(df_urls)
    # Remove duplicates
    df_urls = df_urls.drop_duplicates('url')
    n_unique = len(df_urls)
    print("\nFound a total of", n_total, " URLs, of which", n_unique,\
          " were unique.")
    return df_urls

def request_pages(df_urls):
    web_pages = []
    for i in range(len(df_urls)):
        u = df_urls.iloc[i]
        url = u[2]
        short_url = url[0:50]
        short_url = short_url.replace("https//", "")
        short_url = short_url.replace("http//", "")
        n = 0
        # Allow for a maximum of 5 download failures
        stop_sec=3 # Initial max wait time in seconds
        while n<3:
            try:
                r = requests.get(url, timeout=(stop_sec))
                if r.status_code == 408:
                    print("-->HTML ERROR 408", short_url)
                    raise ValueError()
                if r.status_code == 200:
                    print("Obtained: "+short_url)
                else:
                    print("-->Web page: "+short_url+" status code:", \
                          r.status_code)
                n=99
                continue # Skip this page
            except:
                n += 1
                # Timeout waiting for download
                t0 = time()
                tlapse = 0
                print("Waiting", stop_sec, "sec")
                while tlapse<stop_sec:
                    tlapse = time()-t0
        if n != 99:
            # download failed skip this page
            continue
        # Page obtained successfully
        html_page = r.text
        page_text = clean_html(html_page)
        web_pages.append([url, page_text])
    df_www = pd.DataFrame(web_pages, columns=['url', 'text'])
    n_total = len(df_urls)
    # Remove duplicates
    df_www = df_www.drop_duplicates('url')
    n_unique = len(df_urls)
    print("Found a total of", n_total, " web pages, of which", n_unique,\
          " were unique.")
    return df_www

# Increase column width to let pandas read large text columns
pd.set_option('max_colwidth', 32000)
# Read N=13,575 California Cabernet Savignon Reviews
file_path = 'C:/Users/Saistout/Desktop/656 Applied Analytics/Python/Final Exam/'
df = pd.read_excel(file_path+"HondaComplaints.xlsx")
sw = pd.read_excel(file_path+"afinn_sentiment_words(1).xlsx")
#Drop unwanted NhtsaID and State columns
df = df.drop(['NhtsaID', 'State'],axis=1)

#Rare event problem
sum(df['crash']=='Y')/len(df['crash'])

# Setup program constants and reviews
n_description = len(df['description'])
n_topics = 7 # number of topics
stopw = set(STOPWORDS) # Word Cloud Stop Words
#stopw.add()  add "jul" to stopword list

# Create Word Frequency by Review Matrix using Custom Analyzer
# max_df is a stop limit for terms that have more than this
# proportion of documents with the term (max_df - don't ignore any terms)
cv = CountVectorizer(max_df=0.7, min_df=4, max_features=None,\
                     analyzer=my_analyzer)
tf = cv.fit_transform(df['description'])
terms = cv.get_feature_names()
print('{:.<22s}{:>6d}'.format("Number of Reviews", n_description))
print('{:.<22s}{:>6d}'.format("Number of Terms", len(terms)))
# Term Dictionary with Terms as Keys and frequency as Values
td = term_dic(tf, terms)
print("The Corpus contains a total of ", len(td), " unique terms.")
print("The total number of terms in the Corpus is", sum(td.values()))
term_sums = tf.sum(axis=0)
term_counts = []
for i in range(len(terms)):
    term_counts.append([terms[i], term_sums[0,i]])
def sortSecond(e):
    return e[1]
term_counts.sort(key=sortSecond, reverse=True)
print("\nTerms with Highest Frequency:")
for i in range(10):
    print('{:<15s}{:>5d}'.format(term_counts[i][0], term_counts[i][1]))
    
# Construct the TF/IDF matrix from the data
print("\nConducting Term/Frequency Matrix using TF-IDF")
# Default for norm is 'l2', use norm=None to supress
tfidf_vect = TfidfTransformer(norm=None, use_idf=True) #set norm=None
# tf matrix is (n_reviews)x(m_features
tf = tfidf_vect.fit_transform(tf)
term_idf_sums = tf.sum(axis=0)
term_idf_scores = []
for i in range(len(terms)):
    term_idf_scores.append([terms[i], term_idf_sums[0,i]])
print("The Term/Frequency matrix has", tf.shape[0], " rows, and",\
      tf.shape[1], " columns.")
print("The Term list has", len(terms), " terms.")
term_idf_scores.sort(key=sortSecond, reverse=True)
print("\nTerms with Highest TF-IDF Scores:")
for i in range(10):
    j = i
    print('{:<15s}{:>8.2f}'.format(term_idf_scores[j][0], \
          term_idf_scores[j][1]))
    
# LDA Analysis, SVD Analysis was found to be inferior for this application
uv = LatentDirichletAllocation(n_components=n_topics, \
                               learning_method='online', random_state=12345)
U = uv.fit_transform(tf)

print("\n********** GENERATED TOPICS **********")
display_topics(uv.components_, terms, n_terms=15)

# Store topic selection for each doc in topics[]
topics = [0] * n_description
for i in range(n_description):
    max = abs(U[i][0])
    topics[i] = 0
    for j in range(n_topics):
        x = abs(U[i][j])
        if x > max:
            max = x
            topics[i] = j
            
U_rev_scores = []
for i in range(n_description):
    u = [0] * (n_topics+1)
    u[0] = topics[i]
    for j in range(n_topics):
        u[j+1] = U[i][j]
        U_rev_scores.append(u)
rev_scores = U_rev_scores

# Integrate Topic Scores into Main Data Frame (df)
cols = ["Topic"]
for i in range(n_topics):
    s = "T"+str(i+1)
    cols.append(s)
df_rev = pd.DataFrame.from_records(rev_scores, columns=cols)
df = df.join(df_rev)

print(" TOPIC DISTRIBUTION")
print('{:<6s}{:>4s}{:>12s}'.format("TOPIC", "N", "PERCENT"))
print("----------------------")
topic_counts = df['Topic'].value_counts(sort=False)
for i in range(len(topic_counts)):
    percent = 100*topic_counts[i]/n_description
    print('{:>3d}{:>8d}{:>9.1f}%'.format((i+1), topic_counts[i], percent))
    
# Setup Sentiment dictionary
sentiment_dic = {}
for i in range(len(sw)):
    sentiment_dic[sw.iloc[i][0]] = sw.iloc[i][1]
    
#Define the preprocessor for use with sentiment analysis
cv = CountVectorizer(max_df=1.0, min_df=1, max_features=None, \
                     preprocessor=my_preprocessor, ngram_range=(1,2))
tf = cv.fit_transform(df['description'])
s_terms = cv.get_feature_names()
n_descriptions = tf.shape[0]
n_terms = tf.shape[1]

#Caculate semtiment for each document and for the whole corpus
min_sentiment = +5
max_sentiment = -5
avg_sentiment, min, max = 0,0,0
min_list, max_list = [],[]
sentiment_score = [0]*n_descriptions
for i in range(n_descriptions):
    # Iterate over the terms with nonzero scores
    n_sw = 0
    term_list = tf[i].nonzero()[1]
    if len(term_list)>0:
        for t in np.nditer(term_list):
            score = sentiment_dic.get(s_terms[t])
            if score != None:
                sentiment_score[i] += score * tf[i,t]
                n_sw += tf[i,t]
    if n_sw>0:
        sentiment_score[i] = sentiment_score[i]/n_sw
    if sentiment_score[i]==max_sentiment and n_sw>3:
        max_list.append(i)
    if sentiment_score[i]>max_sentiment and n_sw>3:
        max_sentiment=sentiment_score[i]
        max = i
        max_list = [i]
    if sentiment_score[i]==min_sentiment and n_sw>3:
        min_list.append(i)
    if sentiment_score[i]<min_sentiment and n_sw>3:
        min_sentiment=sentiment_score[i]
        min = i
        min_list = [i]
        avg_sentiment += sentiment_score[i]
    avg_sentiment = avg_sentiment/n_descriptions
print("\nCorpus Average Sentiment:{:>5.2f} ".format(avg_sentiment))
print("\nMost Negative Reviews with 4 or more Sentiment Words:")
for i in range(len(min_list)):
    print("{:<s}{:>5d}{:<s}{:>5.2f}".format(" Review ", min_list[i], \
          " Sentiment is ", min_sentiment))
print("\nMost Positive Reviews with 4 or more Sentiment Words:")
for i in range(len(max_list)):
    print("{:<s}{:>5d}{:<s}{:>5.2f}".format(" Review ", max_list[i], \
          " Sentiment is ", max_sentiment))

#Add sentiment score into the df
sens=pd.DataFrame({'Sentiment':sentiment_score})
df=pd.concat([df,sens],axis=1)
#Print out averages by topic and make
print(df.groupby(['Make'])['Sentiment'].mean(), "\n")
print(df.groupby(['Topic'])['Sentiment'].mean(), "\n")

#Modeling of crashes
attribute_map = {
    'Year'       :[2,(2001, 2002,2003),[0,0]],
    'Make'       :[2,('HONDA', 'ACURA'),[0,0]],
    'Model'      :[2,('TL', 'ODYSSEY', 'CR-V', 'CL', 'CIVIC', 'ACCORD'),[0,0]],
    'description':[3,(''),[0,0]],
    'crash'      :[1,('N', 'Y'),[0,0]],
    'cruise'     :[1,('N', 'Y'),[0,0]],
    'abs'        :[1,('N', 'Y'),[0,0]],
    'mph'        :[0,(0, 80),[0,0]],
    'mileage'    :[0,(0, 200000),[0,0]],
    'Topic'      :[2,(0,1,2,3,4,5,6),[0,0]],
    'Sentiment'  :[0,(-5,5),[0,0]],
    'T1':[0,(-1e+8,1e+8),[0,0]],
    'T2':[0,(-1e+8,1e+8),[0,0]],
    'T3':[0,(-1e+8,1e+8),[0,0]],
    'T4':[0,(-1e+8,1e+8),[0,0]],
    'T5':[0,(-1e+8,1e+8),[0,0]],
    'T6':[0,(-1e+8,1e+8),[0,0]],
    'T7':[0,(-1e+8,1e+8),[0,0]]
    }

varlist = ['crash']

#Neural encoding
rie_n = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', \
                          interval_scale = 'std', drop=False, display=False)
encoded_df_n = rie_n.fit_transform(df)
X_n = encoded_df_n.drop(varlist, axis=1)
y_n = encoded_df_n[varlist]
np_y_n = np.ravel(y_n) #convert dataframe column to flat array

#Tree encoding
rie_t = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot', \
                           drop=False,interval_scale = None, display=True)
encoded_df_t = rie_t.fit_transform(df)
X_t = encoded_df_t.drop(varlist, axis=1)
y_t = encoded_df_t[varlist]
np_y_t = np.ravel(y_t) #convert dataframe column to flat array
col = list(encoded_df_t) #Needed for variable importance list

#Tree Methods
#Start with Random Forest
# Cross-Validation
estimators_list   = [10,20,30,50]
max_features_list = ['auto', .6, .7, .8]
score_list = ['accuracy', 'recall', 'precision', 'f1']
max_f1 = 0
for e in estimators_list:
    for f in max_features_list:
        print("\nNumber of Trees: ", e, " Max_features: ", f)
        rfc = RandomForestClassifier(n_estimators=e, criterion="gini", \
                    max_depth=100, min_samples_split=2, \
                    min_samples_leaf=1, max_features=f, \
                    n_jobs=1, bootstrap=True, random_state=12345)
        rfc= rfc.fit(X_t, np_y_t)
        scores = cross_validate(rfc, X_t, np_y_t, scoring=score_list, \
                                return_train_score=False, cv=10)
        
        print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
        for s in score_list:
            var = "test_"+s
            mean = scores[var].mean()
            std  = scores[var].std()
            print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
        if mean > max_f1:
            max_f1 = mean
            best_estimator    = e
            best_max_features = f

print("\nBest based on F1-Score")
print("Best Number of Estimators (trees) = ", best_estimator)
print("Best Maximum Features = ", best_max_features)

# Evaluate the random forest with the best configuration
X_train_t, X_validate_t, y_train_t, y_validate_t = \
            train_test_split(X_t, np_y_t,test_size = 0.3, random_state=12345)

rfc = RandomForestClassifier(n_estimators=best_estimator, criterion="gini", \
                    max_depth=100, min_samples_split=2, \
                    min_samples_leaf=1, max_features=best_max_features, \
                    n_jobs=1, bootstrap=True, random_state=12345)
rfc= rfc.fit(X_train_t, y_train_t)
DecisionTree.display_importance(rfc,col)
                
#Decision Tree Models
# Cross Validation
depth_list = [6, 7, 8, 10, 12]
max_f1 = 0
for d in depth_list:
    print("\nMaximum Tree Depth: ", d)
    dtc = DecisionTreeClassifier(max_depth=d, min_samples_leaf=5, \
                                 min_samples_split=5)
    dtc = dtc.fit(X_t,y_t)
    scores = cross_validate(dtc, X_t, y_t, scoring=score_list, \
                            return_train_score=False, cv=10)
    
    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    for s in score_list:
        var = "test_"+s
        mean = scores[var].mean()
        std  = scores[var].std()
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
        if mean > max_f1:
            max_f1 = mean
            best_depth    = d
            
print("\nBest based on F1-Score")
print("Best Depth = ", best_depth)


# Create X and y, numpy arrays
# bad=0 and good=1
y = np.asarray(encoded_df_t['crash']) # The target is not scaled or imputed
X = np.asarray(encoded_df_t.drop('crash',axis=1))
# Evaluate the tree with the best depth
dtc = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=5, min_samples_split=5)
dtc = dtc.fit(X,y)

#Neural Network Models
network_list = [(3),(4),(5),(6)]
max_f1 = 0
for nn in network_list:
    print("\nNetwork: ", nn)
    fnn = MLPClassifier(hidden_layer_sizes=nn, activation='logistic', \
                    solver='lbfgs', max_iter=1000, random_state=12345)
    fnn = fnn.fit(X_n, np_y_n)
# Neural Network Cross-Validation
    mean_score = []
    std_score  = []
    for s in score_list:
        fnn_10 = cross_val_score(fnn, X_n, np_y_n, cv=10, scoring=s)
        mean_score.append(fnn_10.mean())
        std_score.append(fnn_10.std())

    print("{:.<13s}{:>6s}{:>13s}".format("\nMetric", "Mean", "Std. Dev."))
    for i in range(len(score_list)):
        score_name = score_list[i]
        mean       = mean_score[i]
        std        = std_score[i]
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(score_name, mean, std))
        if mean > max_f1:
            max_f1 = mean
            best_list = nn
        
print("\nBest based on F1-Score")
print("Best Network Configuration = ", best_list)

X_train_n, X_validate_n, y_train_n, y_validate_n = \
            train_test_split(X_n,y_n,test_size = 0.3, random_state=7)
np_y_validate_n = np.ravel(y_validate_n)
np_y_train_n = np.ravel(y_train_n)
# Evaluate the network with the best structure
nnc = MLPClassifier(hidden_layer_sizes=best_list, activation='logistic', \
                    solver='lbfgs', max_iter=1000, random_state=12345)
nnc = nnc.fit(X_train_n,np_y_train_n)

#Compare Best Models
print("\nTraining Data\nRandom Selection of 70% of Original Data")
print("\nRandom Forest")
print("\nEstimators (trees) = ", best_estimator)
print("\nMaximum Features = ", best_max_features)
DecisionTree.display_binary_split_metrics(rfc, X_train_t, y_train_t, \
                                              X_validate_t, y_validate_t)
print("\nDecision Tree")
print("\nDepth",best_depth)
DecisionTree.display_binary_split_metrics(dtc, X_train_t, y_train_t, \
                                     X_validate_t, y_validate_t)
print("\nNeural Network")
print("\nNetwork Configuration = ", best_list)
NeuralNetwork.display_binary_split_metrics(nnc, X_train_n, np_y_train_n, \
                                     X_validate_n, np_y_validate_n)

#Word Cloud
# CORPUS WORD CLOUD
print("\nCorpus Word Cloud")
word_cloud_dic(td, mask=None, max_words=100)
#Corpus Sentiment Words
corpus_sentiment = term_dic(tf, s_terms, scores=sentiment_dic)
word_cloud_dic(corpus_sentiment, mask=None, max_words=100)

#Web Scraping
agency_urls = {
        'huffington': 'http://huffingtonpost.com',
        'reuters': 'http://www.reuters.com',
        'cbs-news': 'http://www.cbsnews.com',
        'usa-today': 'http://usatoday.com',
        'cnn': 'http://cnn.com',
        'npr': 'http://www.npr.org',
        'wsj': 'http://wsj.com',
        'fox': 'http://www.foxnews.com',
        'nyt': 'http://nytimes.com',
        'washington-post': 'http://washingtonpost.com',
        'us-news': 'http://www.usnews.com',
        'msn': 'http://msn.com',
        'pbs': 'http://www.pbs.org',
        'nbc-news': 'http://www.nbcnews.com',
        'la-times': 'http://www.latimes.com'
}

#Search for new on Takata
search_words = ['Takata']
df_urls = newsapi_get_urls(search_words, agency_urls)
print("Total Articles:", df_urls.shape[0])

# Download Discovered Pages
df_www = request_pages(df_urls)
df_www.to_excel('df_www.xlsx')

for i in range(df_www.shape[0]):
    short_url = df_www.iloc[i]['url']
    short_url = short_url.replace("https://", "")
    short_url = short_url.replace("http://", "")
    short_url = short_url[0:60]
    page_char = len(df_www.iloc[i]['text'])
    print("{:<60s}{:>10d} Characters".format(short_url, page_char))

#Text Analysis
#Define a new analyzer for the Takata articles to allow for different stop words
def my_analyzer_t(s):
    # Synonym List
    syns = {"n't":'not', 'wont':'would not', 'cant':'can not', \
            'cannot':'can not', 'couldnt':'could not', \
            'shouldnt':'should not', 'wouldnt':'would not'}
    
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
    other = ['apr', 'value=', 'vowg_p.de', 'video']
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

# Setup program constants and reviews
n_text = len(df_www['text'])
n_topics = 2 # number of topics

cv = CountVectorizer(max_df=0.9, min_df=4, max_features=None,\
                     analyzer=my_analyzer_t)
tf = cv.fit_transform(df_www['text'])
terms = cv.get_feature_names()
print('{:.<22s}{:>6d}'.format("Number of Reviews", n_text))
print('{:.<22s}{:>6d}'.format("Number of Terms", len(terms)))
# Term Dictionary with Terms as Keys and frequency as Values
td = term_dic(tf, terms)
print("The Corpus contains a total of ", len(td), " unique terms.")
print("The total number of terms in the Corpus is", sum(td.values()))
term_sums = tf.sum(axis=0)
term_counts = []
for i in range(len(terms)):
    term_counts.append([terms[i], term_sums[0,i]])
def sortSecond(e):
    return e[1]
term_counts.sort(key=sortSecond, reverse=True)
print("\nTerms with Highest Frequency:")
for i in range(10):
    print('{:<15s}{:>5d}'.format(term_counts[i][0], term_counts[i][1]))
    
# Construct the TF/IDF matrix from the data
print("\nConducting Term/Frequency Matrix using TF-IDF")
# Default for norm is 'l2', use norm=None to supress
tfidf_vect = TfidfTransformer(norm=None, use_idf=True) #set norm=None
# tf matrix is (n_reviews)x(m_features
tf = tfidf_vect.fit_transform(tf)
term_idf_sums = tf.sum(axis=0)
term_idf_scores = []
for i in range(len(terms)):
    term_idf_scores.append([terms[i], term_idf_sums[0,i]])
print("The Term/Frequency matrix has", tf.shape[0], " rows, and",\
      tf.shape[1], " columns.")
print("The Term list has", len(terms), " terms.")
term_idf_scores.sort(key=sortSecond, reverse=True)
print("\nTerms with Highest TF-IDF Scores:")
for i in range(10):
    j = i
    print('{:<15s}{:>8.2f}'.format(term_idf_scores[j][0], \
          term_idf_scores[j][1]))
    
# LDA Analysis, SVD Analysis was found to be inferior for this application
uv = LatentDirichletAllocation(n_components=n_topics, \
                               learning_method='online', random_state=12345)
U = uv.fit_transform(tf)

print("\n********** GENERATED TOPICS **********")
display_topics(uv.components_, terms, n_terms=15)
# Store topic selection for each doc in topics[]
topics = [0] * n_text
for i in range(n_text):
    max = abs(U[i][0])
    topics[i] = 0
    for j in range(n_topics):
        x = abs(U[i][j])
        if x > max:
            max = x
            topics[i] = j
            
U_rev_scores = []
for i in range(n_text):
    u = [0] * (n_topics+1)
    u[0] = topics[i]
    for j in range(n_topics):
        u[j+1] = U[i][j]
        U_rev_scores.append(u)
rev_scores = U_rev_scores

# Integrate Topic Scores into Main Data Frame (df)
cols = ["Topic"]
for i in range(n_topics):
    s = "T"+str(i+1)
    cols.append(s)
df_rev = pd.DataFrame.from_records(rev_scores, columns=cols)
df_www = df_www.join(df_rev)

print(" TOPIC DISTRIBUTION")
print('{:<6s}{:>4s}{:>12s}'.format("TOPIC", "N", "PERCENT"))
print("----------------------")
topic_counts = df_www['Topic'].value_counts(sort=False)
for i in range(len(topic_counts)):
    percent = 100*topic_counts[i]/n_text
    print('{:>3d}{:>8d}{:>9.1f}%'.format((i+1), topic_counts[i], percent))

#Sentiment Analysis    
#Define the preprocessor for use with sentiment analysis
cv = CountVectorizer(max_df=1.0, min_df=1, max_features=None, \
                     preprocessor=my_preprocessor, ngram_range=(1,2))
tf = cv.fit_transform(df_www['text'])
s_terms = cv.get_feature_names()
n_descriptions = tf.shape[0]
n_terms = tf.shape[1]

#Caculate semtiment for each document and for the whole corpus
min_sentiment = +5
max_sentiment = -5
avg_sentiment, min, max = 0,0,0
min_list, max_list = [],[]
sentiment_score = [0]*n_descriptions
for i in range(n_descriptions):
    # Iterate over the terms with nonzero scores
    n_sw = 0
    term_list = tf[i].nonzero()[1]
    if len(term_list)>0:
        for t in np.nditer(term_list):
            score = sentiment_dic.get(s_terms[t])
            if score != None:
                sentiment_score[i] += score * tf[i,t]
                n_sw += tf[i,t]
    if n_sw>0:
        sentiment_score[i] = sentiment_score[i]/n_sw
    if sentiment_score[i]==max_sentiment and n_sw>3:
        max_list.append(i)
    if sentiment_score[i]>max_sentiment and n_sw>3:
        max_sentiment=sentiment_score[i]
        max = i
        max_list = [i]
    if sentiment_score[i]==min_sentiment and n_sw>3:
        min_list.append(i)
    if sentiment_score[i]<min_sentiment and n_sw>3:
        min_sentiment=sentiment_score[i]
        min = i
        min_list = [i]
        avg_sentiment += sentiment_score[i]
    avg_sentiment = avg_sentiment/n_descriptions
print("\nCorpus Average Sentiment:{:>5.2f} ".format(avg_sentiment))
print("\nMost Negative Reviews with 4 or more Sentiment Words:")
for i in range(len(min_list)):
    print("{:<s}{:>5d}{:<s}{:>5.2f}".format(" Review ", min_list[i], \
          " Sentiment is ", min_sentiment))
print("\nMost Positive Reviews with 4 or more Sentiment Words:")
for i in range(len(max_list)):
    print("{:<s}{:>5d}{:<s}{:>5.2f}".format(" Review ", max_list[i], \
          " Sentiment is ", max_sentiment))

#Add sentiment score into the df
sens=pd.DataFrame({'Sentiment':sentiment_score})
df_www=pd.concat([df_www,sens],axis=1)
#Print out averages by topic and make
print(df_www.groupby(['Topic'])['Sentiment'].mean(), "\n")

# CORPUS WORD CLOUD
print("\nCorpus Word Cloud")
word_cloud_dic(td, mask=None, max_words=100)
#Corpus Sentiment Words
corpus_sentiment = term_dic(tf, s_terms, scores=sentiment_dic)
word_cloud_dic(corpus_sentiment, mask=None, max_words=100)