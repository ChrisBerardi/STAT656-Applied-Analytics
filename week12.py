"""
Author: Chris Berardi
Solution to STAT656 Week 12 Assigment, Spring 2017
Sentiment Analysis and Word Clouds
"""

import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import random
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation

# Increase column width to let pandas read large text columns
pd.set_option('max_colwidth', 32000)
# Read N=13,575 California Cabernet Savignon Reviews
file_path = 'C:/Users/Saistout/Desktop/656 Applied Analytics/Python/Week 12 Assignment/'
df = pd.read_excel(file_path+"hotels.xlsx")
sw = pd.read_excel(file_path+"afinn_sentiment_words.xlsx")

#Do cluster analysis to identify 7 word clusters using TFIDF and SVD
def my_analyzer(s):
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
    other = ['go','get','one','room','hotel','day','casino','quot','strip',\
             'amp','get','also','night','would','take','place','time']
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

def display_topics(lda, terms, n_terms=15):
    for topic_idx, topic in enumerate(lda):
        if topic_idx > 8: 
            break
        message  = "Topic #%d: " %(topic_idx+1)
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
    return

# Setup program constants
n_comments  = len(df['Review'])     # Number of hotel reviews
m_features = None                    # Number of SVD Vectors
s_words    = 'english'               # Stop Word Dictionary
comments = df['Review']             # place all text reviews in reviews
n_topics =  7                       # number of topic clusters to extract
max_iter = 100                        # maximum number of itertions  
max_df   = 0.7                      # learning offset for LDAmax proportion of docs/reviews allowed for a term
learning_offset = 10.               # learning offset for LDA
learning_method = 'online'           # learning method for LDA
tfidf = True                    # Set to True for TF-IDF Weighting

# Create Word Frequency by Review Matrix using Custom Analyzer
cv = CountVectorizer(max_df=max_df, min_df=4, max_features=m_features,\
                     analyzer=my_analyzer, ngram_range=(1,2))
tf    = cv.fit_transform(comments)
terms = cv.get_feature_names()
term_sums = tf.sum(axis=0)
term_counts = []
for i in range(len(terms)):
    term_counts.append([terms[i], term_sums[0,i]])
def sortSecond(e):
    return e[1]
term_counts.sort(key=sortSecond, reverse=True)
print("\nTerms with Highest Frequency:")
for i in range(50):
        print('{:<15s}{:>5d}'.format(term_counts[i][0], term_counts[i][1]))
print("")
# Modify tf, term frequencies, to TF/IDF matrix from the data
print("Conducting Term/Frequency Matrix using TF-IDF")
tfidf_vect = TfidfTransformer(norm=None, use_idf=True) #set norm=None
tf         = tfidf_vect.fit_transform(tf)

term_idf_sums = tf.sum(axis=0)
term_idf_scores = []
for i in range(len(terms)):
    term_idf_scores.append([terms[i], term_idf_sums[0,i]])
term_idf_scores.sort(key=sortSecond, reverse=True)

# In sklearn, SVD is synonymous with LSA (Latent Semantic Analysis)
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter,\
learning_method=learning_method, \
learning_offset=learning_offset, \
random_state=12345)
lda.fit_transform(tf)

# Display the topic selections
lda_norm = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
# ***** SCORE REVIEWS *****
rev_scores = [[0]*(n_topics+1)] * n_comments
# Last topic count is number of reviews without any topic words
topic_counts = [0] * (n_topics+1)
for r in range(n_comments):
    idx = n_topics
    max_score = 0
    # Calculate Review Score
    j0 = tf[r].nonzero()
    nwords = len(j0[1])
    rev_score = [0]*(n_topics+1)
    # get scores for rth doc, ith topic
    for i in range(n_topics):
        score = 0
        for j in range(nwords):
            j1 = j0[1][j]
            if tf[r,j1] != 0:
                score += lda_norm[i][j1] * tf[r,j1]
        rev_score [i+1] = score
        if score>max_score:
            max_score = score
            idx = i
    # Save review's highest scores
    rev_score[0] = idx
    rev_scores [r] = rev_score
    topic_counts[idx] += 1
print('{:<6s}{:>8s}{:>8s}'.format("TOPIC", "REVIEWS", "PERCENT"))
for i in range(n_topics):
    print('{:>3d}{:>10d}{:>8.1%}'.format((i+1), topic_counts[i], \
          topic_counts[i]/n_comments))
    
topics = pd.DataFrame(rev_scores)
topics=topics.iloc[:,0]
topics = pd.DataFrame(topics)
topics.columns=['Topic']
df=pd.concat([df,topics],axis=1)

# Setup Sentiment dictionary
sentiment_dic = {}
for i in range(len(sw)):
    sentiment_dic[sw.iloc[i][0]] = sw.iloc[i][1]

#Define the preprocessor for use with sentiment analysis
cv = CountVectorizer(max_df=1.0, min_df=1, max_features=None, \
                     preprocessor=my_preprocessor, ngram_range=(1,2))
tf = cv.fit_transform(df['Review'])
terms = cv.get_feature_names()
n_reviews = tf.shape[0]
n_terms = tf.shape[1]
print('{:.<22s}{:>6d}'.format("Number of Reviews", n_reviews))
print('{:.<22s}{:>6d}'.format("Number of Terms", n_terms))
print("\nTopics Identified using LDA with TF_IDF")
display_topics(lda.components_, terms, n_terms=15)

#Caculate semtiment for each document and for the whole corpus
min_sentiment = +5
max_sentiment = -5
avg_sentiment, min, max = 0,0,0
min_list, max_list = [],[]
sentiment_score = [0]*n_reviews
for i in range(n_reviews):
    # Iterate over the terms with nonzero scores
    n_sw = 0
    term_list = tf[i].nonzero()[1]
    if len(term_list)>0:
        for t in np.nditer(term_list):
            3
            score = sentiment_dic.get(terms[t])
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
avg_sentiment = avg_sentiment/n_reviews
print("\nCorpus Average Sentiment: ", avg_sentiment)
print("\nMost Negative Reviews with 4 or more Sentiment Words:")
for i in range(len(min_list)):
    print("{:<s}{:<d}{:<s}{:<5.2f}".format(" Review ", min_list[i], \
          " Sentiment is ", min_sentiment))
print("\nMost Positive Reviews with 4 or more Sentiment Words:")
for i in range(len(max_list)):
    print("{:<s}{:<d}{:<s}{:<5.2f}".format(" Review ", max_list[i], \
          " Sentiment is ", max_sentiment))
    
#Calculate the averages by hotel, cluster and hotel x cluster
#Add sentiment score into a new df
sens=pd.DataFrame({'Score':sentiment_score})
df=pd.concat([df,sens],axis=1)
hotels = ["Bally's",'Bellagio','Circus Circus','Encore','Excalibur']

print("\n********** Average Sentiment by Hotel **********")
for h in hotels:
    idx = df.index[df['hotel'] == h]
    this_hotel = df.loc[idx]
    sc = this_hotel['Score'].mean()
    print("\nAverage Sentiment for", h, ":", sc)

print("\n********** Average Sentiment by Topic **********")  
for c in range(0,7):
    idx = df.index[df['Topic'] == c]
    this_topic = df.loc[idx]
    sc = this_topic['Score'].mean()
    print("\nAverage Sentiment for Topic", c, ":", sc)
hi_topic = 2
low_topic = 4
    
print("\n********** Average Sentiment by Hotel x Topic **********")
for h in hotels:
    idx = df.index[df['hotel'] == h]
    this_hotel = df.loc[idx]
    for c in range(0,7):
        idx = this_hotel.index[this_hotel['Topic'] == c]
        this_topic = this_hotel.loc[idx]
        sc = this_topic['Score'].mean()
        print("\nAverage Sentiment for" ,h, "Topic", c, ":", sc)
        
#Word Clouds
def shades_of_grey(word, font_size, position, orientation, random_state=None, \
                   **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60,1000)
#Word cloud for all word in the reviews
st = set(STOPWORDS)
st.add("hotel")
st.add("room")
st.add("quot")
st.add("one")
st.add("casino")
wc = WordCloud(stopwords=st,width=600, height=400)
s = ""
for i in range(len(comments)):
    s += comments[i]
wc.generate(s)
# Display the word cloud.
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.figure()
plt.show()

#From the sentiment words of all words
corpus_sentiment = {}
n_sw = 0
for i in range(n_reviews):
    # Iterate over the terms with nonzero scores
    term_list = tf[i].nonzero()[1]
    if len(term_list)>0:
            for t in np.nditer(term_list):
                score = sentiment_dic.get(terms[t])
                if score != None:
                    n_sw += tf[i,t]
                    current_count = corpus_sentiment.get(terms[t])
                    if current_count == None:
                        corpus_sentiment[terms[t]] = tf[i,t]
                    else:
                        corpus_sentiment[terms[t]] += tf[i,t]
print("The Corpus contains a total of ", len(corpus_sentiment), " unique sentiment words")
print("The total number of sentiment words in the Corpus is", n_sw)

wc = WordCloud(background_color="green", max_words=200, stopwords=sw, \
max_font_size=40, min_font_size=10, prefer_horizontal=0.7, \
relative_scaling=0.5, width=400, height=200, \
margin=10, random_state=341)
wc.generate_from_frequencies(corpus_sentiment)
plt.imshow(wc.recolor(color_func=shades_of_grey, random_state=3), interpolation="bilinear")
plt.axis("off")
plt.figure()

#Words in cluster with lowest sentiment score
idx=df.index[df['Topic'] == low_topic]
low = df.loc[idx]
wc = WordCloud(stopwords=st, width=600, height=400)
s = ""
low_topics=low['Review']
for i in idx:
    s += low_topics[i]
wc.generate(s)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.figure()
plt.show()

#Sentiment Words in cluster with lowest sentiment score
low_sentiment = {}
n_sw = 0
for i in idx:
    # Iterate over the terms with nonzero scores
    term_list = tf[i].nonzero()[1]
    if len(term_list)>0:
            for t in np.nditer(term_list):
                score = sentiment_dic.get(terms[t])
                if score != None:
                    n_sw += tf[i,t]
                    current_count = low_sentiment.get(terms[t])
                    if current_count == None:
                        low_sentiment[terms[t]] = tf[i,t]
                    else:
                        low_sentiment[terms[t]] += tf[i,t]
                        
wc = WordCloud(background_color="red", max_words=200, stopwords=sw, \
max_font_size=40, min_font_size=10, prefer_horizontal=0.7, \
relative_scaling=0.5, width=400, height=200, \
margin=10, random_state=341)
wc.generate_from_frequencies(low_sentiment)
plt.imshow(wc.recolor(color_func=shades_of_grey, random_state=3), interpolation="bilinear")
plt.axis("off")
plt.figure()
                      
#Words in cluster with higest sentiment score
idx=df.index[df['Topic'] == hi_topic]
hi = df.loc[idx]
wc = WordCloud(stopwords=st,width=600, height=400)
s = ""
hi_topics=hi['Review']
for i in idx:
    s += hi_topics[i]
wc.generate(s)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.figure()
plt.show()

#Sentiment Words in cluster with highest sentiment score
hi_sentiment = {}
n_sw = 0
for i in idx:
    # Iterate over the terms with nonzero scores
    term_list = tf[i].nonzero()[1]
    if len(term_list)>0:
            for t in np.nditer(term_list):
                score = sentiment_dic.get(terms[t])
                if score != None:
                    n_sw += tf[i,t]
                    current_count = hi_sentiment.get(terms[t])
                    if current_count == None:
                        hi_sentiment[terms[t]] = tf[i,t]
                    else:
                        hi_sentiment[terms[t]] += tf[i,t]
                        
wc = WordCloud(background_color="blue", max_words=200, stopwords=sw, \
max_font_size=40, min_font_size=10, prefer_horizontal=0.7, \
relative_scaling=0.5, width=400, height=200, \
margin=10, random_state=341)
wc.generate_from_frequencies(hi_sentiment)
plt.imshow(wc.recolor(color_func=shades_of_grey, random_state=3), interpolation="bilinear")
plt.axis("off")
plt.figure() 