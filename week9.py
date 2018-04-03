# -*- coding: utf-8 -*-
"""
Author: Chris Berardi
Solution to STAT656 Week 9 Assigment, Spring 2017
Basic Text Analytics
"""

import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.probability import FreqDist

"""
#Download files once
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
"""

file_path = 'C:/Users/Saistout/Desktop/656 Applied Analytics/Data/txt/'
#Create txt file dictionary to loop through reading of text files
adoc={}
for n in range(1,9):
    with open (file_path+"T" + str(n) +".txt", "r") as text_file:
        doc = text_file.read()
        adoc.update({n:doc})
        
# Convert to all lower case - required
#a_discussion = ("%s" %df[0:1]).lower()
a_discussion={}
for n in range(1,9):
    a_discussion.update({n:("%s" %adoc[n]).lower()})
    a_discussion[n] = a_discussion[n].replace('-', ' ')
    a_discussion[n] = a_discussion[n].replace('_', ' ')
    a_discussion[n] = a_discussion[n].replace(',', ' ')
    a_discussion[n] = a_discussion[n].replace("'nt", " not")

# Tokenize
tokens = {}
for n in range(1,9):
    tokens.update({n:word_tokenize(a_discussion[n])})
    tokens[n] = [word.replace(',', '') for word in tokens[n]]
    tokens[n] = [word for word in tokens[n] if ('*' not in word) and \
           word != "''" and word !="``"]
    # Remove punctuation
    for word in tokens[n]:
        word = re.sub(r'[^\w\d\s]+','',word)
    print("\nDocument " + str(n) +" contains a total of", len(tokens[n]),\
          " terms.")
    
# POS Tagging
tagged_tokens = {}
for n in range(1,9):
    tagged_tokens.update({n: nltk.pos_tag(tokens[n])})
    pos_list = [word[1] for word in tagged_tokens[n] if word[1] != ":" and \
                word[1] != "."]
    pos_dist = FreqDist(pos_list)
    pos_dist.plot(title="Parts of Speech: Document "+str(n))
    for pos, frequency in pos_dist.most_common(pos_dist.N()):
        print('{:<15s}:{:>4d}'.format(pos, frequency))
        

# Remove stop words
stop = stopwords.words('english') + list(string.punctuation)
stop_tokens={}
for n in range(1,9):
    stop_tokens.update({n:[word for word in tagged_tokens[n] if word[0] not in stop]})
# Remove single character words and simple punctuation
    stop_tokens[n] = [word for word in stop_tokens[n] if len(word) > 1]
# Remove numbers and possive "'s"
    stop_tokens[n] = [word for word in stop_tokens[n] \
               if (not word[0].replace('.','',1).isnumeric()) and \
               word[0]!="'s" ]
    print("\nDocument " + str(n)+" contains", len(stop_tokens[n]), \
                      " terms after removing stop words.\n")
    token_dist = FreqDist(stop_tokens[n])
    for word, frequency in token_dist.most_common(20):
        print('{:<15s}:{:>4d}'.format(word[0], frequency))
        
# Lemmatization - Stemming with POS
# WordNet Lematization Stems using POS
stemmer = SnowballStemmer("english")
wn_tags = {'N':wn.NOUN, 'J':wn.ADJ, 'V':wn.VERB, 'R':wn.ADV}
wnl = WordNetLemmatizer()
stemmed_tokens = {}
for n in range(1,9):
    stem = []
    for token in stop_tokens[n]:
        term = token[0]
        pos  = token[1]
        pos  = pos[0]
        try:
            pos   = wn_tags[pos]
            stem.append(wnl.lemmatize(term, pos=pos))
        except:
            stem.append(stemmer.stem(term))   
    stemmed_tokens.update({n:stem})
    print("Document "+str(n)+" contains", len(stemmed_tokens[n]), "terms after stemming.\n") 
    
    
#Print out total number of terms
num=0
for n in range(1,9):
    num=num+len(stemmed_tokens[n])

# Word distribution after applying POS and Stemming and stopwords
#fdist = FreqDist(word for word in stemmed_tokens)
#Create word list that will contain the stemmed_tokens from all of the files
word = []
for n in range(1,9):
    word.extend(stemmed_tokens[n])
# Use with Wordnet
    fdist=FreqDist(word)
    print('Top 20 Terms')
    for word, freq in fdist.most_common(20):
        print('{:<15s}:{:>4d}'.format(word, freq))
    print('\n')
 

'''
Stemming Stopwords use raw tokens with stopwords then stem without POS
'''
stop_tokens={}
for n in range(1,9):
#Use the tokens list to not apply POS
    stop_tokens.update({n:[word for word in tokens[n] if word[0] not in stop]})
# Remove single character words and simple punctuation
    stop_tokens[n] = [word for word in stop_tokens[n] if len(word) > 1]
# Remove numbers and possive "'s"
    stop_tokens[n] = [word for word in stop_tokens[n] \
               if (not word[0].replace('.','',1).isnumeric()) and \
               word[0]!="'s" ]
        

stemmed_tokens = {}
for n in range(1,9):
    stem = []
    for token in stop_tokens[n]:
        try:
            stem.append(wnl.lemmatize(token))
        except:
            stem.append(stemmer.stem(token))   
    stemmed_tokens.update({n:stem})
    print("Document "+str(n)+" contains", len(stemmed_tokens[n]), "terms after stemming.\n") 

#Print out total number of terms
num=0
for n in range(1,9):
    num=num+len(stemmed_tokens[n])

word = []
for n in range(1,9):
    word.extend(stemmed_tokens[n])
# Use with Wordnet
    fdist=FreqDist(word)
    print('Top 20 Terms')
    for word, freq in fdist.most_common(20):
        print('{:<15s}:{:>4d}'.format(word, freq))
    print('\n')
    
'''
# Stopwords Only, so use the stopwords from above without POS
'''
num=0
for n in range(1,9):
    num=num+len(stop_tokens[n])

word = []
for n in range(1,9):
    word.extend(stop_tokens[n])
# Use with Wordnet
    fdist=FreqDist(word)
    print('Top 20 Terms')
    for word, freq in fdist.most_common(20):
        print('{:<15s}:{:>4d}'.format(word, freq))
    print('\n')
    
''' 
# No stopwords, or POS or stemming: so just the raw tokens
'''
num=0
for n in range(1,9):
    num=num+len(tokens[n])

word = []
for n in range(1,9):
    word.extend(tokens[n])
# Use with Wordnet
    fdist=FreqDist(word)
    print('Top 20 Terms')
    for word, freq in fdist.most_common(20):
        print('{:<15s}:{:>4d}'.format(word, freq))
    print('\n')