#!/usr/bin/env python
# coding: utf-8

# In[102]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import scipy.spatial.distance as dist
import re
import psycopg2
from sklearn.externals import joblib
import pandas.io.sql as sqlio

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# In[103]:


# URL Removal
def remove_urls(articles):
    removedUrlArticles = []
    for article in articles:
        removedUrlArticles.append(re.sub('http\S+', '', article))
    return removedUrlArticles


# Lower Case Conversion
def convert_to_lower(articles):
    lowerArticles = []
    for article in articles:
        lowerArticles.append(article.lower())
    return lowerArticles


# Stop word removal
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download the set of stop words the first time
import nltk
nltk.download('stopwords')
nltk.download('punkt')


def remove_stopwords(articles):
    noStopWordArticles = []

    stop_words = set(stopwords.words('english')) # Set improves performance
  
    for article in articles:  
        word_tokens = word_tokenize(article) 
        filtered_article = [word for word in word_tokens if not word in stop_words] 
        filteredArticleString = ' '.join(word for word in filtered_article)
        noStopWordArticles.append(filteredArticleString)
        #print(tweet)
        #print(filtered_tweet)    
        #print (filteredTweetString)
    return noStopWordArticles


# Character removal
def remove_unwanted_characters(articles):
    unwantedChars = '''()-[]{};:'"\,<>./@#$%^&*_~1234567890'''
    cleanArticles = []
    for article in articles:
        for punc in list(unwantedChars):
            article = article.replace(punc,'')
        cleanArticles.append(article)
    return cleanArticles


def list_to_str(alist):
    string_list = [str(i) for i in alist]
    return '{'+','.join(string_list)+'}'



def initialize_DB(file):
    
    try:
        rawData = pd.read_csv(file, encoding='ISO-8859-1')
        data = rawData[['author', 'link', 'title', 'text']]
        data.drop_duplicates(subset ="title", keep = False, inplace = True) 

        articles = data['text'].tolist()

        articles = remove_urls(articles)
        articles = convert_to_lower(articles)
        articles = remove_stopwords(articles)
        articles = remove_unwanted_characters(articles)
        data['author'] = remove_unwanted_characters(data['author'])
        data['title'] = remove_unwanted_characters(data['title'])

        tfidfVec = TfidfVectorizer()
        text_tfidf = tfidfVec.fit_transform(articles)
        text_tfidf = text_tfidf.toarray()
        data['tf_idf'] = text_tfidf.tolist()

        joblib.dump(tfidfVec, 'tfidf_vectorizer.pkl')


        #connect to the db
        con = psycopg2.connect(
            host = 'localhost',
            database = 'Article',
            user = 'jim',
            password = 'postgre'
        )

        #cursor
        cur = con.cursor()


        cur.execute(f"CREATE TABLE articles (article_id int  primary key NOT NULL, author varchar(255), title varchar(255), article_link text, tf_idf float[])")


        #execute query
        for index, row in data.iterrows():
            cur.execute(f"insert into articles (article_id, author, title, article_link, tf_idf) values ({index}, '{row['author']}' , '{row['title']}', '{row['link']}', '{list_to_str(row['tf_idf'])}')")



        #commit the transaction
        con.commit()

        #close the cursor
        cur.close()

        #close the connection
        con.close()
        
    except:
        print('DB already initialized')
        
    database = retrieve_db()
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    
    return database, vectorizer
    
    
def retrieve_db():
    
    #connect to the db
    con = psycopg2.connect(
        host = 'localhost',
        database = 'Article',
        user = 'jim',
        password = 'postgre'
    )
    
    
    #execute query
    sql = 'select * from articles;'
    
    data = sqlio.read_sql_query(sql, con)
    

    #close the connection
    con.close()
    
    return data

    
    
def retrieve_relevant_articles(query, k, distance, vectorizer, database):
    
    query = query.lower()
    lst = [query]
    series = pd.Series(lst)
    query_tfidf = vectorizer.transform(series)
    query_tfidf = query_tfidf.toarray()
    
    
    if distance == 'euclidean':
        euclidean_dist_list = []
        for vector in database['tf_idf']:
            euclidean_dist_list.append(dist.euclidean(vector, query_tfidf))
        database["euclidean_dist"] = euclidean_dist_list
        relevant_articles = database.sort_values(by=['euclidean_dist'])[0:k]
     
    
    elif distance == 'minkowski':
        minkowski_dist_list = []
        for vector in database['tf_idf']:
            minkowski_dist_list.append(dist.minkowski(vector, query_tfidf))
        database["minkowski"] = minkowski_dist_list
        relevant_articles = database.sort_values(by=['minkowski'])[0:k]
    
    
    elif distance == 'chebyshev':
        chebyshev_dist_list = []
        for vector in database['tf_idf']:
            chebyshev_dist_list.append(dist.chebyshev(vector, query_tfidf))
        database["chebyshev_dist"] = chebyshev_dist_list
        relevant_articles = database.sort_values(by=['chebyshev_dist'])[0:k]
        
        
    elif distance == 'dice':
        dice_dist_list = []
        for vector in database['tf_idf']:
            dice_dist_list.append(dist.dice(vector, query_tfidf))
        database["dice_dist"] = dice_dist_list
        relevant_articles = database.sort_values(by=['dice_dist'])[0:k]
        
        
    elif distance == 'cosine':
        cosine_dist_list = []
        for vector in database['tf_idf']:
            cosine_dist_list.append(dist.cosine(vector, query_tfidf))
        database["cosine_dist"] = cosine_dist_list
        relevant_articles = database.sort_values(by=['cosine_dist'])[0:k]

    
    
    return relevant_articles



def convert_to_dict (results): 
    return {'result' : results[["author","title","article_link"]].to_dict('records')}


# In[ ]:


from flask import Flask, current_app, request, render_template

app = Flask(__name__)

database, vectorizer = initialize_DB('articles.csv')

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/post', methods = ['POST'])
def post():
    data = request.form
    results = retrieve_relevant_articles(data['query'], int(data['k']), data['distance'], vectorizer, database)
    results_dict = convert_to_dict(results)
    
    return results_dict

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
@app.route('/shutdown', methods=['GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

if __name__ == '__main__':
    app.run()


# In[ ]:




