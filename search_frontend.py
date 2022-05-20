from flask import Flask, request, jsonify
# import sys
from collections import Counter, OrderedDict
# import itertools
# import islice, count, groupby
import pandas as pd
# import os
import re
import numpy as np
# import builtins
# from numpy import dot
# from numpy.linalg import norm
# from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
# from time import time
# from timeit import timeit
from pathlib import Path
# from bs4 import BeautifulSoup
from google.cloud import storage
import math
# import hashlib
from contextlib import closing
from inverted_index_gcp import MultiFileReader
# def _hash(s):
#     return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

nltk.download('stopwords')

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)
search_stopwords=english_stopwords.union(['new', 'births', 'born', 'time', 'year', 'united', 'years', 'american', 'known', 'living', 'national', 'later', 'three', 'world','best', 'made', 'states', 'early', 'family', 'life', 'name', 'city', 'state', 'since', 'university', 'well', 'south', 'used', 'career', 'list', 'school'])

TUPLE_SIZE = 6       
TF_MASK = 2 ** 16 - 1 # Masking the 16 low bits of an integer

# from inverted_index_colab import *
import inverted_index_gcp
import pickle



drive="./postings_gcp/"
driveText="./text_gcp/"
driveC="./pr/"
driveTilte="./indexTitle/"

from inverted_index_gcp import InvertedIndex

index_text=InvertedIndex.read_index(driveText, "index")
index_title= InvertedIndex.read_index(driveTilte, "index_title")
index_anchor=InvertedIndex.read_index(drive, "index_anchor")

wid2pv=None
with open(driveC+"pageviews-202108-user.pkl", 'rb') as f:
  wid2pv = pickle.loads(f.read())

nf=None
with open(driveC+"postings_gcp_norm.pkl", 'rb') as f:
    nf=pickle.load(f)

titlesDocs=None
with open(driveC+"titles.pkl", 'rb') as f:
    titlesDocs=pickle.load(f)

pagerank = pd.read_csv(driveC+'part-00000-e292183f-3bf4-4e21-9745-5b824a3714d0-c000.csv.gz', compression='gzip', header=0, sep=' ', quotechar='"', error_bad_lines=False)
pagerank.columns=["id"]
new = pagerank["id"].str.split(",", n = 1, expand = True)
pagerank["id"]= new[0]
pagerank["page rank"]= new[1]
pagerank.set_index('id')
dict_scores = pagerank['page rank'].to_dict()

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


def read_posting_list(path,inverted, w):
  with closing(MultiFileReader()) as reader:
    locs = inverted.posting_locs[w]
    b = reader.read(path,locs, inverted.df[w] * TUPLE_SIZE)
    posting_list = []
    for i in range(inverted.df[w]):
      doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
      tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
      posting_list.append((doc_id, tf))
    return posting_list

def generate_tfidf (query_to_search, index,N=3):
    epsilon = .0000001
    query = [q for q in query_to_search if q in index.df.keys()]
    counter = Counter(query)
    summ = 0
    for i in (counter).values():
        summ += i * i
    q = 1/math.sqrt(summ)
    sim = {}
    for term in np.unique(query):
        if term in index.df.keys():
            list_of_doc = read_posting_list(driveText, index, term)
            tf = counter[term] / len(query)
            df = index.df[term]
            idf = math.log((len(nf)) / (df + epsilon), 10)
            tfidfQ = tf * idf
            for doc_id, freq in list_of_doc:
                tfidfD = (freq / nf[doc_id]) * math.log(len(nf) / index.df[term], 10)
                sim[doc_id] = sim.get(doc_id, 0) + tfidfD * tfidfQ
    return sorted([(doc_id, score) for doc_id, score in sim.items()], key=lambda x: x[1], reverse=True)[:N]


def title_from_id(lst):
    titles=[]
    for doc_id,_ in lst:
      if doc_id in titlesDocs:
        titles.append((doc_id,titlesDocs[doc_id]))
    return titles


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    tokens=[token for token in tokens if token in index_text.df and index_text.df[token]<300000]
    lst_doc = generate_tfidf(tokens, index_text,100)
    res=title_from_id(lst_doc)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    lst_doc = generate_tfidf(tokens, index_text,100)
    res=title_from_id(lst_doc)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    dic={}
    for term in tokens:
        if index_title.df.get(term):
            ls_doc_freq=read_posting_list(driveTilte,index_title,term)
            for doc,freq in ls_doc_freq:
              dic[doc]=dic.get(doc,0) + freq
    lst_doc=Counter(dic).most_common()
    res = title_from_id(lst_doc)
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    dic = {}
    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    for term in tokens:
        if index_anchor.df.get(term):
            ls_doc_freq = read_posting_list(drive,index_anchor,term)
            for doc, freq in ls_doc_freq:
                dic[doc] = dic.get(doc, 0) + freq
    lst_doc = Counter(dic).most_common()
    res = title_from_id(lst_doc)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    for id_doc in wiki_ids:
        res.append(dict_scores.get(id_doc,0))
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    for id_doc in wiki_ids:
        res.append(wid2pv.get(id_doc,0))
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080)
