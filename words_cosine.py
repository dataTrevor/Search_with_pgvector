# -*- coding: utf-8 -*- 
'''
Version: v3
# Generate and print an embedding with Amazon Titan Text Embeddings V2.
# load data: COPY temp_doc (id, doc_type, doc) FROM 'answer.csv' DELIMITER ',' CSV HEADER;
# RDS:  \copy temp_doc(id, doc_type, doc) from '/home/centos/zhcn_search/answer.csv' WITH DELIMITER ',' CSV HEADER;
# download test data: git clone https://github.com/zhangsheng93/cMedQA2.git
python -m venv zhcn_env
source zhcn_env/bin/activate
pip install numpy
pip install spacy-pkuseg
'''

import sys
import boto3
import json
import psycopg2.extras
from DBUtils.PooledDB import PooledDB
import threading
from typing import Dict, List
import argparse
import datetime
import time
import pytz
import math
import jieba
import spacy_pkuseg as pkuseg
import re
import portalocker
import queue

tz = pytz.timezone('Asia/Shanghai')

host="Your DB Host"
port="Your DB port"
user='Your User name'
password='Your Password'
dbname='Your DB name'
# specify the stop words for jieba
stop_words = ['的','我','你','他','她','得','啊','哈','了','吧', '呢', '如', '就', '可能', '是', '要','和', '好', '哪家', '如何','不','时']

def args_parse():
    parser = argparse.ArgumentParser(description='search test by vector')
    parser.add_argument('--mode', '-m', help='embedding: update embedding, search: search a keyword, split: split document to key words, mandatory', required=True, default='search')
    parser.add_argument('--probes', '-p', help='probes for vectors search, optional', required=False, default=10)
    parser.add_argument('--topk', '-t', help='topk', required=False, default=2)
    parser.add_argument('--input', '-i', help='word to search', required=False)
    parser.add_argument('--maxId', '-r', help='rows to embedding, default 226272, which is same with test data', required=False)
    parser.add_argument('--anaType', '-a', help='jieba: jieba, pkuseg: pkuseg', required=False)
    parser.add_argument('--new', '-n', default='False', action='store_true', help='split words incrementally', required=False)
    args = parser.parse_args()
    return args

class PsycopgConn:

    _instance_lock = threading.Lock()

    def __init__(self):
        self.init_pool()

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            with PsycopgConn._instance_lock:
                if not hasattr(cls, '_instance'):
                    PsycopgConn._instance = object.__new__(cls)
        return PsycopgConn._instance
		
    def get_pool_conn(self):
        """
        get conn from pool
        :return: 
        """
        if not self._pool:
            self.init_pool()
        return self._pool.connection()

    def init_pool(self):
        """
        init pool
        :return: 
        """
        try:
            pool = PooledDB(
                creator=psycopg2,
                maxconnections=64,
                mincached=4,
                maxcached=32,
                blocking=True,
                maxusage=None,
                setsession=[],
                host=host,
                port=port,
                user=user,
                password=password,
                database=dbname)
            self._pool = pool
        except:
            print ('connect postgresql error when init pool')
            self.close_pool()

    def close_pool(self):
        """
        close pool
        :return: 
        """
        if self._pool != None:
            self._pool.close()


    def SelectSql(self,sql):
        """
        query
        :param sql: 
        :return: 
        """
        try:
            conn = self.get_pool_conn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(sql)
            result = cursor.fetchall()
        except Exception as e:
            print('execute sql {0} is error'.format(sql))
            sys.exit('ERROR: query data from database error caused {0}'.format(str(e)))
        finally:
            cursor.close()
            conn.commit()
            conn.close()
        return result
    
    def SelectSqlWithInitSql(self,sql, initSQL):
        """
        query
        :param sql: 
        :param initSQL:
        :return: 
        """
        try:
            conn = self.get_pool_conn()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cursor.execute(initSQL)
            cursor.execute(sql)
            result = cursor.fetchall()
        except Exception as e:
            print('execute sql {0} is error'.format(sql))
            sys.exit('ERROR: query data with init sql from database error caused {0}'.format(str(e)))
        finally:
            cursor.close()
            conn.commit()
            conn.close()
        return result
    
    def InsertSql(self,sql):
        """
        insert data
        :param sql: 
        :return: 
        """
        try:
            conn = self.get_pool_conn()
            cursor = conn.cursor()
            cursor.execute(sql)
            result = True
        except Exception as e:
            print('ERROR: execute  {0} causes error'.format(sql))
            sys.exit('ERROR: insert data from database error caused {0}'.format(str(e)))
        finally:
            cursor.close()
            conn.commit()
            conn.close()
        return result

    def UpdateSql(self,sql,vars=None):
        """
        update
        :param sql: 
        :return: 
        """
        try:
            conn = self.get_pool_conn()
            cursor = conn.cursor()
            cursor.execute(sql, vars)
            result = True
        except Exception as e:
            print('ERROR: execute  {0} causes error'.format(sql))
            sys.exit('ERROR: update data from database error caused {0}'.format(str(e)))
        finally:
            cursor.close()
            conn.commit()
            conn.close()
        return result

def now_time():
    for_now = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    return for_now

def make_print_to_file(path='./', prefix='log_'):
    '''
    path, it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            if not os.path.exists(path):
                os.makedirs(path)
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)
 
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
 
        def flush(self):
            pass
    fileName = prefix + datetime.datetime.now(tz).strftime("%Y%m%d%H%M")
    sys.stdout = Logger(fileName + '.log', path=path)

    print(fileName.center(60,'*'))

# query abstract
def queryAbstracts(pool, tableName, minId, maxId):
    sql = 'select id, doc, embedding_doc from %s where id between %d and %d' % (tableName, minId, maxId)
    return pool.SelectSql(sql)

# query abstract with out key words
def queryAbstractsWithoutKeyword(pool, tableName, minId, maxId):
    sql = 'select id, doc from %s where id between %d and %d' % (tableName, minId, maxId)
    return pool.SelectSql(sql)

def queryAbstractsWithoutKeywordIncre(pool, tableName, minId, maxId):
    sql = 'select id, doc from %s where id between %d and %d and keywords is null' % (tableName, minId, maxId)
    return pool.SelectSql(sql)

def updateEmbeddingById(pool, tableName, id, embedding: List):
    sql = "update %s set embedding_doc='%s'::vector(1536) where id = %d;"
    return pool.UpdateSql(sql % (tableName, embedding, id))

def embedding_titan(input_text: str):
    # Create the request for the model.
    native_request = {"inputText": input_text}
    
    # Convert the native request to JSON.
    request = json.dumps(native_request)
    
    # Invoke the model with the request.
    response = client.invoke_model(modelId=model_id, body=request)
    
    # Decode the model's native response body.
    model_response = json.loads(response["body"].read())
    
    # Extract and print the generated embedding and the input text token count.
    embedding = model_response["embedding"]
    input_token_count = model_response["inputTextTokenCount"]
    return embedding, input_token_count

# batch update the embedding column in table
def batchUpdateEmbedding(pool, maxId: int, tableName):
    rows = queryAbstracts(pool, tableName, 1, maxId)
    for row in rows:
        doc = dict(row)
        doc_text = doc["doc"]
        id = doc["id"]
        embedding, input_token_count = embedding_titan(doc_text)
        updateEmbeddingById(pool, tableName, id, embedding)
        print("doc: %s, token_count: %d " % (doc["doc"], input_token_count))
    pool.close_pool()

# search records by pg vector l2 distance
def searchByWord(input_word: str, pool, probes: int, topk: int, tableName: str):
    word_embedding, count = embedding_titan(input_word)
    initSql = "SET ivfflat.probes = %d" % probes
    sql = "select id, doc, embedding_doc <=> '%s'::vector(1536) as distance from %s order by embedding_doc <=> '%s'::vector(1536) limit %d"
    return pool.SelectSqlWithInitSql(sql % (word_embedding, tableName, word_embedding, topk), initSql)

# merge results in one sql doing vector search and full text search
def searchByWordAndEmbedding(input_word: str, pool, probes: int, topk: int, tableName: str):
    word_embedding, count = embedding_titan(input_word)
    initSql = "SET ivfflat.probes = %d" % probes
    sql = "select id, doc, embedding_doc <=> '%s'::vector(1536) as distance from %s where to_tsvector('simple', keywords) @@ '%s' order by embedding_doc <=> '%s'::vector(1536) limit %d"
    return pool.SelectSqlWithInitSql(sql % (word_embedding, tableName, input_word, word_embedding, topk), initSql)

# search by full text index
def searchByKeyword(input_word: str, pool, topk: int, tableName: str):
    sql = "select id, doc from %s where to_tsvector('simple', keywords) @@ '%s' limit %d"
    return pool.SelectSql(sql % (tableName, input_word, topk))

# search by many words
def searchByMultiKeyword(input_words: str, pool, topk: int, tableName: str):
    if input_words is None or len(input_words) == 0:
        return None
    else:
        keywordList = input_words.split(" ")
        sql = "select id, doc from %s where to_tsvector('simple', keywords) @@ '%s'" % (tableName, keywordList[0])
        for i in range(1, len(keywordList)):
            if i > 4 :
                break
            sql = sql + " and to_tsvector('simple', keywords) @@ '%s'" % keywordList[i]
        sql = sql + " limit %d" % topk
        print("sql for full text: %s" % sql)
        return pool.SelectSql(sql)

def searchRc(tableName: str, input_word: str, pool, probes: int = 10, topk: int = 2):
    idListByembedding = []
    start_time = datetime.datetime.now(tz)
    rows = searchByWord(input_word, pool, probes, topk, tableName)
    end_time = datetime.datetime.now(tz)
    running_seconds = math.ceil((end_time - start_time).total_seconds())
    print("****** search result by embedding: %s , search time: %d sec\n" % (input_word, running_seconds))
    for row in rows:
        doc = dict(row)
        idListByembedding.append(doc['id'])
        print(doc)
    # get words
    segClass = pkuseg.pkuseg(model_name='medicine',user_dict='my_dict.txt')
    ana_type = 'pkuseg'
    keywords, word_cnt = splitDoc(input_word, segClass, ana_type)
    start_time = datetime.datetime.now(tz)
    rowsBykeyWord = searchByMultiKeyword(keywords, pool, topk, tableName)
    end_time = datetime.datetime.now(tz)
    running_seconds = math.ceil((end_time - start_time).total_seconds())
    print("&&&&&& search result by key word: %s , search time: %d sec\n" % (input_word, running_seconds))
    DocsIntersect = []
    for row in rowsBykeyWord:
        doc = dict(row)
        print(doc)
        # intersect
        id = doc['id']
        if id in idListByembedding:
            DocsIntersect.append(doc)
    print("###### search result intersect by Dual-path Recall Search: %s " % (input_word))
    for row in DocsIntersect:
        print(row)
    return

# Create a Bedrock Runtime client in the AWS Region of your choice.
client = boto3.client("bedrock-runtime", region_name="ap-northeast-1")
# Set the model ID, e.g., Titan Text Embeddings V2: amazon.titan-embed-text-v2:0
model_id = "amazon.titan-embed-text-v1"

def test_titan():
    # The text to convert to an embedding.
    input_text = "外周神经病变"
    
    # Create the request for the model.
    native_request = {"inputText": input_text}
    
    # Convert the native request to JSON.
    request = json.dumps(native_request)
    
    # Invoke the model with the request.
    response = client.invoke_model(modelId=model_id, body=request)
    
    # Decode the model's native response body.
    model_response = json.loads(response["body"].read())
    
    # Extract and print the generated embedding and the input text token count.
    embedding = model_response["embedding"]
    input_token_count = model_response["inputTextTokenCount"]
    
    print("\nYour input:")
    print(input_text)
    print("Number of input tokens: %d" % input_token_count)
    print("Size of the generated embedding: %d" % len(embedding))
    print("Embedding:")
    print(embedding)

# batch update the key words to column in table
def splitDoc(doc_text, segClass, ana_type: str = 'pkuseg'):
    if ana_type == 'jieba':
        return splitDocByJieba(doc_text)
    elif ana_type == 'pkuseg' and segClass is not None:
        return splitDocByPkuseg(doc_text, segClass)
    else:
        return splitDocByJieba(doc_text)

# update key word using multi thread
def batchUpdateKeywordsWithSize(pool, batchSize: int, tableName, offsetFile: str, ana_type: str, incre_type: bool):
    beginID, endId = lockFileAndUpdateOffset(offsetFile, batchSize)
    segClass = None
    if ana_type == "pkuseg":
        segClass = pkuseg.pkuseg(model_name='medicine',user_dict='my_dict.txt') 
    rows = queryAbstractsWithoutKeywordIncre(pool, tableName, beginID, endId) if incre_type else queryAbstractsWithoutKeyword(pool, tableName, beginID, endId)
    for row in rows:
        doc = dict(row)
        doc_text = doc["doc"]
        id = doc["id"]
        keywords, word_cnt = splitDoc(doc_text, segClass, ana_type)
        updateKeywordsById(pool, tableName, id, keywords)
        print("doc id: %s, %d key words been splited and updated" % (id, word_cnt))
    pool.close_pool()

# split document by jieba 
def splitDocByJieba(doc_text: str):
    word_list = []
    word_cnt = 0
    # delete punctuation mark
    text = re.sub('\W*', '', doc_text)
    origin_list = jieba.cut(text)
    for word in origin_list:
        if word not in stop_words:
            word_list.append(word)
            word_cnt = word_cnt + 1
    return " ".join(word_list), word_cnt

# split document by spacy_pkuseg 
def splitDocByPkuseg(doc_text: str, segClass):
    word_list = []
    word_cnt = 0
    # delete punctuation mark
    text = re.sub('\W*', '', doc_text)
    origin_list = segClass.cut(text)
    for word in origin_list:
        if word not in stop_words:
            word_list.append(word)
            word_cnt = word_cnt + 1
    return " ".join(word_list), word_cnt

def updateKeywordsById(pool, tableName, id, keywords: str):
    sql = "update %s set keywords='%s, doc_type=' where id = %d;"
    return pool.UpdateSql(sql % (tableName, keywords, id))

# subscribe the queue until empty
def workFromQueue(q, pool, batchSize, tableName, offsetFile, ana_type, incre_type):
    while True:
        if q.empty():          
            end_time = time.strftime('%m-%d %H:%M:%S', time.localtime())
            print ("Queue is empty, end_time:", end_time)
            return
        else:
            t = q.get()
            batchUpdateKeywordsWithSize(pool, batchSize, tableName, offsetFile, ana_type, incre_type)

def lockFileAndUpdateOffset(lockFilename, batchSize: int):
    with open(lockFilename, mode="r+") as f:
        portalocker.lock(f, portalocker.LOCK_EX) # 加锁 / 获取锁独占
        lines = f.readlines() 
        
        firstline = lines[0].split(':')
        offsetString = firstline[1].strip()
        offset = int(offsetString)
        newOffset = offset + batchSize
        f.seek(0)
        f.write("offset: %s" % str(newOffset))
        portalocker.unlock(f)  # 释放锁
    return offset, newOffset

if __name__ == "__main__":
    args = args_parse()
    mode = args.mode
    probes= int(args.probes) if args.probes is not None else None
    topk=int(args.topk) if args.topk is not None else None
    input_word = args.input
    # if maxId not specified , update 100,000 rows
    maxId = int(args.maxId) if args.maxId is not None else 100000
    ana_type = args.anaType
    incre_type = args.new if args.new is not None else False
    tableName = 'text_embedding_cos'
    pool = PsycopgConn()
    if mode == "embedding":
        batchUpdateEmbedding(pool, maxId, tableName)
    elif mode == "search":
        searchRc(tableName, input_word, pool, probes, topk)
    elif mode == "split":
        totalCnt = maxId
        # append element to queue, product queue
        batchSize = 5000
        totalBatches = math.ceil(totalCnt/batchSize)
        offsetFile = 'split_keywords_offset.txt'
        with open(offsetFile, mode="w") as f:
            f.write("offset: %s" % str(0))
        f.close()
        q = queue.Queue()
        for i in range(totalBatches):
            q.put(i)
        # subscribe the queue
        thread_list = []
        for i in range(16):
            t=threading.Thread(target=workFromQueue, args=(q, pool, batchSize, tableName, offsetFile, ana_type, incre_type,))
            # t.setDaemon(True) # 设置为守护线程，不会因主线程结束而中断
            thread_list.append(t)
        for t in thread_list:
            time.sleep(1)
            t.start()
        for t in thread_list:
            t.join() 
        end_time = time.strftime('%m-%d %H:%M:%S', time.localtime())
        print ("Main thread end_time:", end_time)
    pool.close_pool()