import os
from itertools import izip
import re
from joblib import Parallel, delayed
import multiprocessing
import threading
# read and organize data

#3 2:3 4:5 5:3 --- document info (word: count)
class document:
    ''' the class for a single document '''
    def __init__(self):
        self.words = []
        self.counts = []
        self.length = 0
        self.total = 0

class corpus:
    ''' the class for the whole corpus'''
    def __init__(self):
        self.size_vocab = 0
        self.docs = []
        self.num_docs = 0

    def read_data(self, filename):
        if not os.path.exists(filename):
            print 'no data file, please check it'
            return
        print 'reading data from %s.' % filename

        for line in file(filename): 
            ss = line.strip().split()
            if len(ss) == 0: continue
            doc = document()
            doc.length = int(ss[0])

            doc.words = [0 for w in range(doc.length)]
            doc.counts = [0 for w in range(doc.length)]
            for w, pair in enumerate(re.finditer(r"(\d+):(\d+)", line)):
                doc.words[w] = int(pair.group(1))
                doc.counts[w] = int(pair.group(2))

            doc.total = sum(doc.counts) 
            self.docs.append(doc)

            if doc.length > 0:
                max_word = max(doc.words)
                if max_word >= self.size_vocab:
                    self.size_vocab = max_word + 1

            if (len(self.docs) >= int(4.1e6) ):
                break
        self.num_docs = len(self.docs)
        print "finished reading %d docs." % self.num_docs

# def read_data(filename):
#     c = corpus()
#     c.read_data(filename)
#     return c

def read_stream_data(f, num_docs):
  c = corpus()
  splitexp = re.compile(r'[ :]')
  for i in range(num_docs):
    line = f.readline()
    line = line.strip()
    if len(line) == 0:
      break
    d = document()
    splitline = [int(i) for i in splitexp.split(line)]
    wordids = splitline[1::2]
    wordcts = splitline[2::2]
    d.words = wordids
    d.counts = wordcts
    d.total = sum(d.counts)
    d.length = len(d.words)
    c.docs.append(d)

  c.num_docs = len(c.docs)
  return c

def ext_data (c, lines):
    print
    for line in lines:
        d = document()
        splitexp = re.compile(r'[ :]')
        splitline = [int(i) for i in splitexp.split(line)]
        wordids = splitline[1::2]
        wordcts = splitline[2::2]
        d.words = wordids
        d.counts = wordcts
        d.total = sum(d.counts)
        d.length = len(d.words)
        c.docs.append(d)

# class read_data_Thread(threading.Thread):
#     c = corpus()
#     c.d = document()
#
#
#     lock = threading.Lock()
#
#     def run(self):
#         (article, articlename) = get_random_wikipedia_article()
#         read_data_Thread.lock.acquire()
#         read_data_Thread.c.d.words =
#         read_data_Thread.articlenames.append(articlename)
#         read_data_Thread.lock.release()


class ext_data_Thread(threading.Thread):
    lock = threading.Lock()
    lines = list()
    docs = list()
    size_vocab = 0

    def run(self):
        tmp = 0
        tmp_doc_list = list()
        cts = 0
        for line in self.lines:
            cts +=1
            d = document()
            splitexp = re.compile(r'[ :]')
            splitline = [int(i) for i in splitexp.split(line)]
            wordids = splitline[1::2]
            wordcts = splitline[2::2]
            d.words = wordids
            d.counts = wordcts
            d.total = sum(d.counts)
            d.length = len(d.words)
            if d.length > 0:
                max_word = max(d.words)
                if max_word >= tmp:
                    tmp = max_word + 1
            tmp_doc_list.append(d)
            if cts % 100 == 0:
                ext_data_Thread.lock.acquire()
                map (ext_data_Thread.docs.append, tmp_doc_list)
                ext_data_Thread.size_vocab = max( tmp, ext_data_Thread.size_vocab)
                ext_data_Thread.lock.release()
                tmp_doc_list =[]
        ext_data_Thread.lock.acquire()
        map (ext_data_Thread.docs.append, tmp_doc_list)
        ext_data_Thread.size_vocab = max( tmp, ext_data_Thread.size_vocab)
        ext_data_Thread.lock.release()

# This version is about 33% faster
def read_data(filename):
    c = corpus()
    splitexp = re.compile(r'[ :]')
    if not os.path.exists(filename):
        print 'no data file, please check it'
        return
    print 'reading data from %s.' % filename
    # To calculate the total number of documents in a corpus file
    for i, l in enumerate(open(filename)):
        pass
    num_lines = i + 1

    if num_lines > 1e5:
        maxthreads = 8
        ext_data_Thread.docs = list()
        ext_data_Thread.size_vocab = 0;
        maxlen = num_lines/maxthreads
        edt_list = list()
        with open(filename) as f:
            for idx in range(maxthreads):
                lines = list()
                for line in f:
                    lines.append(line)
                    if lines.__len__() > maxlen:
                        break
                edt = ext_data_Thread()
                edt.idx = idx; edt.lines = lines;
                edt_list.append(edt)
                edt_list[len(edt_list) - 1].start()

        for idx in range(maxthreads):
            edt_list[idx].join()
        c.docs = ext_data_Thread.docs
        c.size_vocab = ext_data_Thread.size_vocab


        # num_cores = multiprocessing.cpu_count()
        # with Parallel(n_jobs=num_cores) as parallel:
        #     with open(filename) as f:
        #         for lines in f.readlines(3):
        #             parallel( delayed(ext_data)(c, lines))
    else:
        for line in open(filename):
            d = document()
            splitline = [int(i) for i in splitexp.split(line)]
            wordids = splitline[1::2]
            wordcts = splitline[2::2]
            d.words = wordids
            d.counts = wordcts
            d.total = sum(d.counts)
            d.length = len(d.words)
            c.docs.append(d)

            if d.length > 0:
                max_word = max(d.words)
                if max_word >= c.size_vocab:
                    c.size_vocab = max_word + 1

    c.num_docs = len(c.docs)
    print "finished reading %d docs." % c.num_docs

    return c

def count_tokens(filename):
    num_tokens = 0
    splitexp = re.compile(r'[ :]')
    for line in open(filename):
        splitline = [int(i) for i in splitexp.split(line)]
        wordcts = splitline[2::2]
        num_tokens += sum(wordcts)

    return num_tokens

splitexp = re.compile(r'[ :]')
def parse_line(line):
    line = line.strip()
    d = document()
    splitline = [int(i) for i in splitexp.split(line)]
    wordids = splitline[1::2]
    wordcts = splitline[2::2]
    d.words = wordids
    d.counts = wordcts
    d.total = sum(d.counts)
    d.length = len(d.words)
    return d
