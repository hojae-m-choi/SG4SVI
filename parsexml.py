import os, sys
import re
from lxml import etree
from StringIO import StringIO
from batchldavb import parse_doc_list

#3 2:3 4:5 5:3 --- document info (word: count)
class document:
    ''' the class for a single document '''
    def __init__(self):
        self.words = []
        self.counts = []
        self.length = 0
        self.total = 0

def read_xmlfile_to_corpus(filename):
    # Our vocabulary
    vocab_list = file('./dictnostops.txt').readlines()
    vocab = dict()
    for word in vocab_list:
        word = word.lower()
        word = re.sub(r'[^a-z]', '', word)
        vocab[word] = len(vocab)

    if not os.path.exists(filename):
        print 'no data file, please check it'
        return
    print 'reading data from %s.' % filename

    f_in = open(filename,'r')
    f_out = open(re.sub('\.xml', '.corpus1', filename), 'w')

    doc_cts = [0,0]
    record = False
    xml = ''
    for line in f_in:
        if line.startswith('<doc>'):
            record = True
            xml = line
            continue
        elif line.startswith('</doc>'):
            xml = xml + line
            record = False
        else:
            if record == True:
                xml = xml + line
            continue

        root = etree.fromstring(xml)
        abstract = root.find('abstract').text
    # context = etree.iterparse(StringIO(all), events=('start', 'end'), tag=('title', 'abstract'))
    # for action, elem in context:
    #     if action == 'start':
    #         if elem.tag == 'title':
                # title = re.sub(r'Wikipedia: ', '',elem.text)
                # continue
            # elif elem.tag == 'abstract':
            #     abstract = elem.text
        doc_cts[1] += 1
        if abstract is None:
            print u'No abstract for this title.' # % title.encode('utf-8')
            continue
        (wordids, wordcts) = parse_doc_list([abstract], vocab)
        if wordids[0].__len__() == 0:
            print u'No word in this document can be recognized with our vocabulary' # % title.encode('utf-8')
            continue
        doc_cts[0] += 1
        w_cts = zip(wordids[0], wordcts[0])
        str_w_cts = ["%s:%s" % (w, cts) for w, cts in w_cts]
        line_pairs = ' '.join(str_w_cts)
        f_out.write(str(wordids[0].__len__()) + ' ')
        f_out.write(line_pairs +'\n')
    print 'The number of documents : %d / %d(total)' % tuple(doc_cts)
    f_in.close()
    f_out.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        read_xmlfile_to_corpus('corpus/test-enwiki.xml')
    else:
        read_xmlfile_to_corpus(sys.argv[1])




