import batchldavb
import numpy as n
import corpus, csv, sys
from batchldavb import dirichlet_expectation
from joblib import Parallel, delayed, load, dump
import multiprocessing, tempfile, os
def calculate_prob(wordids, wordcts, gamma, lamb):
    if wordids[0].__class__ == list:
        predprob = 0
        batchD = len(wordids)
        for d in range(0,batchD):
            ids = wordids[d]
            cts = wordcts[d]
            Ebeta = n.dot(dirichlet_expectation(lamb)[:,ids], cts )
            predprob += n.dot(dirichlet_expectation(gamma), Ebeta)
    else:
        ids = wordids
        cts = wordcts
        Ebeta = n.dot(dirichlet_expectation(lamb)[:, ids], cts)
        predprob = n.dot(dirichlet_expectation(gamma), Ebeta)
    return predprob

def pred_prob( wordids, wordcts, gamma, lamb):
    batchD = len(wordids)
    if batchD > 1e4:
        # temp_folder = tempfile.mkdtemp()
        # filename1 = os.path.join(temp_folder, 'joblib_wikiabs_wordids.mmap')
        # filename2 = os.path.join(temp_folder, 'joblib_wikiabs_wordcts.mmap')
        # if os.path.exists(filename1): os.unlink(filename1)
        # if os.path.exists(filename2): os.unlink(filename2)
        # _ = dump(wordids, filename1)
        # _ = dump(wordcts, filename2)
        num_cores = multiprocessing.cpu_count()
        word_list = list()
        tmpids = list()
        tmpcts = list()
        for i in xrange(0, batchD):
            if (i+1) % 10001 == 0:
                word_list.append((tmpids, tmpcts))
                tmpids = []
                tmpcts = []
            else:
                tmpids.append(wordids[i])
                tmpcts.append(wordcts[i])

        predprob = Parallel(n_jobs=num_cores, verbose=1000, pre_dispatch='4*n_jobs')(
            delayed(calculate_prob)(wordids, wordcts, gamma, lamb)
            for wordids, wordcts in word_list)
        predprob = n.sum(predprob)
    else:
        predprob = calculate_prob(wordids, wordcts, gamma, lamb)

    return predprob


def main():
    # parameter setting
    K = 100
    # D is set below
    alpha = 0.5
    eta = 0.5
    kappa = -1
    L = int(sys.argv[1])
    S = 300                 # batchsize

    docs = corpus.read_data('corpus/enwiki-latest-abstract.corpus')
    D = int(docs.num_docs*99./100.)
    vocab = open('dictnostops.txt').readlines()
    max_iter = int(D/S*3)

    model = batchldavb.batchLDA(vocab, K, D, alpha, eta, 0.001, kappa, L)

    testset = range( int(99./100. * docs.num_docs), docs.num_docs )
    num_test = testset.__len__()
    print 'The number of testset is %d' % num_test
    print '%d data are used to fitting local parameter "gamma"' % int(num_test / 2.)
    print '%d data are used to calculating predictive probability' % (num_test - int(num_test / 2.))
    test1_wordids = [docs.docs[idx].words for idx in testset[int(num_test / 2.):]]
    test1_wordcts = [docs.docs[idx].counts for idx in testset[int(num_test / 2.):]]
    test2_wordids = [docs.docs[idx].words for idx in testset[:int(num_test / 2.)]]
    test2_wordcts = [docs.docs[idx].counts for idx in testset[:int(num_test / 2.)]]

    for i in range(0,max_iter):
        minibatch = n.random.randint( int(docs.num_docs) - num_test, size=S)
        wordids = [docs.docs[idx].words for idx in minibatch]
        wordcts = [docs.docs[idx].counts for idx in minibatch]
        # learning lambda with training set
        model.update_lambda(wordids, wordcts)
        if i % S == 0:
            print 'Validation for %d-th iteration' % i
            # learning local param gamma, fixing lambda with half of the test set
            (gamma, sstat) = model.do_e_step(test1_wordids, test1_wordcts)
            # calculate the predictive probability with others of test set.
            predprob = pred_prob(test2_wordids, test2_wordcts, gamma, model._lambda)
            with open('predprob-%d.csv' % L, 'wa') as file:
                wr = csv.writer(file)
                wr.writerow([i, predprob])
            print 'result of %d-th iteration was recorded in file' % i

if __name__ == '__main__':
    main()