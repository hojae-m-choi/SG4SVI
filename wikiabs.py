import batchldavb
import numpy as n
from scipy.special import psi
import corpus, csv, sys
from batchldavb import dirichlet_expectation
from joblib import Parallel, delayed

def pred_prob( wordids, wordcts, gamma, lamb):
    batchD = len(wordids)
    if batchD > 1e4:
        num_cores = multiprocessing.cpu_count()
        predprob = Parallel(n_jobs=num_cores)(delayed(pred_prob)(wordids[d],wordcts[d], gamma, lamb) for d in range(0,batchD))
        predprob = n.sum(predprob)
    else:
        predprob = 0
        for d in range(0,batchD):
            ids = wordids[d]
            cts = wordcts[d]
            Ebeta = n.dot(dirichlet_expectation(lamb)[:,ids], cts )
            predprob += n.dot(dirichlet_expectation(gamma), Ebeta)

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
    D = int(docs.num_docs*3./4.)
    vocab = open('dictnostops.txt').readlines()
    max_iter = int(D/S*3)

    model = batchldavb.batchLDA(vocab, K, D, alpha, eta, 0.001, kappa, L)

    testset = range( int(3./4. * docs.num_docs), docs.num_docs )
    len = testset.__len__()
    test1_wordids = [docs.docs[idx].words for idx in testset[int(len / 2.):]]
    test1_wordcts = [docs.docs[idx].counts for idx in testset[int(len / 2.):]]
    test2_wordids = [docs.docs[idx].words for idx in testset[:int(len / 2.)]]
    test2_wordcts = [docs.docs[idx].counts for idx in testset[:int(len / 2.)]]

    for i in range(0,max_iter):
        minibatch = n.random.randint( int(docs.num_docs) - len, size=S)
        wordids = [docs.docs[idx].words for idx in minibatch]
        wordcts = [docs.docs[idx].counts for idx in minibatch]
        # learning lambda with training set
        model.update_lambda(wordids, wordcts)
        # learning local param gamma, fixing lambda with half of the test set
        (gamma, sstat ) = model.do_e_step(test1_wordids, test1_wordcts)
        # calculate the predictive probability with others of test set.
        predprob = pred_prob( test2_wordids, test2_wordcts, gamma, model._lambda)
        if i % S == 0:
            with open('predprob-%d.csv' % L, 'wa') as file:
                wr = csv.writer(file)
                wr.writerow([i, predprob])

if __name__ == '__main__':
    main()