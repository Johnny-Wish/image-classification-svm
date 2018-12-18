import os
import heapq
import argparse
import pandas as pd
import pickle as pkl
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection._search import BaseSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from support_vector_machine import svc, param_dist
from dataset import ImageDataset


def print_cv_result(result, n=-1):
    if isinstance(result, BaseSearchCV):
        result = result.cv_results_

    scores = result['mean_test_score']
    params = result['params']

    if n < 0:
        n = len(scores)

    print("Cross Validation result in descending order: (totalling {} trials)".format(n))
    for rank, candidate, in enumerate(heapq.nlargest(n, zip(scores, params), key=lambda tup: tup[0])):
        print("rank {}, score = {}\n hyperparams = {}".format(rank + 1, *candidate))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train0", required=True, help="folder of negative samples for training")
    parser.add_argument("--test0", required=True, help="folder of negative samples for testing")
    parser.add_argument("--train1", required=True, help="folder of positive samples for training")
    parser.add_argument("--test1", required=True, help="folder of positive samples for testing")
    parser.add_argument("--outf", default="output/", help=)
    parser.add_argument("--n_iter", default=20, type=int, help="number of search iterations")
    parser.add_argument("--cv", default=3, type=int, help="number of folds for cross validation")
    parser.add_argument("--n_jobs", default=-1, type=int, help="number of CPU workers")
    opt = parser.parse_args()

    dataset = ImageDataset(opt.train0, opt.test0, opt.train1, opt.test1k)
    searcher = RandomizedSearchCV(svc, param_dist, n_iter=opt.n_iter, cv=opt.cv, n_jobs=opt.n_jobs,
                                  verbose=2, random_state=0, return_train_score=True)
    print("starting hypertuning")
    searcher.fit(dataset.train.X, dataset.train.y)

    print("best model: ", searcher.best_index_, searcher.best_score_, searcher.best_params_)
    print_cv_result(searcher.cv_results_)

    print("testing best model")
    searcher.best_estimator_.fit(dataset.train.X, dataset.train.y)
    y_true = dataset.test.y
    y_pred = searcher.best_estimator_.predict(dataset.test.X)
    print("test accuracy = {}, precision = {}, recall = {}, f1-score = {}, support = {}".format(
        accuracy_score(y_true, y_pred), *precision_recall_fscore_support(y_true, y_pred)
    ))

    print("dumping search results")
    with open(os.path.join(opt.outf, "search_results.pkl"), "wb") as f:
        pkl.dump(pd.DataFrame(searcher.cv_results_))

    print("dumping best model")
    with open(os.path.join(opt.outf, "model.pkl"), "wb") as f:
        pkl.dump(pd.DataFrame(searcher.best_estimator_))
