import sys
import os
import re
import string
import logging
import numpy as np
import pandas as pd
import nltk
import collections
import gensim
from random import shuffle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import KFold, StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.multiclass import OneVsRestClassifier, OutputCodeClassifier
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def multiclass_roc_auc_score(y_true, y_pred, average='macro'):
    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true = lb.transform(y_true)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_true, y_pred, average="weighted")

def preprocessing(text, gene_dict):
    # tokenize into words
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    # remove stopwords
    stop = stopwords.words('english')
    # # remove words less than three letters, remove float number, lower capitalization
    temp = []
    for word in tokens:
        if word not in stop and len(word) >= 3 and not is_number(word):
           try:
              word = word.replace(word, gene_dict[word])
              word = word.replace(word, 'gene')
              temp.append(word.lower())
           except KeyError:
              temp.append(word.lower())

    tokens = temp

    # lemmatize
    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def main():
    # abstract number  3830
    abstract_size = 3830
    # gene replacement list size in gene.csv
    gene_size = 80
    print ''
    print "--- Loading data ---"
    dataset = pd.read_csv('abstract.csv', sep=',', encoding='utf-8')
    dataset = dataset.dropna(subset=['abstract_text', 'penetrance', 'incidence']) # remove nan column
    texts = dataset.ix[0:abstract_size, 3].values.tolist()

    # ix[:, 8] = penetrance, ix[:, 9] = incidence
    label = dataset.ix[0:abstract_size, 9].values.tolist()

    y = []
    for i in label:
        y.append(int(i))
    y=np.array(y)
    print "Read %d rows of data" % len(texts)
    print "Read %d rows of label" % len(label)

    #### load gene.csv as the disctionary to replace the gene name 
    gene_dataset = pd.read_csv('gene.csv', sep=',', encoding='latin-1')
    gene_name = gene_dataset.ix[0:gene_size, 0].values.tolist()
    gene_replace = gene_dataset.ix[0:gene_size, 9].values.tolist()
    gene_dict = {}
    for i in range(len(gene_name)):
        gene_dict[gene_name[i]]=gene_replace[i]

    print ''
    print "--- Text processing ---"
    sms_exp = []
    for line in texts:
        sms_exp.append(preprocessing(line, gene_dict))

### Term-frequency 

    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(min_df = 1, encoding='utf-8')
    X = vectorizer.fit_transform(sms_exp)
    X = X.toarray()

### Tf-idf 
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # vectorizer = TfidfVectorizer(min_df = 2, ngram_range=(1, 2),
    #                              stop_words = 'english', strip_accents = 'unicode', norm = 'l2')
    #
    # X = vectorizer.fit_transform(sms_exp)
    # X = X.toarray()

### machine learning algorithm
    alg_list = [\
        LogisticRegression(penalty='l1', multi_class='ovr', class_weight=None, n_jobs=-1), \
        LogisticRegression(penalty='l2', multi_class='ovr', class_weight=None, n_jobs=-1), \
        #SVC(kernel='linear', probability=True, decision_function_shape='ovr', class_weight=class_wt), \
        CalibratedClassifierCV(base_estimator=LinearSVC(penalty='l2', loss='squared_hinge', C=1.0, multi_class='ovr', class_weight=None, random_state=0, max_iter=1000), cv=5), \
        CalibratedClassifierCV(base_estimator=SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True, epsilon=0.1, n_jobs=-1, random_state=0, learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None), cv=5), \
        # SVC(kernel='rbf', probability=True, decision_function_shape='ovr', class_weight=None), \
        RandomForestClassifier(n_estimators=100, class_weight=None, n_jobs=-1), \
        AdaBoostClassifier(n_estimators=100), \
        #GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)] # GBM can't use sparse matrix, scikit-learn problem
        MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(128, 64, 32), random_state=42), \
        ]
    ### k-fold cross validation
    k=5
    for a in alg_list:
        skf = StratifiedKFold(y, n_folds=k, shuffle=True, random_state=None)
        acc_total = 0.0
        pr_total = 0.0
        re_total = 0.0
        f1_total = 0.0
        auc_total = 0.0
        print ' algorithm result :' + str(a)
        for idx, (train, test) in enumerate(skf):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            clf = OneVsRestClassifier(a).fit(X_train, y_train)
            y_pred_prob = clf.predict_proba(X_test)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            pr, re, f1, xx = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            auc = multiclass_roc_auc_score(y_test, y_pred, average='weighted')
            acc_total += acc
            pr_total += pr
            re_total += re
            f1_total += f1
            auc_total += auc
        # print 'y_pred :' + str(y_pred)
        print '\nacc :' + str(acc_total/k)+'\npr :' + str(pr_total/k)+'\nre :' + str(re_total/k)+'\nf1 :' + str(f1_total/k)+'\nauc :' + str(auc_total/k)
        print '###############################################################'

if __name__ == "__main__":
    main()
