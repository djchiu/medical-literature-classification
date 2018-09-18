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
import matplotlib.pyplot as plt
from random import shuffle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn import cross_validation
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
import tensorflow as tf

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def tovector(words):
    vector_array = []
    for w in words:
        try:
            vector_array.append(model[w])
        except:
            continue
    vector_array = np.array(vector_array)
    v = vector_array.sum(axis=0)
    result = v / np.sqrt((v ** 2).sum())
    # print( result
    return result

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

def preprocessing(text, gene_dict, cancer_dict):
    # tokenize into words
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    # remove stopwords
    stop = stopwords.words('english')

    temp = []
    for word in tokens:
        if word not in stop and len(word) >= 3 and not is_number(word):
           try:
              ########## replace GENE ##################
              word = word.replace(word, gene_dict[word])
              word = word.replace(word, 'gene')
              ##########################################
              temp.append(word.lower())
           except KeyError:
              temp.append(word.lower())

    tokens = temp

    # lemmatize
    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(word) for word in tokens]
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text


# preparing ANN
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def main():

    # abstract number  3830
    abstract_size = 3000
    # gene replacement list size in gene.csv
    gene_size = 70
    cancer_size = 341
    batch_size = 10
    training_steps = 100000

    print( '')
    print( "--- Loading data ---")
    dataset = pd.read_csv('abstract.csv', sep=',', encoding='utf-8')
    dataset = dataset.dropna(subset=['abstract_text', 'penetrance', 'incidence']) # remove nan column
    texts = dataset.ix[0:abstract_size, 3].values.tolist()

    # ix[:, 8] = penetrance, ix[:, 9] = incidence
    label = dataset.ix[0:abstract_size, 9].values.tolist()

    y = []
    for i in label:
        y.append(int(i))
    y = np.array(y)
    print( "Read %d rows of data" % len(texts))
    print( "Read %d rows of label" % len(label))

   #### load gene.csv as the disctionary to replace the gene name 
    gene_dataset = pd.read_csv('gene.csv', sep=',', encoding='latin-1')
    gene_name = gene_dataset.ix[0:gene_size, 0].values.tolist()
    gene_replace = gene_dataset.ix[0:gene_size, 9].values.tolist()
    gene_dict = {}
    for i in range(len(gene_name)):
        gene_dict[gene_name[i]]=gene_replace[i]

####################################################################################

    print( '')
    print( "--- Text processing ---")

    doc_vec = []
    for line in texts:
        list = preprocessing(line, gene_dict, cancer_dict).split()
        doc_vec.append(tovector(list))

    X = np.array(doc_vec)

################## Traditional ML model #####################################

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
  

    k=5
    for a in alg_list:
        skf = StratifiedKFold(y, n_folds=k, shuffle=True, random_state=None)
        acc_total = 0.0
        pr_total = 0.0
        re_total = 0.0
        f1_total = 0.0
        auc_total = 0.0
        print( ' algorithm result :' + str(a))
        for idx, (train, test) in enumerate(skf):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            clf = OneVsRestClassifier(a).fit(X_train, y_train)
            # scores = cross_validation.cross_val_score(clf, np.array(doc_vec), y, cv=5)
            # print( 'scores :' + str(scores)
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
        # print( 'y_pred :' + str(y_pred)
        print( '\nacc :' + str(acc_total/k)+'\npr :' + str(pr_total/k)+'\nre :' + str(re_total/k)+'\nf1 :' + str(f1_total/k)+'\nauc :' + str(auc_total/k))
        print( '###############################################################')

######################## DNN model #####################
    input_size = X.shape[1]
    N_class = 2
    x = tf.placeholder(tf.float32, [None, input_size])
    y_ = tf.placeholder(tf.float32, [None, N_class])

    keep_prob = tf.placeholder(tf.float32)

    W_1 = weight_variable([input_size, 100])
    b_1 = bias_variable([100])

    h_fc1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)

    W_fc2 = weight_variable([100, N_class])
    b_fc2 = bias_variable([N_class])

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    y_conv = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    skf = StratifiedKFold(y, n_folds=k, shuffle=True, random_state=None)
    acc_total_DNN = 0.0
    pr_total_DNN = 0.0
    re_total_DNN = 0.0
    f1_total_DNN = 0.0
    auc_total_DNN = 0.0
    print(y[:10])
    y_DNN = np.zeros([y.shape[0],2])
    for i in range(y.shape[0]):
        if y[i] == 0:
            y_DNN[i,0] = 1
        else:
            y_DNN[i,1] = 1
         
    for idx, (train, test) in enumerate(skf):
        X_train, X_test = X[train], X[test]
        y_train_DNN, y_test_DNN = y_DNN[train], y_DNN[test]
        print(X_train.shape)
        print(X_test.shape)
        print(y_train_DNN.shape)
        print(y_test_DNN.shape)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        train_input = X[train]
        train_label = y_DNN[train]
        test_input = X[test]
        test_label = y_DNN[test]
        for i in range(training_steps): 
            batch_ind = np.random.randint(0,train_input.shape[0],batch_size)
            batch = (train_input[batch_ind,:],train_label[batch_ind,:])
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
##            if i%100 == 0:
##                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
##                print("step %d, training accuracy %g"%(i, train_accuracy))
##                print("test accuracy %g"%accuracy.eval(feed_dict={x: test_input, y_: test_label, keep_prob: 1.0}))          
        DNN_pred = sess.run(tf.argmax(y_conv,1), feed_dict={x: X[test], y_: y_DNN[test], keep_prob: 1.0})
        print(DNN_pred)
        acc_DNN = accuracy_score(y[test], DNN_pred)
        pr_DNN, re_DNN, f1_DNN, xx_DNN = precision_recall_fscore_support(y[test], DNN_pred, average='weighted')
        auc_DNN = multiclass_roc_auc_score(y[test], DNN_pred, average='weighted')
        acc_total_DNN += auc_DNN
        pr_total_DNN += pr_DNN
        re_total_DNN += re_DNN
        f1_total_DNN += f1_DNN
        auc_total_DNN += auc_DNN

    print( '\nacc :' + str(acc_total_DNN/k)+'\npr :' + str(pr_total_DNN/k)+'\nre :' + str(re_total_DNN/k)+'\nf1 :' + str(f1_total_DNN/k)+'\nauc :' + str(auc_total_DNN/k))
    print( '###############################################################')

 

if __name__ == "__main__":
    main()
