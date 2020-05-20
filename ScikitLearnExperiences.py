#!/usr/bin/python3
# -*- coding: utf-8 -*-

###############################################################
#
#   Experiences in Machine Learning with Scikit-learn
#
#   Author: Laurane Castiaux
#
###############################################################

import pickle
import sklearn
import pandas as pd
import statistics as stats

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import VotingClassifier



stop_words_fr = "je tu il elle on nous vous ils elles le la les un une ne pas et à dans par pour en vers avec sans \
sous de chez durant sur comme pendant avant après abord y a est été mon ton son ma ta sa mes tes ses notre \
votre leur nos vos leurs ce ça ces cet cette ceci cela celle celui chacun chacune chaque ceux afin assez \
auquel auxquels auxquelles quelle que quoi quand quel quels qui aussi déjà ou où si".split()



def myProcessor(text):
    """Return text without "RT" if it is the beginning of text"""
    if re.match("RT .+", text) is not None:
        match = re.match("RT (.+)", text)
        group1 = match.group(1)
        return group1   
    else:
        return text



############ 1) Chargement du jeu de données   ############
def save(corpus_file, df_train, df_test, y_train, y_test):
    obj = {"df_train": df_train, "df_test": df_test, "y_train": y_train, "y_test":y_test}
    with open(corpus_file, "wb") as corpus:
        pickle.dump(obj, corpus)


def load(corpus_file):
    with open(corpus_file, "rb") as corpus: #rb = read binary
        obj = pickle.load(corpus)
    return ( obj["df_train"], obj["df_test"], obj["y_train"], obj["y_test"] )


#Lecture du fichier et découpage en sous-corpus
#df = pd.read_csv("all_tweets_param.csv", delimiter = ",", header = 0, encoding = "utf8")
#df_train, df_test, y_train, y_test = train_test_split(df, df["party"], test_size=0.20)
#df_train2, df_dev, y_train2, y_dev = train_test_split(df_train,y_train, test_size=0.20)


#save("sousCorpus.bin", df_train, df_test, y_train, y_test)
df_train, df_test, y_train, y_test = load("sousCorpus.bin")



############ 2) préparation des données   ############

# On entraîne l'espace vectoriel sur le train final
vectorizer = CountVectorizer(token_pattern=r"""(?xumsi)                                                                                                 
                                    (?:[lcdjmnts]|[a-z]*qu)['’]                                                                                                                        
                                    |https?:\/\/[^\s]+                                                                                                                                 
                                    |www\.[^\s]+                                                                                                                                       
                                    |[^\s]+@[^\s]+                                                                                                                                     
                                    |\#\w+                                                                                                                                             
                                    |\@\w+                                                                                                                                             
                                    |[rR]endez-vous                                                                                                                                    
                                    |[aA]ujourd'hui                                                                                                                                    
                                    |M(?:r)?\.                                                                                                                                         
                                    |(?<=\s)(?:[A-Z]{1}\.)+(?=[\s$])                                                                                                                   
                                    |(?:-t)?-(?:ils?|elles?|on)(?=\s)                                                                                                                  
                                    |(?:celui|celles?|ceux)-(?:ci|la|là)(?=\s)                                                                                                         
                                    |-(?:je|tu|moi|toi|nous|vous|la|les?|lui|leur|en|y|ce|ci|là)(?=\s)                                                                                 
                                    |\d+[.,]\d+                                                                                                                                        
                                    |[.-]+                                                                                                                                             
                                    |\w+                                                                                                                                               
                                    |[^\w\s]""",
                            stop_words=stop_words_fr)
                            #preprocessor = myProcessor)

vectorizer.fit(df_train["text"])



############ 3) Arbres de décision   ############
tree_clf = DecisionTreeClassifier(max_depth = None, min_samples_split=2)
tree_clf.fit(vectorizer.transform(df_train["text"]), y_train)

# Évaluation
print("\nArbre de décision :\n")
print("Accuracy : " + str(stats.mean(cross_val_score(tree_clf,
                                    vectorizer.transform(df_train['text']),
                                    y_train,
                                    cv=5,
                                    scoring="accuracy"))))
print("\n---------------------------------------\n")



############ 4) SMV   ############
svm_clf_final = LinearSVC(C=0.1, max_iter=10000)
svm_clf_final.fit(vectorizer.transform(df_train["text"]), y_train)

print("SVM:")
print("Accuracy : " + str(stats.mean(cross_val_score(svm_clf_final,
                                    vectorizer.transform(df_train['text']),
                                    y_train,
                                    cv=5,
                                    scoring="accuracy"))))
print("\n---------------------------------------\n")



############ 5) Multi-layer Perceptron Classifier   ############
mlp_clf = MLPClassifier(hidden_layer_sizes=(50,30,20),
                        max_iter=200,
                        alpha=1e-4,
                        solver='sgd',
                        verbose=False,
                        random_state=1,
                        learning_rate_init=.1)
mlp_clf.fit(vectorizer.transform(df_train["text"]), y_train)

print ("\nPerceptron :\n")
print("Accuracy : " + str(stats.mean(cross_val_score(mlp_clf,
                                    vectorizer.transform(df_train['text']),
                                    y_train,
                                    cv=5,
                                    scoring="accuracy"))))
print("\n---------------------------------------\n")



############ 6) KNN   ############

##### Grid Search
#param_grid = [ {'n_neighbors': [i for i in range(1, 20,2)],
#                'leaf_size' : [i for i in range(1, 102, 10)]}]
#KNN_clf = KNeighborsClassifier()
#                      
#KNN_grid = GridSearchCV(KNN_clf, param_grid, cv=10)
#KNN_grid.fit(vectorizer.transform(df_train["text"]), y_train)
#
#  # Best estimator
#print (KNN_grid.best_params_)
#####


KNN_clf = KNeighborsClassifier(leaf_size = 1,
                        n_neighbors = 5,
                        weights="distance")

KNN_clf.fit(vectorizer.transform(df_train["text"]), y_train)

print ("\nKNN :\n")
print("Accuracy : " + str(stats.mean(cross_val_score(KNN_clf,
                                    vectorizer.transform(df_train['text']),
                                    y_train,
                                    cv=5,
                                    scoring="accuracy"))))
print("\n---------------------------------------\n")



############ 7) Naive Bayes   ############

##### Grid Search
#param_grid = [ {'alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}]
#
#NB_clf = ComplementNB()
#                      
#NB_grid = GridSearchCV(NB_clf, param_grid, cv=10)
#NB_grid.fit(vectorizer.transform(df_train["text"]), y_train)
#
#  # Best estimator
#print (NB_grid.best_params_)
#####


NB_clf = ComplementNB(alpha = 0.5)

NB_clf.fit(vectorizer.transform(df_train["text"]), y_train)

print ("\nNB :\n")
print("Accuracy : " + str(stats.mean(cross_val_score(NB_clf,
                                    vectorizer.transform(df_train['text']),
                                    y_train,
                                    cv=5,
                                    scoring="accuracy"))))
print("\n---------------------------------------\n")



############ 8) VotingClassifier   ############

eclf2 = VotingClassifier(estimators = [("Arbre", tree_clf),
                                      ('SVM', svm_clf_final),
                                      ("Perceptron", mlp_clf),
                                      ("Naive Bayes", NB_clf),
                                      ("KNN", KNN_clf)],
                         voting = "hard")

print("VotingClassifier")
print("Accuracy : " + str(stats.mean(cross_val_score(eclf2,
                                    vectorizer.transform(df_train['text']),
                                    y_train,
                                    cv=5,
                                    scoring="accuracy"))))
print("\n---------------------------------------\n")



########### 9) Final evaluation on the test_corpus

#Découpage du corpus de développement pour en extraire une partie d'entraînement 
df_train2, df_dev, y_train2, y_dev = train_test_split(df_train,y_train, test_size=0.20)

eclf2.fit(vectorizer.transform(df_train2["text"]), y_train2)
y_pred_test = eclf2.predict(vectorizer.transform(df_test["text"]))

print("VotingClassifier - test corpus")
print("Accuracy : " + str(accuracy_score(y_test, y_pred_test)))
