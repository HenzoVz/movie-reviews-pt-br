from preprocessing.preprocessor import Preprocessor
from representation.representation import Representation
from machine_learning.classification import Classification
from evaluation.classifier_evaluation import Evaluation
import pandas as pd
import numpy as np


corpus = pd.read_csv('../data/imdb-reviews-pt-br.csv')
corpus_tokens = Preprocessor.corpus(corpus, 'text_pt', 'sentiment')
feature_tokens = [[j for j in i[0]] for i in corpus_tokens]
labels = [i[1] for i in corpus_tokens]
labels = np.array(labels)
print("lenght corpus = ", len(feature_tokens))
print("lenght target = ", len(labels))

tfidf_bow,  dictionary = Representation.repres_tfidf(feature_tokens)
print(dictionary)
corpus_representation = Representation.represent_tfidf_bow_with_dictionary(tfidf_bow, dictionary)
print("Lenght representation = ", len(corpus_representation))


model = Classification.get_model_by_algorithm_id(corpus_representation, labels, Classification.KNN)
print(model)
score = Evaluation.evaluation_model(model, corpus_representation, labels)

print("Scores")
print(score)