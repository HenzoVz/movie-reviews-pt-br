from sklearn.model_selection import cross_validate
import numpy as np

class Evaluation:

    def evaluation_model(model, corpus_representation, labels):
        SCORING_COLUMNS = ['accuracy']
        score = cross_validate(model, corpus_representation, labels, scoring=SCORING_COLUMNS, cv=10)
        return np.mean(score['test_accuracy'])