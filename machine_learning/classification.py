from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


class Classification:

    MULTINOMIAL = 0
    NAIVE_BAYES = 1
    KNN = 2
    SVM = 3
    DECISION_TREE = 4
    RANDOM_FOREST = 5
    ADA_BOOST = 6

    def get_model_by_algorithm_id(X_train, Y_train, algorithm):

        try:
            if algorithm == Classification.NAIVE_BAYES:
                return Classification.train_model_by_naive(X_train, Y_train)
            elif algorithm == Classification.KNN:
                return Classification.train_model_by_knn(X_train, Y_train)
            elif algorithm == Classification.DECISION_TREE:
                return Classification.train_model_by_decision_tree(X_train, Y_train)
            elif algorithm == Classification.RANDOM_FOREST:
                return Classification.train_model_by_random_forest(X_train, Y_train)
            elif algorithm == Classification.ADA_BOOST:
                return Classification.train_model_by_ada_boost(X_train, Y_train)
            elif algorithm == Classification.MULTINOMIAL:
                return Classification.train_model_by_multinomial(X_train, Y_train)
            elif algorithm == Classification.SVM:
                return Classification.train_model_by_svm(X_train, Y_train)
        except algorithm:
            print("Undefined algorithm :", algorithm)

    def train_model_by_multinomial(X_train, Y_train):
        model = MultinomialNB()
        model.fit(X_train, Y_train)
        return model

    def train_model_by_naive(X_train, Y_train):
        model = GaussianNB()
        model.fit(X_train, Y_train)
        return model

    def train_model_by_knn(X_train, Y_train):
        model = KNeighborsClassifier()
        model.fit(X_train, Y_train)
        return model

    def train_model_by_svm(X_train, Y_train):
        model = SVC()
        model.fit(X_train, Y_train)
        return model

    def train_model_by_decision_tree(X_train, Y_train):
        model = DecisionTreeClassifier()
        model.fit(X_train, Y_train)
        return model

    def train_model_by_random_forest(X_train, Y_train):
        model = RandomForestClassifier(max_depth=10, n_estimators=50)
        model.fit(X_train, Y_train)
        return model

    def train_model_by_ada_boost(X_train, Y_train):
        model = AdaBoostClassifier(n_estimators=50)
        model.fit(X_train, Y_train)
        return model