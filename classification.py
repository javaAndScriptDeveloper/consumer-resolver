import abc

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import config
from enums import DATA_CLASSIFIER_METHOD_TYPE


def get_classifier():

    if(config.DATA_CLASSIFIER_METHOD == DATA_CLASSIFIER_METHOD_TYPE.K_NEAREST_NEIGHBOURS):
        return DataClassifierKnn.get_instance()
    if(config.DATA_CLASSIFIER_METHOD == DATA_CLASSIFIER_METHOD_TYPE.LOGISTIC_REGRESSION):
        return DataClassifierLogisticRegression.get_instance()
    if (config.DATA_CLASSIFIER_METHOD == DATA_CLASSIFIER_METHOD_TYPE.DECISION_TREE):
        return DataClassifierDecisionTree.get_instance()
    if (config.DATA_CLASSIFIER_METHOD == DATA_CLASSIFIER_METHOD_TYPE.LINEAR_DISCRIMINANT):
        return DataClassifierLinearDiscriminant.get_instance()
    if (config.DATA_CLASSIFIER_METHOD == DATA_CLASSIFIER_METHOD_TYPE.GAUSSIAN_NAIVE_BAYES):
        return DataClassifierGaussianNaiveBayes.get_instance()
    if (config.DATA_CLASSIFIER_METHOD == DATA_CLASSIFIER_METHOD_TYPE.SUPPORT_VECTOR_MACHINE):
        return DataClassifierSupportVectorMachine.get_instance()

    raise Exception(f"Data classifier {config.DATA_CLASSIFIER_METHOD} is not supported")

class DataClassifierAbc(abc.ABC):

    def serve(self, data_dict):
        pass

    @staticmethod
    def _generate_instance():
        pass

    @classmethod
    def get_instance(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = cls._generate_instance()
        return cls._instance


class DataClassifierKnn(DataClassifierAbc):

    def __init__(self, properties):
        self.model = None
        self.properties = properties

    def serve(self, data_dict):
        scaled_properties = self.scaler.transform([data_dict])
        return self.model.predict(scaled_properties)[0]

    def _train(self, data_path, type_property_name):

        data = pd.read_table(data_path)

        X = data[self.properties]
        y = data[type_property_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model = KNeighborsClassifier()
        self.model.fit(X_train, y_train)

        print('Accuracy of K-NN classifier on training set: {:.2f}'.format(self.model.score(X_train, y_train)))
        print('Accuracy of K-NN classifier on test set: {:.2f}'.format(self.model.score(X_test, y_test)))

    @staticmethod
    def _generate_instance():
        classifier = DataClassifierKnn(config.CLASSIFICATION_PROPERTY_NAMES)
        classifier._train(config.ROOT_DIR + config.CLASSIFICATION_TRAINING_DATASET_PATH, config.TYPE_PROPERTY_NAME)
        return classifier

class DataClassifierLogisticRegression(DataClassifierAbc):

    def __init__(self, properties):
        self.model = None
        self.properties = properties

    def serve(self, data_dict):
        scaled_properties = self.scaler.transform([data_dict])
        return self.model.predict(scaled_properties)[0]

    def _train(self, data_path, type_property_name):

        data = pd.read_table(data_path)

        X = data[self.properties]
        y = data[type_property_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)
        print('Accuracy of Logistic regression classifier on training set: {:.2f}'
              .format(self.model.score(X_train, y_train)))
        print('Accuracy of Logistic regression classifier on test set: {:.2f}'
              .format(self.model.score(X_test, y_test)))

    @staticmethod
    def _generate_instance():
        classifier = DataClassifierLogisticRegression(config.CLASSIFICATION_PROPERTY_NAMES)
        classifier._train(config.ROOT_DIR + config.CLASSIFICATION_TRAINING_DATASET_PATH, config.TYPE_PROPERTY_NAME)
        return classifier

class DataClassifierDecisionTree(DataClassifierAbc):

    def __init__(self, properties):
        self.model = None
        self.properties = properties

    def serve(self, data_dict):
        scaled_properties = self.scaler.transform([data_dict])
        return self.model.predict(scaled_properties)[0]

    def _train(self, data_path, type_property_name):

        data = pd.read_table(data_path)

        X = data[self.properties]
        y = data[type_property_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model = DecisionTreeClassifier().fit(X_train, y_train)
        print('Accuracy of Decision Tree classifier on training set: {:.2f}'
              .format(self.model.score(X_train, y_train)))
        print('Accuracy of Decision Tree classifier on test set: {:.2f}'
              .format(self.model.score(X_test, y_test)))

    @staticmethod
    def _generate_instance():
        classifier = DataClassifierDecisionTree(config.CLASSIFICATION_PROPERTY_NAMES)
        classifier._train(config.ROOT_DIR + config.CLASSIFICATION_TRAINING_DATASET_PATH, config.TYPE_PROPERTY_NAME)
        return classifier

class DataClassifierLinearDiscriminant(DataClassifierAbc):

    def __init__(self, properties):
        self.model = None
        self.properties = properties

    def serve(self, data_dict):
        scaled_properties = self.scaler.transform([data_dict])
        return self.model.predict(scaled_properties)[0]

    def _train(self, data_path, type_property_name):

        data = pd.read_table(data_path)

        X = data[self.properties]
        y = data[type_property_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model = LinearDiscriminantAnalysis()
        self.model.fit(X_train, y_train)
        print('Accuracy of LDA classifier on training set: {:.2f}'
              .format(self.model.score(X_train, y_train)))
        print('Accuracy of LDA classifier on test set: {:.2f}'
              .format(self.model.score(X_test, y_test)))

    @staticmethod
    def _generate_instance():
        classifier = DataClassifierLinearDiscriminant(config.CLASSIFICATION_PROPERTY_NAMES)
        classifier._train(config.ROOT_DIR + config.CLASSIFICATION_TRAINING_DATASET_PATH, config.TYPE_PROPERTY_NAME)
        return classifier

class DataClassifierGaussianNaiveBayes(DataClassifierAbc):

    def __init__(self, properties):
        self.model = None
        self.properties = properties

    def serve(self, data_dict):
        scaled_properties = self.scaler.transform([data_dict])
        return self.model.predict(scaled_properties)[0]

    def _train(self, data_path, type_property_name):

        data = pd.read_table(data_path)

        X = data[self.properties]
        y = data[type_property_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model = GaussianNB()
        self.model .fit(X_train, y_train)
        print('Accuracy of GNB classifier on training set: {:.2f}'
              .format(self.model .score(X_train, y_train)))
        print('Accuracy of GNB classifier on test set: {:.2f}'
              .format(self.model .score(X_test, y_test)))

    @staticmethod
    def _generate_instance():
        classifier = DataClassifierGaussianNaiveBayes(config.CLASSIFICATION_PROPERTY_NAMES)
        classifier._train(config.ROOT_DIR + config.CLASSIFICATION_TRAINING_DATASET_PATH, config.TYPE_PROPERTY_NAME)
        return classifier

class DataClassifierSupportVectorMachine(DataClassifierAbc):

    def __init__(self, properties):
        self.model = None
        self.properties = properties

    def serve(self, data_dict):
        scaled_properties = self.scaler.transform([data_dict])
        return self.model.predict(scaled_properties)[0]

    def _train(self, data_path, type_property_name):

        data = pd.read_table(data_path)

        X = data[self.properties]
        y = data[type_property_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model = SVC()
        self.model.fit(X_train, y_train)
        print('Accuracy of SVM classifier on training set: {:.2f}'
              .format(self.model.score(X_train, y_train)))
        print('Accuracy of SVM classifier on test set: {:.2f}'
              .format(self.model.score(X_test, y_test)))

    @staticmethod
    def _generate_instance():
        classifier = DataClassifierSupportVectorMachine(config.CLASSIFICATION_PROPERTY_NAMES)
        classifier._train(config.ROOT_DIR + config.CLASSIFICATION_TRAINING_DATASET_PATH, config.TYPE_PROPERTY_NAME)
        return classifier
