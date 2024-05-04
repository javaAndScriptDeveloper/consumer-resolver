import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

import config


class DataClassifier:

    def __init__(self, properties):
        self.model = None
        self.properties = properties

    def serve(self, data_model):
        scaled_properties = self.scaler.transform([data_model])
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
        classifier = DataClassifier(config.CLASSIFICATION_PROPERTY_NAMES)
        classifier._train(config.ROOT_DIR + config.CLASSIFICATION_TRAINING_DATASET_PATH, config.TYPE_PROPERTY_NAME)
        return classifier

    @staticmethod
    def get_instance():
        if not hasattr(DataClassifier, "_instance"):
            DataClassifier._instance = DataClassifier._generate_instance()
        return DataClassifier._instance


