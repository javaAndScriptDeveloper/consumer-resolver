import os
import random

from enums import DATA_CLASSIFIER_METHOD_TYPE

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CLASSIFICATION_TRAINING_DATASET_PATH = '/data/fruit_data_with_colors.txt'

TYPE_PROPERTY_NAME = 'fruit_name'
CLASSIFICATION_PROPERTY_NAMES = ['mass', 'width', 'height', 'color_score']

TYPE_ACTION_DICT = {
    'apple': lambda x: print('apple = ' + str(x)),
    'mandarin': lambda x: print('mandarin = ' + str(x))
}

KAFKA_CONFIG = {
            'bootstrap.servers': 'localhost:9092',
            'group.id': 'mygroup' + str(random.randrange(0, 10000)),
            'auto.offset.reset': 'earliest',
            'security.protocol': 'PLAINTEXT'
}

KAFKA_TOPIC_NAME = 'fruits'

DATA_CLASSIFIER_METHOD = DATA_CLASSIFIER_METHOD_TYPE.K_NEAREST_NEIGHBOURS
