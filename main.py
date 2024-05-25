import json

import config
from classification import get_classifier
from kafka_messaging.consumer import DataConsumer

def process_data(data_json):
    data_dict = json.loads(data_json)

    data_list = [data_dict[i] for i in config.CLASSIFICATION_PROPERTY_NAMES]

    data_type = get_classifier().serve(data_list)

    action = config.TYPE_ACTION_DICT.get(data_type)
    if action is not None:
        action(data_dict)

if __name__ == '__main__':
    data_consumer = DataConsumer(config.KAFKA_CONFIG, config.KAFKA_TOPIC_NAME, process_data)
    data_consumer.consume()
