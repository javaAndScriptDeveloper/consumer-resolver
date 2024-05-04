from confluent_kafka import Consumer

class DataConsumer:

    def __init__(self, kafka_config, topic_name, process_function):
        self.process_function = process_function
        self.consumer = Consumer(kafka_config)
        self.consumer.subscribe([topic_name])

    def consume(self):
        while True:
            msg = self.consumer.poll(1.0)

            if msg is None:
                continue
            if msg.error():
                print("Consumer error: {}".format(msg.error()))
                continue
            decoded_msg = msg.value().decode('utf-8')
            print('Received message: {}'.format(decoded_msg))
            self.process_function(decoded_msg)

    def close(self):
        self.consumer.close()
