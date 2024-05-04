import json

from confluent_kafka import Producer

producer = Producer({
'bootstrap.servers': 'localhost:9092',
    'security.protocol': 'PLAINTEXT'
}
)


class Fruit:
    def __init__(self, fruit_label, fruit_name, fruit_subtype, mass, width, height, color_score):
        self.fruit_label = fruit_label
        self.fruit_name = fruit_name
        self.fruit_subtype = fruit_subtype
        self.mass = mass
        self.width = width
        self.height = height
        self.color_score = color_score

    def to_dict(self):
        return {key: value for key, value in vars(self).items()}

# List of objects
fruits_objects = [
    Fruit(1, 'lemon', 'granny_smith',116,	5.9,	8.1,	0.73),
]

for i in fruits_objects:
    producer.produce("fruits", value=json.dumps(i.to_dict()))

producer.flush()