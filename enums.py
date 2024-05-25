from enum import Enum, auto

class DATA_CLASSIFIER_METHOD_TYPE(Enum):
    K_NEAREST_NEIGHBOURS = auto(),
    LOGISTIC_REGRESSION = auto(),
    DECISION_TREE = auto(),
    LINEAR_DISCRIMINANT = auto(),
    GAUSSIAN_NAIVE_BAYES = auto(),
    SUPPORT_VECTOR_MACHINE = auto()
