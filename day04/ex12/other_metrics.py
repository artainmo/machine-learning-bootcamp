import numpy as np

def binary_predict(predicted_values):
    i = 0
    while i < predicted_values.shape[0]:
        if predicted_values[i] < 0.5:
            predicted_values[i] = 0
        else:
            predicted_values[i] = 1
        i += 1
    return predicted_values

def positives_negatives(expected_values, predicted_values, class_):
    data = {"true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0}
    # expected_values = binary_predict(predicted_values)
    for expected_value, predicted_value in zip(expected_values, predicted_values):
        if expected_value == predicted_value:
            if expected_value != class_:
                data["true_negative"] += 1
            else:
                data["true_positive"] += 1
        else:
            if predicted_value != class_:
                data["false_negative"] += 1
            else:
                data["false_positive"] += 1
    return data



def accuracy_score_(expected_values, predicted_values, class_=1):
    data = positives_negatives(expected_values, predicted_values, class_)
    return (data["true_positive"] + data["true_negative"]) / (data["true_positive"] + data["false_positive"] + data["true_negative"] + data["false_negative"])


def precision_score_(expected_values, predicted_values, class_=1):
    data = positives_negatives(expected_values, predicted_values, class_)
    return (data["true_positive"]) / (data["true_positive"] + data["false_positive"])


def recall_score_(expected_values, predicted_values, class_=1):
    data = positives_negatives(expected_values, predicted_values, class_)
    return (data["true_positive"]) / (data["true_positive"] + data["false_negative"])


def f1_score_(expected_values, predicted_values, class_=1):
    precision_score = precision_score_(expected_values, predicted_values, class_)
    recall_score = recall_score_(expected_values, predicted_values, class_)
    return (2 * precision_score * recall_score) / (precision_score + recall_score)
