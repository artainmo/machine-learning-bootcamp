import numpy as np
import pandas as pd

def get_labels(predicted_values):
    labels = []
    for item in predicted_values:
        if item not in labels:
            labels.append(item)
    return labels[::-1]


def comparison(expected_values, predicted_values, label, labels):
    counter = np.zeros(len(labels))
    for expected_value, predicted_value, in zip(expected_values, predicted_values):
        if expected_value == label:
            try:
                i = labels.index(predicted_value)
                counter[i] += 1
            except:
                pass
    return counter




def confusion_matrix_(expected_values, predicted_values, labels=None, df_option=None):
    confusion_matrix = []
    if labels == None:
        labels = get_labels(predicted_values)
    for label in labels:
         confusion_matrix.append(comparison(expected_values, predicted_values, label, labels))
    if df_option == None:
        return np.array(confusion_matrix)
    else:
        return pd.DataFrame(confusion_matrix, index=labels, columns=labels)


y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'bird'])
y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet'])
print(confusion_matrix_(y, y_hat))
print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet']))
print(confusion_matrix_(y, y_hat, labels=['bird', 'dog'], df_option=True))
