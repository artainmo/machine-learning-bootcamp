import numpy as np
from statistics import *


def MSE_cost_function(predicted_values, expected_values):
    i = 0
    summation = 0
    lenght = len(predicted_values)
    while i < lenght:
        summation += (predicted_values[i] - expected_values[i]) ** 2
        i += 1
    return summation / lenght

def linear_regression_cost_function(predicted_values, expected_values):
    return MSE_cost_function(predicted_values, expected_values) / 2

def RMSE_cost_function(predicted_values, expected_values):
    return MSE_cost_function(predicted_values, expected_values) ** 0.5

def MAE_cost_function(predicted_values, expected_values):
    i = 0
    summation = 0
    lenght = len(predicted_values)
    while i < lenght:
        summation += (predicted_values[i] - expected_values[i])
        i += 1
    return summation / lenght


def R2_cost_function(predicted_values, expected_values):
    i = 0
    summation = 0
    lenght = len(predicted_values)
    mean = mean(expected_values)
    while i < lenght:
        summation += (predicted_values[i] - mean) ** 2
        i += 1
    res = summation / lenght
    return 1 - (MSE_cost_function(predicted_values, expected_values) / res)
