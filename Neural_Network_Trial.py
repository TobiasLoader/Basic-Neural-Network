
from math import *
from random import *


def sigmoid(x):
    return 1 / (1 + exp(-x))


def training():
    training_num = int(input("Number of training examples: "))
    inner_repeat_num = training_num
    while inner_repeat_num >= training_num:
        inner_repeat_num = int(input("Number of samples per example: "))
    for i in range(training_num):
        x = uniform(-1, 1)
        y = uniform(-1, 1)
        inputs.append([[x, y]])
        if x >= y:
            inputs[i].append([1, 0])
        else:
            inputs[i].append([0, 1])

    new_weights = [0, 0, 0, 0]

    for i in range(len(inputs)):
        values = inputs[i][0]
        aim = inputs[i][1]
        temp = [0, 0]

        temp[0] = weights[0] * values[0] + weights[2] * values[1]
        temp[1] = weights[1] * values[0] + weights[3] * values[1]
        new_vals = temp

        c[0] = (new_vals[0]-aim[0])**2
        c[1] = (new_vals[1]-aim[1])**2

        new_weights[0] += c[0] * values[0]
        new_weights[1] += c[1] * values[0]
        new_weights[2] += c[0] * values[1]
        new_weights[3] += c[1] * values[1]
        values = new_vals

        if not i % inner_repeat_num and i:
            avg = 0
            for k in range(len(weights)):
                weights[k] += sigmoid(new_weights[k]/inner_repeat_num)
                avg += weights[k]
            avg /= 4
            for k in range(len(weights)):
                weights[k] -= avg
            new_weights = [0, 0, 0, 0]

    return values, weights


def trialling():
    trial_num = int(input("Number of actual trials after training: "))
    trials = []
    correct_num = 0
    for i in range(trial_num):
        x = uniform(-1, 1)
        y = uniform(-1, 1)
        trials.append([[x, y]])
        if x >= y:
            trials[i].append([1, 0])
        else:
            trials[i].append([0, 1])

    for i in range(trial_num):
        trial = trials[i][0]
        values[0] = weights[0] * trial[0] + weights[2] * trial[1]
        values[1] = weights[1] * trial[0] + weights[3] * trial[1]
        if (values[0] > values[1] and not trials[i][1][1]) or (values[0] < values[1] and not trials[i][1][0]):
            correct_num += 1
    print('\nMy neural network was correct', 100 * correct_num / trial_num, '% of the time.')


play = True
while play:
    inputs = []
    weights = [0, 0, 0, 0]
    c = [0, 0]
    values, weights = training()
    trialling()
    print("Final weights:",weights,"\n")
    if input("Again [Y/n]: ") not in ["Y","y"]:
        play = False
    print("")
