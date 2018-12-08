from math import *
from random import *
from sys import exit


def sigmoid(x):
    return 1 / (1 + exp(-x))


def wrong_input():
    print("\nOh no!\nYour input is not compatible with our code!\nSorry...")
    exit()


def cost_output_node(i, final_node, aim):
    c[i] = (final_node[i] - aim[i]) ** 2


def round_n_to_rdp(n,r):
    return round((10**r)*n)/(10**r)


def training():
    global training_num, values
    try:
        training_num = int(input("\nNumber of training examples: "))
    except:
        wrong_input()
    inner_repeat_num = training_num
    while inner_repeat_num >= training_num:
        print("\nJust sayin', the number of samples must be less than the number of training examples...")
        try:
            inner_repeat_num = int(input("Number of samples per example: "))
        except:
            wrong_input()
    for i in range(training_num):
        x = uniform(-1, 1)
        y = uniform(-1, 1)
        inputs.append([[x, y]])
        if x >= y:
            inputs[i].append([1, 0])
        else:
            inputs[i].append([0, 1])

    new_w = [[0, 0, 0, 0]]

    for i in range(len(inputs)):
        n = [inputs[i][0], [0, 0]]
        final_node_temp = [0, 0]

        final_node_temp[0] = w[0][0] * n[0][0] + w[0][2] * n[0][1]
        final_node_temp[1] = w[0][1] * n[0][0] + w[0][3] * n[0][1]

        cost_output_node(0, final_node_temp, inputs[i][1])
        cost_output_node(1, final_node_temp, inputs[i][1])

        new_w[0][0] += c[0] * n[0][0]
        new_w[0][1] += c[1] * n[0][0]
        new_w[0][2] += c[0] * n[0][1]
        new_w[0][3] += c[1] * n[0][1]
        n[1] = final_node_temp

        if not i % inner_repeat_num and i:
            avg = 0
            for k in range(len(w[0])):
                w[0][k] += sigmoid(new_w[0][k] / inner_repeat_num)
                avg += w[0][k]
            avg /= 4
            for k in range(len(w[0])):
                w[0][k] -= avg
            new_w[0] = [0, 0, 0, 0]

    m = 0
    for i in w[0]:
        m += abs(i)
    m /= len(w[0])
    for i in range(len(w[0])):
        w[0][i] *= (1/m)

    return w[0]


def trialling():
    global trial_num
    try:
        trial_num = int(input("Number of actual trials after training: "))
    except:
        wrong_input()
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
        n[0] = trials[i][0]
        n[1][0] = w[0][0] * n[0][0] + w[0][2] * n[0][1]
        n[1][1] = w[0][1] * n[0][0] + w[0][3] * n[0][1]
        n[1][0] = round_n_to_rdp(sigmoid(n[1][0]*100), 4)
        n[1][1] = round_n_to_rdp(sigmoid(n[1][1]*100), 4)
        if (n[1][0] > n[1][1] and not trials[i][1][1]) or (n[1][0] < n[1][1] and not trials[i][1][0]):
            correct_num += 1
            print(n[1])
    print('\nMy neural network was correct', 100 * correct_num / trial_num, '% of the time.')


play = True
while play:
    inputs = []
    n = [[0, 0], [0, 0]]
    w = [[0, 0, 0, 0]]
    c = [0, 0]
    w[0] = training()
    trialling()
    print("Final weights:", w[0], "\n")
    if input("Again [Y/n]: ") not in ["Y", "y"]:
        play = False
    print("")

exit()
