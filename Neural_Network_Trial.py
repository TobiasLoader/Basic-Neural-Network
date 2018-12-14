
from math import *
from random import *
from sys import exit


nodesPerLayer = [2, 2, 2]


def generate_inputs_list(training_num):
    l = []
    for i in range(training_num):
        x = uniform(-1, 1)
        y = uniform(-1, 1)
        l.append([[x, y]])
        if x >= y:
            l[i].append([1, 0])
        else:
            l[i].append([0, 1])
    return l


def init_node_structure():
    x = []
    for layer in range(len(nodesPerLayer)):
        x.append([])
        for node in range(nodesPerLayer[layer]):
            x[layer].append(0)
    return x


def init_weight_structure():
    x = []
    for layer in range(len(nodesPerLayer) - 1):
        x.append([])
        wireNum = nodesPerLayer[layer] * nodesPerLayer[layer + 1]
        for wire in range(wireNum):
            tempW = 0
            if len(nodesPerLayer) > 2:
                while not tempW:
                    tempW = uniform(0, 0.00000000001)
            x[layer].append(tempW)
    return x


def init_network():
    global nodeStructure, weightStructure
    nodeStructure = init_node_structure()
    weightStructure = init_weight_structure()


def sigmoid(x):
    return 1 / (1+ exp(-x))


def round_n_to_rdp(n, r):
    return round((10**r)*n)/(10**r)


def exagerate_outputs(x):
    return round_n_to_rdp(sigmoid(round_n_to_rdp(x*100, 1)), 1)


def wrong_input():
    print("\nOh no!\nYour input is not compatible with our code!\nSorry...")
    exit()


###########################


def val_of_node(x, y):
    s = 0
    for i in range(nodesPerLayer[x-1]):
        s += n[x-1][i] * w[x-1][y + i * nodesPerLayer[x]]
    return s


def cost_output_node(i, final_node, aim):
    c[i] = (final_node[i] - aim[i]) ** 2


def aim_of_prev_node(x, y):
    s = 0
    for i in range(nodesPerLayer[x+1]):
        if w[x][y + i * nodesPerLayer[x]]:
            s += a[x+1][i] * w[x][y + i * nodesPerLayer[x]]
    return (1/nodesPerLayer[x-1]) * s


def new_weight_calc(x, y, a, n):
    return (a[x+1][y % nodesPerLayer[x+1]]) * (n[x][y // nodesPerLayer[x+1]])


def refine_weights():
    for k in range(len(weightStructure)):
        m = 0
        for l in w[k]:
            m += abs(l)
        m /= len(w[k])
        for l in range(len(w[k])):
            if m:
                w[k][l] *= (1/m)


def training():
    global training_num
    try:
        training_num = int(input("\nNumber of training examples: "))
    except:
        wrong_input()
    if training_num > 10:
        inner_repeat_num = 10
    else:
        inner_repeat_num = round(sqrt(training_num))
    inputs = generate_inputs_list(training_num)
    new_w = init_weight_structure()
    for i in range(len(inputs)):
        # print(w)
        n = nodeStructure
        n[0] = inputs[i][0]
        final_node_temp = nodeStructure[-1]

        if len(nodesPerLayer) > 2:
            for j in range(len(nodesPerLayer)-1):
                if j:
                    for k in range(nodesPerLayer[j]):
                       n[j][k] = val_of_node(j, k)

        for j in range(nodesPerLayer[-1]):
            final_node_temp[j] = val_of_node(len(nodesPerLayer)-1, j)

        for j in range(nodesPerLayer[-1]):
            cost_output_node(j, final_node_temp, inputs[i][1])

        a[-1] = c

        n[len(nodesPerLayer)-1] = final_node_temp

        if len(nodesPerLayer) > 2:
            for j in range(len(nodesPerLayer)-2):
                k = len(nodesPerLayer) - j - 2
                for l in range(nodesPerLayer[k]):
                    a[k][l] = aim_of_prev_node(k, l)

        for j in range(len(weightStructure)):
            for k in range(len(weightStructure[j])):
                new_w[j][k] += new_weight_calc(j, k, a, n)

        if not i % inner_repeat_num and i:
            for j in range(len(weightStructure)):
                avg = 0
                for k in range(len(w[j])):
                    w[j][k] += sigmoid(new_w[j][k] / inner_repeat_num)
                    avg += w[j][k]
                avg /= len(weightStructure[j])
                for k in range(len(w[j])):
                    w[j][k] -= avg
                refine_weights()
            new_w = init_weight_structure()

    refine_weights()

    return w


def trialling():
    global trial_num
    try:
        trial_num = int(input("Number of actual trials after training: "))
    except:
        wrong_input()
    trials = []
    correct_num = 0
    trials = generate_inputs_list(trial_num)

    for i in range(trial_num):
        n[0] = trials[i][0]
        if len(nodesPerLayer) > 2:
            for j in range(len(nodesPerLayer)-1):
                if j:
                    for k in range(nodesPerLayer[j]):
                        n[j][k] = val_of_node(j, k)

        for j in range(nodesPerLayer[-1]):
            n[len(nodesPerLayer)-1][j] = exagerate_outputs(val_of_node(len(nodesPerLayer)-1, j))

        if (n[1][0] > n[1][1] and not trials[i][1][1]) or (n[1][0] < n[1][1] and not trials[i][1][0]):
            correct_num += 1
    print('\nMy neural network was correct', 100 * correct_num / trial_num, '% of the time.\n')


init_network()
play = True
while play:
    inputs = []
    n = init_node_structure()
    w = init_weight_structure()
    a = init_node_structure()
    c = [0, 0]
    w = training()
    trialling()
    print("Last example:", n)
    print("Final weights:", w, "\n")
    if input("Again [Y/n]: ") not in ["Y", "y"]:
        play = False
    print("")

exit()