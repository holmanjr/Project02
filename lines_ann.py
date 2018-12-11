#/usr/bin/python

####################
# Name: Jason Holman
# A#:   A01895834
####################

# deleted row 662 from AIPiCar/PI_CAR_DATA3/PI_Car_Runs.csv
# maximum value in inputs is 340

from network import *
import csv
import numpy as np
import os
import pickle as cPickle

def process_csv(directory):
    input = []
    output = []
    for dirname, dirnames, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith('.csv'):
                path = os.path.join(dirname, f)
                with open(path) as f:
                    csv_reader = csv.reader(f, delimiter=',')
                    line_count = 0
                    for line in csv_reader:
                        if line_count == 0:
                            line_count += 1
                        else:
                            input.append(create_input(eval(line[2]), eval(line[3]), eval(line[4]), eval(line[5])))
                            output.append(create_output(line[1]))
    return zip(input, output)


def create_output(output):
        commands = []
        for x in output.split(','):
            if x == 'up pressed':
                commands.append(1)
            elif x == 'right pressed':
                commands.append(2)
            elif x == 'left pressed':
                commands.append(3)
            elif x == 'down pressed':
                commands.append(4)
        if len(commands) == 0:
            commands.append(1)
        if commands[0] == 1:
            e = np.zeros((4, 1))
            e[0] = 1
            return e
        elif commands[0] == 2:
            e = np.zeros((4, 1))
            e[1] = 1
            return e
        elif commands[0] == 3:
            e = np.zeros((4, 1))
            e[2] = 1
            return e
        elif commands[0] == 4:
            e = np.zeros((4, 1))
            e[3] = 1
            return e


def create_input(x_left, x_right, slope_left, slope_right):
    inputs = np.array([x_left, x_right, slope_left, slope_right])
    inputs = inputs / 340.0
    inputs = np.reshape(inputs, (4, 1))
    return inputs


def train_ann(net, eta, mini_batch, num_epochs, lmbda, train_d, test_d, path):
    net.SGD(train_d, num_epochs, mini_batch, eta, lmbda, test_d,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=True,
            monitor_training_cost=False,
            monitor_training_accuracy=True)
    save(net, path)


def save(net, file_name):
    with open(file_name, 'wb') as fp:
        cPickle.dump(net, fp)


def load(file_name):
    with open(file_name, 'rb') as fp:
        nn = cPickle.load(fp)
    return nn


def main():
    data = process_csv('AIPiCar')
    image_ann = Network([4, 95, 45, 4], CrossEntropyCost)
    random.shuffle(data)
    train_data = data[int(len(data)*.1):]
    test_data = data[:int(len(data)*.1)]
    print('Data prepared starting training process')
    train_ann(image_ann, .25, 10, 10, 2.0, train_data, test_data, 'pck_nets/lines_ann.pck')
    train_ann(image_ann, .1, 10, 10, 2.0, train_data, test_data, 'pck_nets/lines_ann.pck')
    train_ann(image_ann, .01, 10, 10, 2.0, train_data, test_data, 'pck_nets/lines_ann.pck')

    net = load('pck_nets/lines_ann.pck')
    print(test_data[0][1])
    print(np.argmax(net.feedforward(test_data[0][0])))
    print(net.feedforward(test_data[0][0]))


main()
