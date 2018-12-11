#!/usr/bin/python

####################
# Name: Jason Holman
# A#:   A01895834
####################

from sklearn import tree, metrics
from sklearn.metrics import confusion_matrix, classification_report
import random
import numpy as np
import csv
import os
import pickle as cPickle


def main():
    data_x, data_y = process_csv('AIPiCar')
    train_x = data_x[int(len(data_x) * .1):]
    test_x = data_x[:int(len(data_x) * .1)]
    train_y = data_y[int(len(data_y) * .1):]
    test_y = data_y[:int(len(data_y) * .1)]
    print('Data prepared starting training process')

    test_dtr(train_x, train_y, test_x, test_y)

    dtr = load('pck_nets/lines_dtr.pck')
    valid_preds = dtr.predict(test_x)
    print(metrics.classification_report(test_y, valid_preds))


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
    return (input, output)


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
            return 1
        elif commands[0] == 2:
            e = np.zeros((4, 1))
            e[1] = 1
            return 2
        elif commands[0] == 3:
            e = np.zeros((4, 1))
            e[2] = 1
            return 3
        elif commands[0] == 4:
            e = np.zeros((4, 1))
            e[3] = 1
            return 4


def create_input(x_left, x_right, slope_left, slope_right):
    inputs = np.array([x_left, x_right, slope_left, slope_right])
    inputs = inputs / 340.0
    inputs = np.reshape(inputs, (4, ))
    return inputs


def test_dtr(train_x, train_y, test_x, test_y):
    for _ in range(10):
        clf = tree.DecisionTreeClassifier(random_state=random.randint(0, 100))
        dtr = clf.fit(train_x, train_y)
        print('Training completed...')
        valid_preds = dtr.predict(test_x)
        print(metrics.classification_report(test_y, valid_preds))
    save(dtr, 'pck_nets/lines_dtr.pck')


def save(tree, file_name):
    with open(file_name, 'wb') as fp:
        cPickle.dump(tree, fp)


def load(file_name):
    with open(file_name, 'rb') as fp:
        tree = cPickle.load(fp)
    return tree


main()
