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
import cv2


def main():
    data_x, data_y = process_csv('AIPiCar')
    train_x = data_x[int(len(data_x) * .1):]
    test_x = data_x[:int(len(data_x) * .1)]
    train_y = data_y[int(len(data_y) * .1):]
    test_y = data_y[:int(len(data_y) * .1)]
    print('Data prepared starting training process')
    test_dtr(train_x, train_y, test_x, test_y)


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
                            image_path = os.path.join(dirname, 'rawImages', line[0])
                            input.append(create_input(image_path))
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


def create_input(input_file):
    img = cv2.imread(input_file)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled_img_data = gray_image / 255.0
    scaled_img_data = np.reshape(scaled_img_data, (38400, ))
    return scaled_img_data


def test_dtr(train_x, train_y, test_x, test_y):
    for _ in range(10):
        clf = tree.DecisionTreeClassifier(random_state=random.randint(0, 100))
        dtr = clf.fit(train_x, train_y)
        print('Training completed...')
        valid_preds = dtr.predict(test_x)
        print(metrics.classification_report(test_y, valid_preds))


main()
