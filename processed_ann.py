#/usr/bin/python

####################
# Name: Jason Holman
# A#:   A01895834
####################

# deleted row 662 from AIPiCar/PI_CAR_DATA3/PI_Car_Runs.csv

from network import *
import csv
import numpy as np
import os
import cv2
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
                            image_path = os.path.join(dirname, 'processedImages', line[0])
                            input.append(create_input(image_path))
                            output.append(create_output(line[1]))
                            # data.append((create_input(image_path), create_output(line[1])))
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


def create_input(input_file):
    img = cv2.imread(input_file)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled_img_data = gray_image / 255.0
    scaled_img_data = np.reshape(scaled_img_data, (76800, 1))
    return scaled_img_data


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
    image_ann = Network([76800, 95, 45, 4], CrossEntropyCost)
    random.shuffle(data)
    train_data = data[int(len(data)*.1):]
    test_data = data[:int(len(data)*.1)]
    print('Data prepared starting training process')
    train_ann(image_ann, .5, 10, 10, 2.0, train_data, test_data, 'pck_nets/image_ann.pck')
    train_ann(image_ann, .25, 10, 10, 2.0, train_data, test_data, 'pck_nets/image_ann.pck')
    train_ann(image_ann, .1, 10, 10, 2.0, train_data, test_data, 'pck_nets/image_ann.pck')

    net = load('pck_nets/image_ann.pck')
    print(test_data[0][1])
    print(np.argmax(net.feedforward(test_data[0][0])))


main()
