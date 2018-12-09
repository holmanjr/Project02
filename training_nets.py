#/usr/bin/python

####################
# Name: Jason Holman
# A#:   A01895834
####################

# deleted row 662 from AIPiCar/PI_CAR_DATA3/PI_Car_Runs.csv

import csv
import numpy as np
import os
import cv2


def process_csv(directory, data):
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
                            data.append((create_input(image_path), create_output(line[1])))


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
            return np.array([1, 0, 0, 0])
        elif commands[0] == 2:
            return np.array([0, 1, 0, 0])
        elif commands[0] == 3:
            return np.array([0, 0, 1, 0])
        elif commands[0] == 4:
            return np.array([1, 0, 0, 0])


def create_input(input_file):
    img = cv2.imread(input_file)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled_img_data = gray_image / 255.0
    scaled_img_data = np.reshape(scaled_img_data, (38400, ))
    return scaled_img_data



def main():
    data = []
    process_csv('AIPiCar', data)
    for line in data:
        print(line)
    print(len(data))


main()
