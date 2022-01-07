import csv
import matplotlib.pyplot as plt
import numpy as np

from modules.visualize import show_graphs

data = []
with open('out_data/nmf_r10_m10000.csv') as f:
    reader = csv.reader(f)
    data.append([row for row in reader])

with open('out_data/nmf_r49_m10000.csv') as f:
    reader = csv.reader(f)
    data.append([row for row in reader])

with open('out_data/nmf_r100_m10000.csv') as f:
    reader = csv.reader(f)
    data.append([row for row in reader])

with open('out_data/nmf_r727_m10000.csv') as f:
    reader = csv.reader(f)
    data.append([row for row in reader])

with open('out_data/nmf_r1000_m10000.csv') as f:
    reader = csv.reader(f)
    data.append([row for row in reader])


x_label = data[0][0][0]

for d in data:
    del d[0]

data_T = []

for d in data:
    data_T.append([x for x in zip(*d)])

x_list = [int(i) for i in data_T[0][0]]
y_lists = []
for d_T in data_T:
    y_lists.append([float(i) for i in d_T[1]])

np_y = np.array(y_lists)
m = 10000
np_y = np_y / m

show_graphs(x_list, np_y, x_label, y_label="F / m", y_labels=("r=10", "r=25","r=49","r=100", "r=439", "r=784","r=1000"))
