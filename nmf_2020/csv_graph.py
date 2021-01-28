import csv
import matplotlib.pyplot as plt

from modules.visualize import show_graph

with open('out_data/nmf_r25.csv') as f:
    reader = csv.reader(f)
    data = [row for row in reader]

x_label = data[0][0]
y_label = data[0][1]
del data[0]    # 先頭1行を消す

data_T = [x for x in zip(*data)]

x_list = [int(i) for i in data_T[0]]
y_list = [float(i) for i in data_T[1]]

show_graph(x_list, y_list, x_label, y_label)
