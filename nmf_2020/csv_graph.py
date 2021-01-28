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
# F_LOG = [float(i) for i in data_T[2]]

show_graph(x_list, y_list, x_label, y_label)

'''
fig = plt.figure()
# fig.subplots_adjust(wspace=0.5)
ax1 = fig.add_subplot(121, xlabel=x_label, ylabel=y_label)
ax1.plot(x_list, y_list)
ax2 = fig.add_subplot(122, xlabel=x_label, ylabel=labels[2])
ax2.plot(x_list, F_LOG)
plt.show()
'''
