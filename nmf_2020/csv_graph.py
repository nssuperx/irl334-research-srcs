import csv
import matplotlib.pyplot as plt

with open('out_data/nmf_r_test_frobenius_mu.csv') as f:
    reader = csv.reader(f)
    data = [row for row in reader]

labels = data[0]
del data[0]    # 先頭1行を消す

data_T = [x for x in zip(*data)]

r_list = [int(i) for i in data_T[0]]
iter_LOG = [float(i) for i in data_T[1]]
F_LOG = [float(i) for i in data_T[2]]

fig = plt.figure()
# fig.subplots_adjust(wspace=0.5)
ax1 = fig.add_subplot(121, xlabel=labels[0], ylabel=labels[1])
ax1.plot(r_list, iter_LOG)
ax2 = fig.add_subplot(122, xlabel=labels[0], ylabel=labels[2])
ax2.plot(r_list, F_LOG)
plt.show()
