wsl wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wsl wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wsl wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wsl wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

wsl gzip -d train-images-idx3-ubyte.gz
wsl gzip -d train-labels-idx1-ubyte.gz
wsl gzip -d t10k-images-idx3-ubyte.gz
wsl gzip -d t10k-labels-idx1-ubyte.gz
