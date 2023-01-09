from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from PIL import Image

train_dataset = datasets.MNIST(
    root='../pt_datasets',
    train=True,
    download=True,
    transform=ToTensor(),
)
test_dataset = datasets.MNIST(
    root='../pt_datasets',
    train=False,
    download=True,
    transform=ToTensor(),
)

MNIST_classes = datasets.MNIST.classes

batch_size = 5

# dl = DataLoader(train_dataset, batch_size, shuffle=True)
dl = DataLoader(train_dataset, batch_size, shuffle=False)

it = iter(dl)
images, classes = next(it)

for image, c in zip(images, classes):
    img = Image.fromarray(image[0].numpy() * 255).convert("L")
    img.save(f"./mnist-sample-{c}.png")
