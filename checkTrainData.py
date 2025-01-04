# 3rd
# It's always good to inspect your data before you use it to train a model just to know everything is fine. You know what they say: garbage in, garbage out.
# Show 10 images from the training set with their labels.
import matplotlib.pyplot as plt
import numpy as np

from loadDataset import trainloader
from helperFun import get_class_name

# function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()  # convert from tensor to numpy array
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # transpose dimensions


images, labels = next(iter(trainloader))  # get the first batch

# show images with labels
fig = plt.figure(figsize=(15, 4))
plot_size = 10

for idx in np.arange(plot_size):
    ax = fig.add_subplot(2, plot_size // 2, idx + 1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(get_class_name(int(labels[idx])))

plt.show()