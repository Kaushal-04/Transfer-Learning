# 6
# Print the loss and accuracy on the test set
from trainModel import device, model, loss_fn
from loadDataset import testloader
import torch
correct = 0
total = 0
loss = 0

for images, labels in testloader:
    # Move tensors to the configured device
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)
    loss += loss_fn(outputs, labels)

    # torch.max return both max and argmax. We get the argmax here.
    _, predicted = torch.max(outputs.data, 1)

    # Compute the accuracy
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print("Test Accuracy of the model on the test images: {} %".format(100 * correct / total))
print("Test Loss of the model on the test images: {}".format(loss))