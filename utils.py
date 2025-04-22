import time
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage
import torch
from torchvision import datasets, transforms

# Initialize GPU monitoring
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

def measure_energy():
    """Measure GPU power usage in watts."""
    power_draw = nvmlDeviceGetPowerUsage(handle) / 1000  # Convert milliwatts to watts
    return power_draw

def load_cifar10(batch_size=32):
    """Load CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def test_model(model, test_loader, criterion,path):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():  # Disable gradient computation
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)

    # Save test results to a text file
    with open(path, "w") as f:
        f.write(f"Test Loss: {avg_test_loss}\n")
        f.write(f"Test Accuracy: {accuracy}%\n")

    print(f"Test Loss: {avg_test_loss}")
    print(f"Test Accuracy: {accuracy}%")