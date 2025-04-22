import torch
import torch.nn as nn
import timm
from utils import measure_energy, test_model, time
from torchvision import datasets, transforms

class ViTForCIFAR10(nn.Module):
    def __init__(self):
        super(ViTForCIFAR10, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)

    def forward(self, x):
        return self.vit(x)

def load_cifar10(batch_size=32):
    """Load CIFAR-10 dataset with resizing to 224x224."""
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_vit():
    train_loader, test_loader = load_cifar10()
    model = ViTForCIFAR10().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    start_time = time.time()
    start_energy = measure_energy()

    print("Starting training...")
    for epoch in range(3):  # Train for 10 epochs
        print(f"Epoch {epoch + 1}/{3}")
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

    end_time = time.time()
    end_energy = measure_energy()

    training_time = end_time - start_time
    energy_consumed = (end_energy - start_energy) * training_time

    # Save training results to a text file
    with open("results/vit_results.txt", "w") as f:
        f.write(f"ViT Training Time: {training_time} seconds\n")
        f.write(f"ViT Energy Consumed: {energy_consumed} Joules\n")

    print(f"ViT Training Time: {training_time} seconds")
    print(f"ViT Energy Consumed: {energy_consumed} Joules")

    # Test the model
    test_model(model, test_loader, criterion, "results/vit_test_results.txt")

if __name__ == "__main__":
    train_vit()