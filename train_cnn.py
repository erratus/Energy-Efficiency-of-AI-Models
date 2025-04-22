import torch
import torch.nn as nn
from utils import load_cifar10, measure_energy, test_model, time

class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def train_cnn():
    train_loader, test_loader = load_cifar10()
    model = ImprovedCNN().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    start_time = time.time()
    start_energy = measure_energy()

    print("Starting training...")
    for epoch in range(15):  # Train for 10 epochs
        print(f"Epoch {epoch + 1}/{15}")
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
    with open("results/cnn_results.txt", "w") as f:
        f.write(f"CNN Training Time: {training_time} seconds\n")
        f.write(f"CNN Energy Consumed: {energy_consumed} Joules\n")

    print(f"CNN Training Time: {training_time} seconds")
    print(f"CNN Energy Consumed: {energy_consumed} Joules")

    # Test the model
    test_model(model, test_loader, criterion, "results/cnn_test_results.txt")

if __name__ == "__main__":
    train_cnn()