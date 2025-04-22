import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_cifar10, measure_energy, test_model, time

class ImprovedGLAM(nn.Module):
    def __init__(self):
        super(ImprovedGLAM, self).__init__()
        # Feature extractor (CNN)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Gated Linear Attention
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.gate = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # CNN feature extraction
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        
        # GLAM
        x = F.relu(self.fc1(x))
        gate = torch.sigmoid(self.gate(x))
        x = x * gate  # Gating
        x = self.fc2(x)
        return x

def train_glam():
    train_loader, test_loader = load_cifar10()
    model = ImprovedGLAM().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    start_time = time.time()
    start_energy = measure_energy()

    print("Training Improved GLAM...")
    for epoch in range(10):  # Train longer
        model.train()
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    end_time = time.time()
    energy_consumed = (measure_energy() - start_energy) * (end_time - start_time)

    with open("results/glam_results.txt", "w") as f:
        f.write(f"Improved GLAM Training Time: {end_time - start_time:.2f} seconds\n")
        f.write(f"Improved GLAM Energy Consumed: {energy_consumed:.2f} Joules\n")

    test_model(model, test_loader, criterion, "results/glam_test_results.txt")

if __name__ == "__main__":
    train_glam()