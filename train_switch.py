import torch
import torch.nn as nn
from transformers import SwitchTransformersModel
from utils import load_cifar10, measure_energy, test_model

class SwitchForCIFAR10(nn.Module):
    def __init__(self):
        super(SwitchForCIFAR10, self).__init__()
        self.switch = SwitchTransformersModel.from_pretrained('google/switch-base-8')
        self.projection = nn.Linear(3 * 32 * 32, 768)  # Project flattened image to hidden_size
        self.fc = nn.Linear(768, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten image to (batch_size, 3*32*32)
        x = self.projection(x)  # Project to (batch_size, 768)
        x = x.unsqueeze(1)  # Add sequence dimension: (batch_size, 1, 768)
        outputs = self.switch(inputs_embeds=x)
        logits = self.fc(outputs.last_hidden_state[:, 0])  # Use the first token's output
        return logits

def train_switch():
    train_loader, test_loader = load_cifar10()
    model = SwitchForCIFAR10().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    start_time = time.time()
    start_energy = measure_energy()

    print("Starting training...")
    for epoch in range(3):  # Train for 3 epochs
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
    with open("results/switch_results.txt", "w") as f:
        f.write(f"Switch Transformer Training Time: {training_time} seconds\n")
        f.write(f"Switch Transformer Energy Consumed: {energy_consumed} Joules\n")

    print(f"Switch Transformer Training Time: {training_time} seconds")
    print(f"Switch Transformer Energy Consumed: {energy_consumed} Joules")

    # Test the model
    test_model(model, test_loader, criterion, "results/switch_test_results.txt")

if __name__ == "__main__":
    train_switch()