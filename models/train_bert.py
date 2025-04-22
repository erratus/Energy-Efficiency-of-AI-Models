import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from utils import load_cifar10, measure_energy, test_model, time, transforms, datasets
from torch.optim.lr_scheduler import CosineAnnealingLR

class ImprovedBERTForCIFAR10(nn.Module):
    def __init__(self):
        super(ImprovedBERTForCIFAR10, self).__init__()
        
        # 1. CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((4, 4))  # Output: 256x4x4 = 4096 dims
        )
        
        # 2. Better Projection to BERT dimensions
        self.projection = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 768)
        )
        
        # 3. Custom BERT configuration for images
        config = BertConfig(
            hidden_size=768,
            num_hidden_layers=4,  # Smaller than standard BERT
            num_attention_heads=8,
            intermediate_size=1024
        )
        self.bert = BertModel(config)
        
        # 4. Improved Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # Extract features with CNN
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        
        # Project to BERT dimensions
        projected = self.projection(features)
        projected = projected.unsqueeze(1)  # Add sequence dim
        
        # BERT processing
        outputs = self.bert(inputs_embeds=projected)
        
        # Classification
        logits = self.classifier(outputs.last_hidden_state[:, 0])
        return logits

def train_bert():
    # Enhanced data loading with augmentation
    def load_cifar10_augmented(batch_size=32):
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    train_loader, test_loader = load_cifar10_augmented()
    model = ImprovedBERTForCIFAR10().cuda()
    criterion = nn.CrossEntropyLoss()
    
    # Better optimizer configuration
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    start_time = time.time()
    start_energy = measure_energy()

    print("Starting training...")
    for epoch in range(10):  # Train for more epochs
        model.train()
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        scheduler.step()
        print(f"Epoch {epoch+1} Average Loss: {total_loss/len(train_loader):.4f}")

    end_time = time.time()
    energy_consumed = (measure_energy() - start_energy) * (end_time - start_time)

    # Save results
    with open("results/bert_results.txt", "w") as f:
        f.write(f"Improved BERT Training Time: {end_time - start_time:.2f} seconds\n")
        f.write(f"Improved BERT Energy Consumed: {energy_consumed:.2f} Joules\n")

    # Test the model
    test_model(model, test_loader, criterion, "results/bert_test_results.txt")

if __name__ == "__main__":
    train_bert()