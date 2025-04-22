import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from utils import load_cifar10, measure_energy, test_model, time
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms, datasets
import torch.nn.functional as F

class EnhancedGPT2ForCIFAR10(nn.Module):
    def __init__(self):
        super(EnhancedGPT2ForCIFAR10, self).__init__()
        
        # 1. CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((4, 4))  # Output: 256x4x4 = 4096 dims
        )
        
        # 2. Enhanced Projection Network
        self.projection = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),
            nn.Dropout(0.1),
            nn.Linear(2048, 768)
        )
        
        # 3. Custom GPT-2 Configuration
        config = GPT2Config(
            n_embd=768,
            n_layer=4,  # Reduced from 12
            n_head=8,
            activation_function="gelu",
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1
        )
        self.gpt2 = GPT2Model(config)
        
        # 4. Improved Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(768, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Dropout(0.1),
            nn.Linear(384, 10)
        )

    def forward(self, x):
        # Extract spatial features
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        
        # Project to transformer space
        projected = self.projection(features)
        projected = projected.unsqueeze(1)  # Add sequence dimension
        
        # GPT-2 processing
        outputs = self.gpt2(inputs_embeds=projected)
        
        # Classification
        logits = self.classifier(outputs.last_hidden_state[:, 0])
        return logits

def train_gpt2():
    # Enhanced data loading with augmentation
    def load_cifar10_augmented(batch_size=64):
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return train_loader, test_loader

    train_loader, test_loader = load_cifar10_augmented()
    model = EnhancedGPT2ForCIFAR10().cuda()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Added label smoothing
    
    # Optimized training configuration
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    start_time = time.time()
    start_energy = measure_energy()

    print("Starting enhanced GPT-2 training...")
    for epoch in range(10):  # Extended training
        model.train()
        total_loss = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(), labels.cuda()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

    end_time = time.time()
    energy_consumed = (measure_energy() - start_energy) * (end_time - start_time)

    # Save results
    with open("results/enhanced_gpt2_results.txt", "w") as f:
        f.write(f"Enhanced GPT-2 Training Time: {end_time - start_time:.2f} seconds\n")
        f.write(f"Enhanced GPT-2 Energy Consumed: {energy_consumed:.2f} Joules\n")

    # Test the model
    test_model(model, test_loader, criterion, "results/enhanced_gpt2_test_results.txt")

if __name__ == "__main__":
    train_gpt2()