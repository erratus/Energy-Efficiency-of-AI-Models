import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from utils import load_cifar10, measure_energy, time
import torch
import torchvision.models as models

def extract_features(loader):
    # Use pretrained ResNet18 for feature extraction
    resnet = models.resnet18(pretrained=True).cuda()
    resnet.eval()
    features, labels = [], []
    
    with torch.no_grad():
        for images, lbls in loader:
            images = images.cuda()
            features.append(resnet(images).cpu().numpy())
            labels.append(lbls.numpy())
    
    return np.vstack(features), np.hstack(labels)

def train_xgboost():
    train_loader, test_loader = load_cifar10()
    
    # Extract features using ResNet18
    X_train, y_train = extract_features(train_loader)
    X_test, y_test = extract_features(test_loader)
    
    # Reduce dimensionality with PCA
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # GPU-accelerated XGBoost
    params = {
        'objective': 'multi:softmax',
        'num_class': 10,
        'tree_method': 'gpu_hist',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 200,
    }

    start_time = time.time()
    start_energy = measure_energy()

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_pca, y_train)

    end_time = time.time()
    energy_consumed = (measure_energy() - start_energy) * (end_time - start_time)

    # Evaluate
    y_pred = model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)

    with open("results/xgboost_results.txt", "w") as f:
        f.write(f"XGBoost (GPU + ResNet18) Test Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Training Time: {end_time - start_time:.2f} seconds\n")
        f.write(f"Energy Consumed: {energy_consumed:.2f} Joules\n")

    print(f"XGBoost (GPU + ResNet18) Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    train_xgboost()