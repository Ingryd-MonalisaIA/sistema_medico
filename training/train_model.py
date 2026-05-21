import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO

# 1. Definir a arquitetura da rede SimpleCNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=14):
        super(SimpleCNN, self).__init__()
        # Entrada: 1 canal (grayscale), 28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Após conv1 + pool: 14x14
        # Após conv2 + pool: 7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train():
    print("=== Iniciando Preparação para o Treinamento ===")
    
    # 2. Carregar informações do ChestMNIST dinamicamente
    db_name = 'chestmnist'
    info = INFO[db_name]
    num_classes = len(info['label'])
    print(f"Dataset: {info['description']}")
    print(f"Número de classes (patologias): {num_classes}")
    
    DataClass = getattr(medmnist, info['python_class'])
    
    # 3. Transformações de dados (Normalização para [-1, 1])
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Baixar e carregar o dataset
    print("Carregando datasets (se necessário, o download será feito)...")
    train_dataset = DataClass(split='train', transform=data_transform, download=True)
    val_dataset = DataClass(split='val', transform=data_transform, download=True)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=False)
    
    # 4. Inicializar modelo, função de perda e otimizador
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilizando dispositivo: {device}")
    
    model = SimpleCNN(num_classes=num_classes).to(device)
    
    # Como ChestMNIST é classificação multi-label, usamos BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    print("\n=== Iniciando Treinamento (5 Épocas) ===")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validação
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device, dtype=torch.float32)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                
                # Para calcular acurácia em multi-label:
                # Comparamos se a previsão (sigmoid(output) >= 0.5) coincide com o ground truth
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                
                correct_predictions += (preds == targets).sum().item()
                total_predictions += targets.numel()
                
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct_predictions / total_predictions
        
        print(f"Época {epoch}/{epochs} | Loss Treino: {train_loss:.4f} | Loss Val: {val_loss:.4f} | Acurácia Val: {val_accuracy:.4f}")
        
    print("\nTreinamento concluído com sucesso!")
    
    # 5. Salvar o modelo
    # Salvar na raiz do projeto 'pulmai_medmnist'
    # Como o arquivo atual está em 'pulmai_medmnist/training/train_model.py',
    # a raiz do projeto é a pasta pai deste diretório.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, 'pulmai_model.pth')
    
    print(f"Salvando pesos do modelo em: {model_path}")
    torch.save(model.state_dict(), model_path)
    print("Pesos salvos com sucesso!")

if __name__ == '__main__':
    train()
