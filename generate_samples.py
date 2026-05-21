import os
import numpy as np
from PIL import Image
import medmnist
from medmnist import INFO

def generate_samples():
    print("=== Gerando Imagens de Teste do ChestMNIST ===")
    
    # Configurar pastas
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'test_samples')
    os.makedirs(output_dir, exist_ok=True)
    
    # Carregar dataset ChestMNIST
    db_name = 'chestmnist'
    info = INFO[db_name]
    DataClass = getattr(medmnist, info['python_class'])
    
    print("Carregando o conjunto de testes...")
    # Carrega sem transformações para pegar a imagem PIL/numpy pura
    test_dataset = DataClass(split='test', download=True)
    
    # Nomes das 14 patologias
    labels_map = info['label']
    
    normal_count = 0
    anomaly_count = 0
    max_samples = 3
    
    print("Processando e salvando imagens...")
    for idx in range(len(test_dataset)):
        # Pega a imagem (PIL Image) e o target (array numpy de 14 classes binárias)
        img, target = test_dataset[idx]
        target = np.array(target)
        
        is_normal = np.sum(target) == 0
        
        if is_normal and normal_count < max_samples:
            normal_count += 1
            filename = f"sample_normal_{normal_count}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            print(f"Salvo: {filename} -> Real: Saudável (Normal)")
            
        elif not is_normal and anomaly_count < max_samples:
            anomaly_count += 1
            # Identificar quais patologias estão presentes
            active_labels = [labels_map[str(i)] for i, val in enumerate(target) if val == 1]
            patologias_str = ", ".join(active_labels)
            
            filename = f"sample_anomalia_{anomaly_count}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            print(f"Salvo: {filename} -> Real: Anomalia ({patologias_str})")
            
        if normal_count >= max_samples and anomaly_count >= max_samples:
            break
            
    print(f"\nSucesso! {normal_count + anomaly_count} imagens de teste geradas na pasta: {output_dir}")

if __name__ == '__main__':
    generate_samples()
