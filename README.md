# 🏥 PulmAI MedMNIST — Sistema de Análise de Imagens Médicas

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3.0+-06B6D4?logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)

O **PulmAI MedMNIST** é uma aplicação médica full-stack integrada que utiliza redes neurais convolucionais (CNN) para realizar a triagem automática de patologias pulmonares a partir de exames de Raio-X de tórax. O sistema foi desenvolvido utilizando o dataset acadêmico **ChestMNIST** (da biblioteca MedMNIST), composto por imagens de $28 \times 28$ pixels.

A aplicação foi estruturada rigorosamente sob uma arquitetura de três camadas (**Treinamento**, **Backend/API** e **Frontend**) e traz como diferencial técnico uma lógica de agregação estatística baseada em decisões independentes (função sigmoide), respeitando a natureza *multi-label* dos diagnósticos clínicos.

---

## 📁 Estrutura do Projeto

O workspace está organizado da seguinte forma:

```plaintext
pulmai_medmnist/
├── backend/
│   ├── main.py             # Servidor FastAPI e carregamento do modelo
│   └── requirements.txt    # Dependências do ambiente Python
├── training/
│   └── train_model.py      # Pipeline isolado de treinamento da SimpleCNN
├── test_samples/           # Banco de imagens de teste em formato PNG
│   ├── sample_normal_1.png
│   └── sample_anomalia_1.png (e outros...)
├── generate_samples.py     # Script utilitário para extrair amostras do MedMNIST
├── frontend/
│   └── index.html          # Interface de usuário (Visual PACS em Dark Mode)
└── pulmai_model.pth        # Pesos binários do modelo treinado

```
## 🛠️ Detalhes do Desenvolvimento Técnico
1. Modelo & Treinamento (training/)
   
Arquitetura: Rede Neural Convolucional (SimpleCNN) adaptada para receber tensores de canal único (tons de cinza) na resolução 28 x 28.

Função de Perda: Foram avaliadas as 14 patologias do ChestMNIST de forma paralela através da função de perda BCEWithLogitsLoss (Binary Cross-Entropy com logits integrados).

Otimizador: Adam, treinado por 5 épocas com carregamento dinâmico dos dados.

Métricas de Validação alcançadas:
Acurácia Média por Rótulo:94.92%

Precisão: 0.85 | Recall: 0.94 | F1-Score: 0.89

2. Backend & API (backend/)
   
Construído sobre o FastAPI, o servidor expõe o endpoint principal POST /analyze.

Lógica de Agregação Multi-Label (Diferencial Clínico): Como um paciente pode apresentar múltiplas patologias simultaneamente (ex: Pneumonia e Derrame Pleural), o modelo calcula a probabilidade de cada uma das 14 classes de forma independente usando a ativação Sigmoide, em vez de Softmax.

Critério de Classificação: * Se o score de qualquer uma das 14 patologias for 0.50, o exame é classificado como Anomalia. A confiança reportada equivale ao maior score encontrado.
Caso nenhum score atinja o limiar, o exame é classificado como Normal. A confiança é o complemento do maior score (1.0 - score_máximo).

3. Interface do Usuário (frontend/)
   
Visual PACS: Interface web inspirada em telas de estações de trabalho de radiologia profissional (Picture Archiving and Communication Systems). Desenvolvida em Tailwind CSS com paleta Dark Mode (#0a0f1d) e efeitos de destaque em tons de azul e verde esmeralda.

Experiência Dinâmica: Integração assíncrona via JavaScript (Fetch API) que simula uma linha de varredura ("scan line") médica animada sobre a imagem durante a inferência. Conta com painel de logs técnicos em tempo real e exibição de cartões estatísticos.

## 🧪 Resposta Padrão da API (Exemplo de Payload JSON)
Ao submeter uma imagem para análise na rota /analyze, o backend retorna a seguinte estrutura sem a necessidade de recarregar a interface:
```
{
  "status": "success",
  "class": "Normal",
  "confidence": 0.9074,
  "max_pathology": "Nenhuma",
  "detected_pathologies": [],
  "all_predictions": {
    "Atelectasia": 0.0166,
    "Cardiomegalia": 0.0013,
    "Derrame Pleural": 0.0081,
    "Infiltração": 0.0926,
    "Massa": 0.011,
    "Nódulo": 0.0287,
    "Pneumonia": 0.0013,
    "Pneumotórax": 0.0074,
    "Consolidação": 0.0027,
    "Edema": 0.0007,
    "Enfisema": 0.0037,
    "Fibrose": 0.0054,
    "Espessamento Pleural": 0.0092,
    "Hérnia": 0.0003
  },
  "academic_metrics": {
    "precision": 0.85,
    "recall": 0.94,
    "f1_score": 0.89
  }
}
```
## 🚀 Como Executar o Projeto Localmente

Pré-requisitos

Certifique-se de ter o Python 3.10 ou superior instalado na sua máquina.

1. Clonar o Repositório e Instalar Dependências
```
Bash
git clone [https://github.com/Ingryd-MonalisaIA/sistema_medico.git](https://github.com/Ingryd-MonalisaIA/sistema_medico.git)
cd sistema_medico
pip install -r backend/requirements.txt
```
2. Iniciar o Servidor FastAPI
   
Execute o Uvicorn apontando para a aplicação do backend:
```
Bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
O terminal exibirá a mensagem indicando que o servidor está online: INFO: Application startup complete.

3. Abrir o Frontend
   
Basta navegar até a pasta frontend/ e abrir o arquivo index.html em qualquer navegador de internet (Chrome, Edge, Firefox).

4. Realizar Testes Clínicos
   
Com a interface aberta, certifique-se de que o indicador no cabeçalho exibe: API ONLINE | Modelo Pronto.

Utilize a seção "Amostras Rápidas para Teste" no rodapé para carregar uma imagem real do banco de testes do ChestMNIST.

Clique em "Executar Diagnóstico Neural" para visualizar a varredura da IA e o relatório detalhado das 14 patologias mapeadas na tela.
