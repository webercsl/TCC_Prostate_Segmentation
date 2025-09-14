# 🩺 Prostate Segmentation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?logo=pytorch&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-Latest-green?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDE1Ljc0TDEyIDIyTDEwLjkxIDE1Ljc0TDQgOUwxMC45MSA4LjI2TDEyIDJaIiBmaWxsPSJ3aGl0ZSIvPgo8L3N2Zz4K&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Segmentação automatizada de próstata em imagens médicas utilizando Deep Learning**

</div>

---

## 📖 Sobre o Projeto

O **Prostate Segmentation** é um projeto de pesquisa acadêmica focado na segmentação automatizada de próstata em imagens médicas utilizando redes neurais profundas. O sistema emprega técnicas avançadas de Deep Learning para fornecer segmentações precisas e confiáveis, auxiliando profissionais de saúde no diagnóstico e tratamento.

### ✨ Características Principais

- 🧠 **Deep Learning**: Utiliza redes neurais convolucionais avançadas
- 🔬 **MONAI Framework**: Implementação baseada no framework MONAI para aplicações médicas
- 🚀 **GPU Acceleration**: Otimizado para treinamento acelerado com CUDA
- 📊 **Cross-Validation**: Suporte a validação cruzada K-fold
- 📈 **Métricas Detalhadas**: Análise completa de performance com múltiplas métricas

---

## 🛠️ Pré-requisitos

### 💻 Sistema e Hardware

| Componente | Especificação | Observações |
|------------|---------------|-------------|
| **Sistema Operacional** | Windows 10/11, Linux (Ubuntu 18.04+), macOS | Testado principalmente no Windows 11 |
| **Python** | 3.10+ | Versão recomendada: 3.10.11 |
| **GPU** | NVIDIA RTX 3060 12GB | Suporte CUDA 12.1 obrigatório |
| **CPU** | AMD Ryzen 7 3800XT | Ou equivalente Intel |
| **RAM** | 32GB | Mínimo recomendado |
| **Armazenamento** | 50GB livres | Para datasets e modelos |

### 🔧 Ferramentas Necessárias

- Git
- Ambiente virtual Python (venv ou conda)
- Driver NVIDIA atualizado
- CUDA Toolkit 12.1

---

## 🚀 Instalação

### 1️⃣ Configuração do Ambiente Python

```bash
# Clonar o repositório
git clone https://github.com/webercsl/TCC_Prostate_Segmentation.git
cd prostate-segmentation

# Criar ambiente virtual
python -m venv NOME_ESCOLHIDO_PARA_O_AMBIENTE

# Ativar ambiente virtual
# Windows
NOME_ESCOLHIDO_PARA_O_AMBIENTE\Scripts\activate

# Atualizar pip
python -m pip install --upgrade pip
```

### 2️⃣ Instalação das Dependências

```bash
# Instalar PyTorch com suporte CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

# Instalar MONAI e dependências adicionais
pip install "monai[all]" munch optuna scikit-learn pandas ipywidgets
```

> **💡 Dica**: Para sistemas sem GPU, use: `pip install torch torchvision torchaudio`

### 3️⃣ Verificação da Instalação

```python
# Verificar instalação do PyTorch e CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponível: {torch.cuda.is_available()}')"
```

---

## 📊 Datasets

### 📥 Download dos Dados

Os datasets estão hospedados no **Zenodo** e devem ser baixados separadamente:

#### Dataset de Treinamento
- 🔗 **Link**: [Dataset de Treino - Zenodo](https://zenodo.org/records/10013697)
- 📁 **Destino**: `prostate158/train/`
- ⚠️ **Importante**: Excluir pacientes 20, 21, 22 e 23 (dados ausentes no CSV)

#### Dataset de Teste
- 🔗 **Link**: [Dataset de Teste - Zenodo](https://zenodo.org/records/10047292)  
- 📁 **Destino**: `prostate158/test/`

### 📂 Estrutura de Diretórios

```
prostate-segmentation/
├── prostate158/
│   ├── train/
│   │   ├── paciente_024/
│   │   ├── paciente_025/
│   │   └── ...
│   └── test/
│       ├── paciente_001/
│       ├── paciente_002/
│       └── ...
├── models/
├── train.py
├── train_kfold.py
├── predict_test.py
├── predict_train.py
├── anatomy_tcc1.yaml
└── anatomy_kfold.yaml
```

---

## 🔬 Execução

### 🧪 Realizando Predições

#### Predição no Dataset de Teste

```bash
python predict_test.py --config anatomy_tcc1.yaml --checkpoint "models/500epocas.pt" --test_case_id 5
```

**Parâmetros:**
- `--config`: Arquivo de configuração YAML
- `--checkpoint`: Caminho para o modelo treinado
- `--test_case_id`: ID do caso de teste específico

#### Predição no Dataset de Treinamento

```bash
python predict_train.py --input_dir "models" --pred_filename "500epocas.pt"
```

### 🎯 Treinamento de Modelos

#### TCC1 - Treinamento Padrão (500 Épocas)

```bash
python train.py --config anatomy_tcc1.yaml
```

Este modo utiliza validação simples e é mais rápido para testes iniciais.

#### TCC2 - Treinamento com Validação Cruzada K-Fold

```bash
python train_kfold.py --config anatomy_kfold.yaml
```

Este modo oferece validação mais robusta através de K-fold cross-validation.

### ⚙️ Configurações

Os arquivos de configuração YAML permitem ajustar diversos parâmetros:

- **Hiperparâmetros**: learning rate, batch size, épocas
- **Arquitetura**: tipo de modelo, layers, ativações  
- **Augmentação**: transformações aplicadas aos dados
- **Otimização**: algoritmo, scheduler, regularização

---

## 🔍 Métricas e Avaliação

O projeto implementa múltiplas métricas para avaliação:

- **Dice Score**: Coeficiente de similaridade
- **IoU (Intersection over Union)**: Medida de sobreposição
- **Hausdorff Distance**: Distância máxima entre contornos
- **Volume Similarity**: Comparação volumétrica
- **Surface Distance**: Distância média entre superfícies

---

## 🛠️ Solução de Problemas

### ❌ Problemas Comuns

<details>
<summary><strong>🔧 Erro de Dependências</strong></summary>

**Solução:**
```bash
# Verificar ambiente virtual ativo
which python  # Linux/macOS
where python   # Windows

# Reinstalar dependências
pip install --force-reinstall -r requirements.txt
```
</details>

<details>
<summary><strong>🎮 Problemas com CUDA</strong></summary>

**Verificações:**
1. Driver NVIDIA atualizado
2. CUDA Toolkit 12.1 instalado
3. Compatibilidade GPU (RTX 3060)

```bash
# Verificar CUDA
nvcc --version
nvidia-smi
```
</details>

<details>
<summary><strong>📁 Dataset Incompleto</strong></summary>

**Checklist:**
- [ ] Downloads completos do Zenodo
- [ ] Estrutura de pastas correta
- [ ] Pacientes 20-23 removidos do treino
- [ ] Permissões de acesso adequadas
</details>

---

## 💡 Dicas de Uso

### 🚀 Otimização de Performance

- ✅ Sempre use o ambiente virtual ativado
- ✅ Mantenha drivers e CUDA atualizados
- ✅ Monitore uso de memória GPU durante treinamento
- ✅ Use batch sizes adequados para sua GPU (recomendado: 4-8 para RTX 3060)
- ✅ Implemente early stopping para evitar overfitting

---

## 🤝 Contribuição

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

---

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 👨‍🎓 Autor

<div align="center">

**Gustavo Weber**

*Formando em Engenharia de Computação 2025/2*  
*UNIFTEC - Centro Universitário*

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/webercsl)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/webercsl)

</div>

---

## 🙏 Agradecimentos

- **UNIFTEC** pelo suporte institucional
- **Comunidade MONAI** pelos frameworks e ferramentas
- **Zenodo** pela hospedagem dos datasets
- **NVIDIA** pelo suporte a CUDA e desenvolvimento em GPU

---

<div align="center">

**⭐ Se este projeto foi útil para você, considere dar uma estrela!**

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=username.prostate-segmentation)

</div>