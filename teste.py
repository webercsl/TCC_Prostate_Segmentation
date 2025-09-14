# check_paths.py
import yaml
import pandas as pd
import os

# Carrega a config para ver os caminhos
config_file = 'anatomy_tcc1.yaml'
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

data_config = config['data']
data_dir = data_config['data_dir']
train_csv_path = data_config['train_csv']

print(f"--- Verificando Caminhos do arquivo: {config_file} ---")
print(f"Diretório de Dados (data_dir): '{data_dir}'")
print(f"Caminho do CSV de Treino: '{train_csv_path}'")

# Verifica se o arquivo CSV existe
if not os.path.exists(train_csv_path):
    print(f"\nERRO CRÍTICO: O arquivo CSV '{train_csv_path}' não foi encontrado!")
else:
    print(f"\nSUCESSO: Arquivo CSV '{train_csv_path}' encontrado.")
    
    # Lê o CSV e pega o caminho da primeira imagem
    df = pd.read_csv(train_csv_path)
    primeira_imagem_relativa = df['t2'].iloc[0]
    
    # Constrói o caminho completo da imagem, como o trainer faz
    caminho_completo_imagem = os.path.join(data_dir, primeira_imagem_relativa)
    
    print(f"Caminho relativo da primeira imagem no CSV: '{primeira_imagem_relativa}'")
    print(f"Caminho completo a ser verificado: '{caminho_completo_imagem}'")

    # Verifica se a primeira imagem realmente existe nesse caminho
    if os.path.exists(caminho_completo_imagem):
        print("\nSUCESSO FINAL: A primeira imagem foi encontrada. Seus caminhos parecem estar CORRETOS!")
    else:
        print("\nERRO CRÍTICO: A primeira imagem NÃO foi encontrada no caminho construído.")
        print("Verifique se a sua estrutura de pastas corresponde exatamente ao que está no arquivo YAML.")