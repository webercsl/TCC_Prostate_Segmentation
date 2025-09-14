import monai
import argparse
import pandas as pd
import numpy as np
import os
import sys

# Garante que o módulo prostate158 seja encontrado
sys.path.append(os.getcwd())

from prostate158.utils import load_config
from prostate158.train import SegmentationTrainer

def check_files_exist(row, base_dir, required_cols):
    """Verifica se os arquivos de uma linha do DataFrame existem."""
    for col in required_cols:
        # Pula células vazias (NaN) que são válidas para colunas opcionais
        if pd.isna(row[col]):
            continue
        file_path = os.path.join(base_dir, row[col])
        if not os.path.exists(file_path):
            print(f"AVISO: Pulando linha. Arquivo não encontrado: {file_path}")
            return False
    return True

def main():
    parser = argparse.ArgumentParser(description='Run a simple training session.')
    parser.add_argument('--config', required=True, type=str, help='Path to the configuration file')
    args = parser.parse_args()
    config_fn = args.config

    config = load_config(config_fn)
    monai.utils.set_determinism(seed=config.seed)

    # --- Verificação de Dados (opcional, mas recomendado) ---
    # Carrega os dataframes
    train_df = pd.read_csv(config.data.train_csv)
    valid_df = pd.read_csv(config.data.valid_csv)

    # Define quais colunas são OBRIGATÓRIAS para este treinamento
    required_columns_for_run = config.data.image_cols + config.data.label_cols
    
    # Filtra o DataFrame de treino
    print("Verificando arquivos de treino...")
    train_mask = train_df.apply(check_files_exist, axis=1, base_dir=config.data.data_dir, required_cols=required_columns_for_run)
    train_df = train_df[train_mask]
    
    # Filtra o DataFrame de validação
    print("Verificando arquivos de validação...")
    valid_mask = valid_df.apply(check_files_exist, axis=1, base_dir=config.data.data_dir, required_cols=required_columns_for_run)
    valid_df = valid_df[valid_mask]
    
    print("Verificação completa. Iniciando treinamento...")
    # --- Fim da Verificação ---
    
    # Cria o trainer (ele usará os CSVs definidos no config)
    trainer = SegmentationTrainer(config=config)

    if 'OneCycleLR' in config.lr_scheduler:
        trainer.fit_one_cycle()
    
    # Inicia o treinamento
    trainer.run()

    print("\n--- Treinamento Concluído ---")
    print(f"Resultados salvos na pasta: {config.run_id}")
    print(f"Melhor época: {trainer.evaluator.state.best_metric_epoch}")
    print(f"Melhor Mean Dice na validação: {trainer.evaluator.state.best_metric:.4f}")

# Proteção para multiprocessamento no Windows
if __name__ == '__main__':
    main()