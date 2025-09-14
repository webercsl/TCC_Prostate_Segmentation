import monai
import argparse
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import os
import sys

# Garante que o módulo prostate158 seja encontrado
sys.path.append(os.getcwd())

from prostate158.utils import load_config
from prostate158.train import SegmentationTrainer

def check_files_exist(row, base_dir, required_cols):
    """Filtra o dataframe para usar apenas dados cujos arquivos existem."""
    for col in required_cols:
        if pd.isna(row[col]): # Pula células vazias, se houver
            continue
        file_path = os.path.join(base_dir, row[col])
        if not os.path.exists(file_path):
            print(f"AVISO: Pulando linha. Arquivo não encontrado: {file_path}")
            return False
    return True

def main():
    parser = argparse.ArgumentParser(description='Train a model with k-fold cross-validation.')
    parser.add_argument('--config', required=True, type=str, help='Path to the configuration file')
    args = parser.parse_args()
    config_fn = args.config

    config = load_config(config_fn)
    monai.utils.set_determinism(seed=config.seed)

    # Carrega e combina todos os dados de treino e validação
    train_df = pd.read_csv(config.data.train_csv)
    valid_df = pd.read_csv(config.data.valid_csv)
    all_files_df = pd.concat([train_df, valid_df], ignore_index=True).dropna(
        subset=config.data.label_cols
    ).reset_index(drop=True)

    # Filtra o DataFrame para garantir que todos os arquivos existem
    required_cols = config.data.image_cols + config.data.label_cols
    valid_mask = all_files_df.apply(check_files_exist, axis=1, base_dir=config.data.data_dir, required_cols=required_cols)
    all_files_df = all_files_df[valid_mask]
    print(f"Total de {len(all_files_df)} amostras válidas para validação cruzada.")
    
    N_SPLITS = 5
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=config.seed)
    fold_metrics = []

    # Loop principal da validação cruzada
    for fold, (train_idx, val_idx) in enumerate(kfold.split(all_files_df)):
        print(f"\n{'='*20} FOLD {fold + 1}/{N_SPLITS} {'='*20}")
        
        train_fold_df = all_files_df.iloc[train_idx]
        val_fold_df = all_files_df.iloc[val_idx]

        def create_file_list(df):
            data_dicts = []
            for _, row in df.iterrows():
                file_dict = {}
                # Junta todas as imagens em uma lista
                file_dict["image"] = [
                    os.path.join(config.data.data_dir, row[col])
                    for col in config.data.image_cols
                    if not pd.isna(row[col])
                ]
                # Junta todos os labels em uma lista
                file_dict["label"] = [
                    os.path.join(config.data.data_dir, row[col])
                    for col in config.data.label_cols
                    if not pd.isna(row[col])
                ]
                data_dicts.append(file_dict)
            return data_dicts

        train_files = create_file_list(train_fold_df)
        val_files = create_file_list(val_fold_df)

        original_run_id = config.run_id
        config.run_id = f"{original_run_id}_fold_{fold+1}"
        
        # Cria e treina um NOVO modelo do zero para este fold
        trainer = SegmentationTrainer(
            config=config,
            train_files=train_files,
            val_files=val_files
        )

        if 'OneCycleLR' in config.lr_scheduler:
            trainer.fit_one_cycle()
        
        trainer.run(try_resume_from_checkpoint=False)

        best_metric = trainer.evaluator.state.best_metric
        fold_metrics.append(best_metric)
        print(f"\n--- FIM DO FOLD {fold + 1} --- Melhor Dice: {best_metric:.4f} ---")
        config.run_id = original_run_id

    mean_metric = np.mean(fold_metrics)
    std_metric = np.std(fold_metrics)

    print(f"\n\n{'='*20} RESULTADOS FINAIS DA VALIDAÇÃO CRUZADA {'='*20}")
    print(f"Métricas de Dice para cada fold: {[f'{m:.4f}' for m in fold_metrics]}")
    print(f"Dice Médio (k={N_SPLITS}): {mean_metric:.4f}")
    print(f"Desvio Padrão: {std_metric:.4f}")
    print(f"\nResultado Final para o TCC: {mean_metric:.4f} ± {std_metric:.4f}")

if __name__ == '__main__':
    main()