# tcc/optimize_anatomy.py - VERSÃO CORRIGIDA

import optuna
import argparse
import numpy as np
import monai  # Importar monai para acessar o handler
from prostate158.utils import load_config
from prostate158.train import SegmentationTrainer

parser = argparse.ArgumentParser(description='Optimize anatomy segmentation hyperparameters.')
parser.add_argument('--config', default='anatomy.yaml', type=str, help='Base config file')
parser.add_argument('--n_trials', type=int, default=5, help='Number of trials')
args = parser.parse_args()

def objective(trial: optuna.Trial):
    config = load_config(args.config)

    # 1. Sugerir hiperparâmetros para otimizar
    config.optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Novograd"])
    config.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    config.model.dropout = trial.suggest_float("dropout", 0.1, 0.5)
    config.model.num_res_units = trial.suggest_int("num_res_units", 2, 4)
    initial_channels = trial.suggest_categorical("initial_channels", [16, 24, 32])
    config.model.channels = [initial_channels * (2**i) for i in range(len(config.model.strides) + 1)]
    
    config.optimizer = {config.optimizer_name: {'lr': config.lr}}

    # --- INÍCIO DA CORREÇÃO ---
    # Em vez de modificar o handler depois, alteramos o config ANTES.
    # O Trainer vai ler estes valores ao ser criado.
    config.training.max_epochs = 1
    config.training.early_stopping_patience = 15
    # --- FIM DA CORREÇÃO ---

    # 2. Executar um treinamento rápido para avaliação
    print(f"\n--- TRIAL {trial.number}: PARAMS {trial.params} ---")
    trainer = SegmentationTrainer(config=config, progress_bar=False)
    
    # A linha que causava o erro foi removida.
    # trainer.evaluator.get_handler(...).patience = 15 # <--- LINHA REMOVIDA
    
    if hasattr(config, 'lr_scheduler') and 'OneCycleLR' in config.lr_scheduler:
      trainer.fit_one_cycle()
      
    trainer.run(try_resume_from_checkpoint=False)
    
    best_metric = trainer.evaluator.state.best_metric
    
    if best_metric is None or np.isnan(best_metric):
        return 0.0

    return best_metric

# 3. Iniciar o estudo
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=args.n_trials)

print("\n--- OPTIMIZATION FINISHED ---")
print("Melhor Trial:", study.best_trial.number)
print("Melhor Métrica (val_mean_dice):", study.best_value)
print("Melhores Hiperparâmetros:", study.best_params)