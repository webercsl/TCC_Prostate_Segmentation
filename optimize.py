# tcc/optimize.py - NOVO ARQUIVO

import optuna
import argparse
import numpy as np 
from prostate158.utils import load_config
from prostate158.train import SegmentationTrainer

# Carregar config base
parser = argparse.ArgumentParser(description='Optimize hyperparameters.')
parser.add_argument('--config', type=str, required=True, help='path to the base config file')
parser.add_argument('--n_trials', type=int, default=100, help='number of optimization trials')
args = parser.parse_args()
config_fn = args.config

def objective(trial: optuna.Trial):
    config = load_config(config_fn)

    # 1. Sugerir hiperparâmetros
    config.optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Novograd"])
    config.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    config.model.dropout = trial.suggest_float("dropout", 0.0, 0.5)
    config.model.num_res_units = trial.suggest_int("num_res_units", 2, 4)
    # Adicione outros parâmetros que queira testar, ex: profundidade do modelo, etc.
    
    # Para passar o otimizador e lr para o trainer
    config.optimizer = {config.optimizer_name: {'lr': config.lr}}
    
    # 2. Executar um único treinamento (pode ser em apenas 1 fold ou na divisão original para velocidade)
    print(f"\n--- TRIAL {trial.number} ---")
    print(f"Params: {trial.params}")
    trainer = SegmentationTrainer(config=config, progress_bar=False) # Desligar a pbar para menos poluição visual
    
    # Use um scheduler se desejar, ou apenas treine com LR fixo
    # trainer.fit_one_cycle()
    trainer.run(try_resume_from_checkpoint=False)
    
    # 3. Retornar a métrica
    best_metric = trainer.evaluator.state.best_metric
    
    # Lidar com falhas (ex: NaN na loss)
    if best_metric is None or np.isnan(best_metric):
        return 0.0

    return best_metric

# 4. Iniciar o estudo
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=args.n_trials)

print("\n--- OPTIMIZATION FINISHED ---")
print("Melhor Trial:", study.best_trial.number)
print("Melhor Métrica (val_mean_dice):", study.best_value)
print("Melhores Hiperparâmetros:", study.best_params)