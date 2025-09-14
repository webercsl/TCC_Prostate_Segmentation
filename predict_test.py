import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import monai

# Garante que o módulo prostate158 seja encontrado
sys.path.append(os.getcwd())

from prostate158.utils import load_config
from prostate158.model import get_model
from prostate158.transforms import get_val_transforms

def main():
    parser = argparse.ArgumentParser(description='Generate a visual comparison for a test case.')
    parser.add_argument('--config', required=True, type=str, help='Path to the config file used for training (e.g., anatomy_tcc1.yaml)')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to the trained model checkpoint (.pt file)')
    parser.add_argument('--test_case_id', required=True, type=int, help='The ID of the case from the test CSV to predict on.')
    parser.add_argument('--output_dir', default='test_comparisons', type=str, help='Directory to save the output PNG images.')
    parser.add_argument('--slice_idx', type=int, help='(Optional) Specific slice to visualize. If not given, a central slice is chosen automatically.')
    args = parser.parse_args()

    try:
        # --- 1. Carregar Configuração e Modelo ---
        print("--- PASSO 1: CARREGANDO CONFIGURAÇÃO E MODELO ---")
        config = load_config(args.config)
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        model = get_model(config=config).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.eval()
        print(f"Modelo '{args.checkpoint}' carregado e pronto para inferência no dispositivo: {device}")

        # --- 2. Carregar e Pré-processar os Dados do Paciente de Teste ---
        print(f"\n--- PASSO 2: PREPARANDO DADOS PARA O PACIENTE ID: {args.test_case_id} ---")
        df = pd.read_csv(config.data.test_csv)
        case_row = df[df['ID'] == args.test_case_id].iloc[0]

        image_key = config.data.image_cols[0]
        label_key = config.data.label_cols[0]

        image_path = os.path.join(config.data.data_dir, case_row[image_key])
        label_path = os.path.join(config.data.data_dir, case_row[label_key])

        input_dict = {image_key: image_path, label_key: label_path}

        transforms = get_val_transforms(config)
        processed_data = transforms(input_dict)
        input_tensor = processed_data[image_key].unsqueeze(0).to(device)
        print(f"Dados do paciente pré-processados. Shape do tensor de entrada: {input_tensor.shape}")

        # --- 3. Executar Inferência ---
        print("\n--- PASSO 3: EXECUTANDO PREDIÇÃO ---")
        with torch.no_grad():
            output_logits = monai.inferers.sliding_window_inference(
                inputs=input_tensor,
                roi_size=config.transforms.rand_crop_pos_neg_label.spatial_size,
                sw_batch_size=4,
                predictor=model,
                overlap=0.5
            )
        
        pred_mask_tensor = torch.argmax(output_logits, dim=1).squeeze(0)
        pred_mask_np = pred_mask_tensor.cpu().numpy()
        print("Predição concluída.")

        # --- 4. Preparar Arrays para Visualização ---
        image_np = processed_data[image_key].squeeze().numpy()
        label_np = processed_data[label_key].squeeze().numpy()

        # --- 5. Escolher Slice e Gerar Imagem PNG ---
        print("\n--- PASSO 4: GERANDO IMAGEM DE COMPARAÇÃO ---")
        if args.slice_idx is not None:
            slice_idx = args.slice_idx
            print(f"Usando slice especificado pelo usuário: {slice_idx}")
        else:
            # Lógica para encontrar um slice representativo (com base no gabarito)
            slice_sums = np.sum(label_np, axis=(1, 2))
            non_empty_slices = np.where(slice_sums > 0)[0]
            slice_idx = non_empty_slices[len(non_empty_slices) // 2] if len(non_empty_slices) > 0 else label_np.shape[0] // 2
            print(f"Slice escolhido automaticamente: {slice_idx}")

        image_slice = image_np[slice_idx, :, :]
        label_slice = label_np[slice_idx, :, :]
        pred_slice = pred_mask_np[slice_idx, :, :]

        fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=150)
        
        axes[0].imshow(image_slice, cmap='gray')
        axes[0].set_title(f'Imagem Original (Slice {slice_idx})')
        axes[0].axis('off')

        axes[1].imshow(image_slice, cmap='gray')
        axes[1].imshow(np.ma.masked_where(label_slice == 0, label_slice), cmap='autumn', alpha=0.6)
        axes[1].set_title('Gabarito (Máscara Real)')
        axes[1].axis('off')

        axes[2].imshow(image_slice, cmap='gray')
        axes[2].imshow(np.ma.masked_where(pred_slice == 0, pred_slice), cmap='cool', alpha=0.6)
        axes[2].set_title('Predição do Modelo')
        axes[2].axis('off')

        fig.suptitle(f'Comparação para o Paciente de Teste ID: {args.test_case_id}', fontsize=16)
        plt.tight_layout()

        os.makedirs(args.output_dir, exist_ok=True)
        
        # Gera um nome de arquivo descritivo
        checkpoint_name = os.path.basename(args.checkpoint).replace(".pt", "")
        output_filename = os.path.join(args.output_dir, f"comparison_ID_{args.test_case_id}_model_{checkpoint_name}.png")

        plt.savefig(output_filename)
        plt.close(fig)

        print(f"\nSUCESSO! Imagem de comparação salva em: {output_filename}")

    except Exception as e:
        print(f"\n--- OCORREU UM ERRO ---")
        print(e)
        raise

if __name__ == '__main__':
    main()