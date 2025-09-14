import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Export a comparison slice from a .pt prediction tensor to a .png file.')
    parser.add_argument(
        '--input_dir', 
        required=True, 
        type=str, 
        help='Path to the directory containing the saved predictions (e.g., "anatomy_simple_500_epochs/output/preds")'
    )
    parser.add_argument(
        '--pred_filename', 
        required=True, 
        type=str, 
        help='The filename of the prediction tensor to convert (e.g., "pred_epoch_410.pt")'
    )
    parser.add_argument(
        '--output_dir', 
        default=None, 
        type=str, 
        help='(Optional) Directory to save the output PNG. Defaults to the input directory.'
    )
    parser.add_argument(
        '--slice_idx', 
        default=None, 
        type=int, 
        help='(Optional) Specific slice index to export. If not provided, a central slice with content will be chosen automatically.'
    )
    args = parser.parse_args()

    # --- 1. Definir Caminhos ---
    image_ref_path = os.path.join(args.input_dir, "image.pt")
    label_ref_path = os.path.join(args.input_dir, "label.pt")
    prediction_path = os.path.join(args.input_dir, args.pred_filename)

    # Define o diretório de saída
    output_dir = args.output_dir if args.output_dir else args.input_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 2. Carregar os Arquivos .pt ---
    try:
        prediction_tensor = torch.load(prediction_path, map_location='cpu')
        image_metatensor = torch.load(image_ref_path, map_location='cpu')
        label_metatensor = torch.load(label_ref_path, map_location='cpu')
    except FileNotFoundError as e:
        print(f"ERRO: Arquivo não encontrado. Verifique os caminhos. {e}")
        return

    # --- 3. Processar Tensores para Visualização ---
    
    # Converte a predição em uma máscara (C,H,W,D) -> (H,W,D)
    prediction_mask = torch.argmax(prediction_tensor, dim=0).numpy()
    
    # Converte o label em uma máscara
    label_mask = torch.argmax(label_metatensor, dim=0).numpy()

    # Pega a imagem e remove a dimensão do canal (1,H,W,D) -> (H,W,D)
    image_array = image_metatensor.squeeze(0).numpy()

    # --- 4. Escolher o Slice a ser Exportado ---
    if args.slice_idx is not None:
        slice_idx = args.slice_idx
        print(f"Usando slice especificado pelo usuário: {slice_idx}")
    else:
        # Lógica automática para encontrar um slice interessante (com o maior conteúdo no label)
        slice_sums = np.sum(label_mask, axis=(1, 2))
        non_empty_slices = np.where(slice_sums > 0)[0]
        if len(non_empty_slices) > 0:
            # Pega o slice do meio dentre os que têm conteúdo
            slice_idx = non_empty_slices[len(non_empty_slices) // 2]
        else:
            # Se o label estiver vazio, pega o slice central do volume
            slice_idx = label_mask.shape[0] // 2
        print(f"Slice escolhido automaticamente: {slice_idx}")

    # --- 5. Gerar e Salvar a Imagem de Comparação ---
    
    # Pega os slices 2D dos arrays 3D
    image_slice = image_array[slice_idx, :, :]
    label_slice = label_mask[slice_idx, :, :]
    pred_slice = prediction_mask[slice_idx, :, :]

    # Cria a figura com 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), dpi=150) # dpi=150 para maior resolução
    
    # Plot 1: Imagem Original
    axes[0].imshow(image_slice, cmap='gray')
    axes[0].set_title(f'Imagem Original (Slice {slice_idx})')
    axes[0].axis('off')

    # Plot 2: Ground Truth (Label)
    axes[1].imshow(image_slice, cmap='gray')
    axes[1].imshow(np.ma.masked_where(label_slice == 0, label_slice), cmap='autumn', alpha=0.6)
    axes[1].set_title('Gabarito (Máscara Real)')
    axes[1].axis('off')

    # Plot 3: Predição do Modelo
    axes[2].imshow(image_slice, cmap='gray')
    axes[2].imshow(np.ma.masked_where(pred_slice == 0, pred_slice), cmap='cool', alpha=0.6)
    axes[2].set_title('Predição do Modelo')
    axes[2].axis('off')

    fig.suptitle(f'Comparação de Predição - {args.pred_filename}', fontsize=16)
    plt.tight_layout()

    # Define o nome do arquivo de saída
    base_name = args.pred_filename.replace(".pt", "")
    output_filename = os.path.join(output_dir, f"{base_name}_slice_{slice_idx}.png")

    # Salva a figura
    plt.savefig(output_filename)
    plt.close(fig) # Libera a memória

    print(f"\nSUCESSO! Imagem de comparação salva em: {output_filename}")

if __name__ == '__main__':
    main()