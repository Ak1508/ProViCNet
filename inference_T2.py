import argparse
import os
import torch
import yaml
from huggingface_hub import hf_hub_download

from ProViCNet.ModelArchitectures.Models import GetModel
from ProViCNet.util_functions.Prostate_DataGenerator import US_MRI_Generator
from ProViCNet.util_functions.utils_weighted import set_seed
from ProViCNet.util_functions.inference import (
    ProViCNet_data_preparation,
    visualize_max_cancer,
    saveData,
    visualize_featuremap,
    merge_cancer,
    keep_csPCa_only,
)
from ProViCNet.util_functions.train_functions import getPatchTokens


def load_weight_from_url(url, device):
    """Download the weight file from Hugging Face Hub and load it."""
    parts = url.split('/')
    repo_id = f"{parts[3]}/{parts[4]}"
    filename = parts[-1]
    weight_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return torch.load(weight_path, map_location=device)


def main(args):
    dataset = args.config['sample_dataset']

    test_generator = US_MRI_Generator(
        imageFileName=dataset['T2'],
        glandFileName=dataset['Gland'],
        cancerFileName=dataset['Cancer'],
        modality='MRI',
        cancerTo2=False,
        Augmentation=False,
        img_size=args.img_size,
        nChannel=args.nChannel,
    )

    model = GetModel(
        args.ModelName,
        args.nClass,
        args.nChannel,
        args.img_size,
        vit_backbone=args.vit_backbone,
        contrastive=args.contrastive,
    )
    model = model.to(args.device)

    state_dict = load_weight_from_url(args.config['model_weights']['T2'], args.device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    os.makedirs(args.visualization_folder, exist_ok=True)
    for sample_idx in range(len(test_generator)):
        image_t2, _, _, posit, label = ProViCNet_data_preparation(
            sample_idx, args, {'T2': test_generator, 'ADC': test_generator, 'DWI': test_generator}, modality='MRI'
        )
        tokens_t2 = getPatchTokens(model, image_t2, posit, args).to(args.device)

        patient_id = os.path.basename(test_generator.imageFileName[sample_idx]).split('_t2')[0]
        visualize_featuremap(
            tokens_t2,
            image_t2,
            label,
            os.path.join(args.visualization_folder, f'{patient_id}_featuremap_T2.png')
        )

    os.makedirs(args.save_folder, exist_ok=True)
    for sample_idx in range(len(test_generator)):
        image_t2, _, _, posit, label = ProViCNet_data_preparation(
            sample_idx, args, {'T2': test_generator, 'ADC': test_generator, 'DWI': test_generator}, modality='MRI'
        )

        with torch.no_grad():
            pred_t2 = model(image_t2, pos=posit).cpu()
            preds_t2_softmax = torch.softmax(pred_t2, dim=1)
            if args.only_csPCa:
                preds_t2_softmax = keep_csPCa_only(preds_t2_softmax)
            else:
                preds_t2_softmax = merge_cancer(preds_t2_softmax)

        patient_id = os.path.basename(test_generator.imageFileName[sample_idx]).split('_t2')[0]
        prob_filename = os.path.join(args.save_folder, f'{patient_id}_ProViCNet_T2_probability.nii.gz')
        saveData(preds_t2_softmax[:, 2], test_generator.imageFileName[sample_idx], prob_filename)

        label_filename = os.path.join(args.save_folder, f'{patient_id}_ProViCNet_T2_predLabel.nii.gz')
        saveData((preds_t2_softmax[:, 2] > args.threshold).float(), test_generator.imageFileName[sample_idx], label_filename)

        vis_filename = os.path.join(args.visualization_folder, f'{patient_id}_T2_visualization.png')
        visualize_max_cancer(
            image_t2,
            image_t2,
            image_t2,
            label,
            preds_t2_softmax,
            preds_t2_softmax,
            preds_t2_softmax,
            preds_t2_softmax,
            vis_filename,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T2-only Inference Script for ProViCNet")

    parser.add_argument('--ModelName', type=str, default="ProViCNet")
    parser.add_argument('--vit_backbone', type=str, default='dinov2_s_reg')
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--nClass', type=int, default=4)
    parser.add_argument('--nChannel', type=int, default=3)
    parser.add_argument('--contrastive', type=bool, default=True)

    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--only_csPCa', type=bool, default=False)

    parser.add_argument('--save_folder', type=str, default='results_ProViCNet/')
    parser.add_argument('--visualization_folder', type=str, default='visualization_ProViCNet/')
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--small_batchsize', type=int, default=16)
    parser.add_argument('--config_file', type=str, default='configs/config_infer_MRI.yaml')

    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    with open(args.config_file) as f:
        args.config = yaml.load(f, Loader=yaml.FullLoader)

    set_seed(42)
    main(args)
