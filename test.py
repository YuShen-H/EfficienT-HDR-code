import argparse
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio

from dataset.dataset_sig17V1 import SIG17_Test_Dataset
from models import MODEL_VARIANTS
from utils.checkpoints import load_model_weights
from utils.evaluation import predict_scene
from utils.utils import AverageMeter, calculate_ssim, range_compressor, save_hdr


DEFAULT_CHECKPOINTS = {
    'main': 'checkpoints/V11_1/best_checkpoint.pth',
    'lite': 'checkpoints/V10-3/best_checkpoint.pth',
}


def main():
    parser = argparse.ArgumentParser(description='Evaluate EfficienT-HDR on SIG17 scenes')
    parser.add_argument('--model', choices=MODEL_VARIANTS, default='lite')
    parser.add_argument('--dataset_dir', default='./data')
    parser.add_argument('--test_split', help='Subdirectory under <dataset_dir>/Test')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--use_dynamic_tanh', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--dyt_alpha_init_value', type=float, default=0.5)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--pretrained_model', type=Path, help='Path to a checkpoint for the selected model')
    parser.add_argument('--save_results', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--save_dir', type=Path, default=Path('results/hdr_transformer'))
    args = parser.parse_args()

    checkpoint_path = args.pretrained_model or Path(DEFAULT_CHECKPOINTS[args.model])
    if not checkpoint_path.is_file():
        parser.error(f'Checkpoint not found: {checkpoint_path}. Pass its path with --pretrained_model.')

    device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    model = MODEL_VARIANTS[args.model](
        img_size=args.patch_size, mlp_ratio=2, use_dynamic_tanh=args.use_dynamic_tanh,
        dyt_alpha_init_value=args.dyt_alpha_init_value,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    load_model_weights(model, checkpoint['state_dict'])
    model.eval()
    print(f'Evaluating {args.model} on {device} with {checkpoint_path}')

    datasets = SIG17_Test_Dataset(args.dataset_dir, args.patch_size, args.test_split)
    psnr_l = AverageMeter()
    ssim_l = AverageMeter()
    psnr_mu = AverageMeter()
    ssim_mu = AverageMeter()
    if args.save_results:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_dataset in enumerate(datasets):
        pred, label = predict_scene(model, img_dataset, device, args.test_batch_size, args.num_workers)
        label_mu = range_compressor(label)
        pred_mu = range_compressor(pred)
        psnr_l.update(peak_signal_noise_ratio(label, pred, data_range=1.0))
        psnr_mu.update(peak_signal_noise_ratio(label_mu, pred_mu, data_range=1.0))
        ssim_l.update(image_ssim(pred, label))
        ssim_mu.update(image_ssim(pred_mu, label_mu))

        if args.save_results:
            save_hdr(args.save_dir / f'{idx}_pred.hdr', pred.transpose(1, 2, 0)[..., ::-1])

    print(f'Average PSNR_mu: {psnr_mu.avg:.4f}  PSNR_l: {psnr_l.avg:.4f}')
    print(f'Average SSIM_mu: {ssim_mu.avg:.4f}  SSIM_l: {ssim_l.avg:.4f}')


def image_ssim(pred, label):
    pred = np.clip(pred * 255.0, 0.0, 255.0).transpose(1, 2, 0)
    label = np.clip(label * 255.0, 0.0, 255.0).transpose(1, 2, 0)
    return calculate_ssim(pred, label)


if __name__ == '__main__':
    main()
