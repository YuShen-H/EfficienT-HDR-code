import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def predict_scene(model, img_dataset, device, batch_size=1, num_workers=1):
    loader = DataLoader(img_dataset, batch_size=batch_size, num_workers=num_workers)
    for batch in tqdm(loader, leave=False):
        output = model(batch['input0'].to(device), batch['input1'].to(device), batch['input2'].to(device))
        for patch in output.cpu().numpy().astype(np.float32):
            img_dataset.update_result(patch)
    return img_dataset.rebuild_result()
