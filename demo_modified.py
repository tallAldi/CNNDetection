import os
import sys
import torch
import torch.nn
import argparse
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from networks.resnet import resnet50
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
import csv

def save_results_csv(
    output_path: str,
    y_true: List[int],
    y_pred: List[float]
) -> None:
    # Input validation
    assert isinstance(output_path, str), "output_path must be a string"
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
    assert len(y_true) > 0, "y_true must not be empty"

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write results
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred"])
        for true_val, pred_val in zip(y_true, y_pred):
            assert isinstance(true_val, (int, str)), "Invalid type in y_true"
            assert isinstance(pred_val, float), "Invalid type in y_pred"
            writer.writerow([true_val, pred_val])

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-d','--dir', nargs='+', type=Path, default=None)
group.add_argument('-f','--file', default=None)

parser.add_argument('-m','--model_path', type=str, default='weights/blur_jpg_prob0.5.pth')
parser.add_argument('-c','--crop', type=int, default=None, help='by default, do not crop. specify crop size')
parser.add_argument('--use_cpu', action='store_true', help='uses gpu by default, turn on to use cpu')
parser.add_argument('-o','--output_csv', type=str, default=None, help='if specified, write results to csv file')
parser.add_argument('-b','--batch_size', type=int, default=32)
parser.add_argument('-j','--workers', type=int, default=4, help='number of workers')

opt = parser.parse_args()

if opt.dir is not None:
  for d in opt.dir:
    assert(d.exists()), 'Directory %s does not exist'%str(d)
    assert(d.is_dir()), '%s is not a directory'%str(d)


model = resnet50(num_classes=1)
state_dict = torch.load(opt.model_path, map_location='cpu')
model.load_state_dict(state_dict['model'])
if(not opt.use_cpu):
  model.cuda()
model.eval()

# Transform
trans_init = []
if(opt.crop is not None):
  trans_init = [transforms.CenterCrop(opt.crop),]
  print('Cropping to [%i]'%opt.crop)
else:
  print('Not cropping')
trans = transforms.Compose(trans_init + [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if opt.file is not None:

  img = trans(Image.open(opt.file).convert('RGB'))

  with torch.no_grad():
      in_tens = img.unsqueeze(0)
      if(not opt.use_cpu):
          in_tens = in_tens.cuda()
      prob = model(in_tens).sigmoid().item()

  print('probability of being synthetic: {:.2f}%'.format(prob * 100))

  if opt.dir is not None:
    dataset = datasets.ImageFolder(opt.dir, transform=trans)
    
    data_loader=torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=False,
                                              num_workers=opt.workers)

    y_true, y_pred = [], []
    with torch.no_grad():
      for data, label in tqdm(data_loader):

        # Convert numeric labels to class names
        class_names = [dataset.classes[idx] for idx in label.flatten().tolist()]

        # Prefix each class name with opt.dir.name
        prefixed_names = [f"{opt.dir.name}_{name}" for name in class_names]
        y_true.extend(prefixed_names)
        if(not opt.use_cpu):
            data = data.cuda()
        y_pred.extend(model(data).sigmoid().flatten().tolist())
    
    if opt.output_csv is not None:
      save_results_csv(opt.output_csv / f'{opt.dir.name}', y_true, y_pred)
      print(f"Results saved to {opt.output_csv}")

