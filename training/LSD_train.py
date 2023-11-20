import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from LSD_model import get_LSD_model
from LabelStudio_LSD_Dataset import LS2BlinkerDetectionDataset

from tqdm import tqdm

# https://github.com/pytorch/vision/tree/main/references/detection
from utils import collate_fn
from engine import train_one_epoch, evaluate

writer = SummaryWriter()

torch.cuda.empty_cache()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dataset_json_path_train = './datasets/Apollo3D/labelstudio_LSD_annotations_train.json'
dataset_json_path_test = './datasets/Apollo3D/labelstudio_LSD_annotations_val.json'
path_to_project_image_dir = './datasets/Apollo3D/'

dataset_train = LS2BlinkerDetectionDataset(dataset_json_path=dataset_json_path_train, path_to_project_image_dir=path_to_project_image_dir)
dataset_test = LS2BlinkerDetectionDataset(dataset_json_path=dataset_json_path_test, path_to_project_image_dir=path_to_project_image_dir)

data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, collate_fn=collate_fn)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

weights_path = "./weights/LSD_weights_Apollo3D_all_off_100.pth" # None # "./weights/LSD_weights_latest.pth" # Set to None to train from scratch
model = get_LSD_model(weights_path=weights_path, device=device)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=5e-4, weight_decay=0.0005)
num_epochs = 1000

for epoch in tqdm(range(num_epochs)):
    # train_one_epoch(model, optimizer, data_loader_train, device, epoch, writer, print_freq=1000)
    evaluate(model, data_loader_test, device, epoch, writer)
    break
    
torch.save(model.state_dict(), './weights/LSD_weights_latest.pth')
