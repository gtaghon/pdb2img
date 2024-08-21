import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import label_ranking_average_precision_score
import logging

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

class ProteinDataset(Dataset):
    def __init__(self, data, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Filter out entries with missing images
        self.data = self._filter_existing_images(data)

    def _filter_existing_images(self, data):
        existing_data = []
        for _, row in data.iterrows():
            img_name = f"AF-{row['Entry']}-F1-model_v4_matrix_with_pdc.png"
            img_path = os.path.join(self.img_dir, img_name)
            if os.path.exists(img_path):
                existing_data.append(row)
            else:
                logging.warning(f"Image not found: {img_path}")
        return pd.DataFrame(existing_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = f"AF-{self.data.iloc[idx]['Entry']}-F1-model_v4_matrix_with_pdc.png"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.FloatTensor(self.data.iloc[idx]['labels'])
        return image, label

class ProteinClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ProteinClassifier, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

def train(args, model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and args.local_rank == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def validate(args, model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            all_outputs.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    val_loss /= len(val_loader)
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    lrap = label_ranking_average_precision_score(all_targets, all_outputs)
    
    if args.local_rank == 0:
        print(f'Validation set: Average loss: {val_loss:.4f}, LRAP: {lrap:.4f}')
    return lrap

def find_optimal_batch_size(model, train_dataset, init_batch_size=4, max_batch_size=1024, step_factor=2):
    device = next(model.parameters()).device
    optimal_batch_size = init_batch_size

    while optimal_batch_size <= max_batch_size:
        try:
            # Create a small subset of the data for testing
            test_loader = DataLoader(
                Subset(train_dataset, range(optimal_batch_size * 2)),
                batch_size=optimal_batch_size, num_workers=4, pin_memory=True
            )
            
            # Try to process a batch
            for batch in test_loader:
                inputs, _ = batch
                inputs = inputs.to(device)
                outputs = model(inputs)
                break

            # If successful, increase batch size
            optimal_batch_size *= step_factor
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # If we're out of memory, revert to the last successful batch size
                optimal_batch_size //= step_factor
                break
            else:
                raise e

    return optimal_batch_size

def main():
    parser = argparse.ArgumentParser(description='Protein Function Classifier')
    parser.add_argument('--local-rank', type=int, default=-1, metavar='N',
                        help='Local process rank.')  # Add this line
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--max-classes', type=int, default=100,
                        help='maximum number of classes to use (default: 100)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--data-path', type=str, required=True,
                        help='path to the CSV file containing protein data')
    parser.add_argument('--img-dir', type=str, required=True,
                        help='path to the directory containing protein images')
    args = parser.parse_args()

    # Use the local_rank from args instead of dist.get_rank()
    args.local_rank = int(os.environ['LOCAL_RANK'])  # Add this line

    dist.init_process_group(backend='nccl')
    args.world_size = dist.get_world_size()
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
    
    # Load and preprocess data
    data = pd.read_csv(args.data_path)
    mlb = MultiLabelBinarizer()
    # Handle NaN values and split GO IDs
    go_ids = data['Gene Ontology IDs'].fillna('').astype(str).str.split('; ')
    # Remove empty strings from each list
    go_ids = go_ids.apply(lambda x: [item for item in x if item])
    labels = mlb.fit_transform(go_ids)
    logging.info(f"Found {len(mlb.classes_)} unique GO IDs")
    
    if args.max_classes < len(mlb.classes_):
        top_classes = np.argsort(labels.sum(axis=0))[-args.max_classes:]
        labels = labels[:, top_classes]
        mlb.classes_ = mlb.classes_[top_classes]
        logging.info(f"Limited to top {args.max_classes} GO IDs")
    
    data['labels'] = list(labels)
    logging.info(f"Processed labels for {len(data)} entries")

    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ProteinDataset(train_data, args.img_dir, transform)
    val_dataset = ProteinDataset(val_data, args.img_dir, transform)
    test_dataset = ProteinDataset(test_data, args.img_dir, transform)

    model = ProteinClassifier(len(mlb.classes_)).to(device)
    model = DDP(model, device_ids=[args.local_rank])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalLoss()

    # Find optimal batch size
    if args.local_rank == 0:
        logging.info("Finding optimal batch size...")
        optimal_batch_size = find_optimal_batch_size(model.module, train_dataset)
        logging.info(f"Optimal batch size: {optimal_batch_size}")
    else:
        optimal_batch_size = None

    # Broadcast optimal batch size to all processes
    optimal_batch_size = torch.tensor(optimal_batch_size if optimal_batch_size is not None else 0).to(device)
    dist.broadcast(optimal_batch_size, src=0)
    optimal_batch_size = optimal_batch_size.item()

    # Use the optimal batch size
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=optimal_batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=optimal_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=optimal_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    best_lrap = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, criterion, epoch, device)
        lrap = validate(args, model, val_loader, criterion, device)
        
        if args.local_rank == 0 and lrap > best_lrap:
            best_lrap = lrap
            if args.save_model:
                torch.save(model.state_dict(), "protein_classifier.pt")
    
    if args.local_rank == 0:
        print(f'Best validation LRAP: {best_lrap:.4f}')

if __name__ == '__main__':
    main()