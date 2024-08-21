import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from PIL import Image
import numpy as np

class ProteinDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, mlb=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.mlb = mlb
        
        # Prepare GO IDs
        go_ids = self.data['Gene Ontology IDs'].fillna('').str.split('; ')
        # Remove empty strings
        go_ids = go_ids.apply(lambda x: [item for item in x if item])
        
        # Filter out entries with missing images
        valid_entries = []
        valid_go_ids = []
        for idx, row in self.data.iterrows():
            img_name = f"AF-{row['Entry']}-F1-model_v4_matrix_with_pdc.png"
            img_path = os.path.join(self.img_dir, img_name)
            if os.path.exists(img_path):
                valid_entries.append(idx)
                valid_go_ids.append(go_ids[idx])
            else:
                logging.warning(f"Image not found for entry {row['Entry']}, skipping.")
        
        self.data = self.data.iloc[valid_entries]
        
        if self.mlb is None:
            self.mlb = MultiLabelBinarizer()
            self.labels = self.mlb.fit_transform(valid_go_ids)
        else:
            self.labels = self.mlb.transform(valid_go_ids)
        
        logging.info(f"Loaded {len(self.data)} valid entries out of {len(go_ids)} total entries.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = f"AF-{self.data.iloc[idx]['Entry']}-F1-model_v4_matrix_with_pdc.png"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.FloatTensor(self.labels[idx])
        return image, label

class ProteinClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ProteinClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        return self.sigmoid(x)

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

def evaluate_model(model, test_loader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (outputs > threshold).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='micro')
    print(f'F1 Score: {f1}')
    return f1

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        train_csv = 'datasets/dataset1/ecoliK12_5star_080124.csv'
        train_img_dir = 'datasets/dataset1/images'
        test_csv = 'datasets/dataset2/pputida_all_080524.csv'
        test_img_dir = 'datasets/dataset2/images'

        # Set up data transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create datasets and dataloaders
        logging.info("Loading training dataset...")
        train_dataset = ProteinDataset(train_csv, train_img_dir, transform)
        
        logging.info("Loading test dataset...")
        test_dataset = ProteinDataset(test_csv, test_img_dir, transform, mlb=train_dataset.mlb)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

        # Set up model, loss function, and optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        
        num_classes = len(train_dataset.mlb.classes_)
        logging.info(f"Number of classes: {num_classes}")
        
        model = ProteinClassifier(num_classes).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        logging.info("Starting model training...")
        train_model(model, train_loader, criterion, optimizer, device)

        # Evaluate the model
        logging.info("Evaluating model...")
        f1_score = evaluate_model(model, test_loader, device)

        # Save the model
        logging.info("Saving model...")
        torch.save(model.state_dict(), 'protein_classifier.pth')

        # Save the MultiLabelBinarizer
        import joblib
        joblib.dump(train_dataset.mlb, 'mlb.joblib')
        logging.info("Model and MultiLabelBinarizer saved successfully.")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()