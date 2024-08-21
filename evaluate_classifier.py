import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import hamming_loss, jaccard_score, precision_recall_fscore_support, accuracy_score, coverage_error, label_ranking_loss, label_ranking_average_precision_score
import numpy as np
import joblib
import logging
from classifier2 import ProteinDataset  # Make sure to import your dataset class
from classifier2 import ProteinClassifier  # Make sure to import your model class

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model, test_loader, device, mlb):
    model.eval()
    all_preds = []
    all_labels = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()  # Use 0.5 as threshold
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_outputs = np.vstack(all_outputs)
    
    # Calculate metrics
    hl = hamming_loss(all_labels, all_preds)
    js = jaccard_score(all_labels, all_preds, average='samples', zero_division=1)
    
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro', zero_division=1)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=1)
    
    subset_accuracy = accuracy_score(all_labels, all_preds)
    
    coverage = coverage_error(all_labels, all_outputs)
    ranking_loss = label_ranking_loss(all_labels, all_outputs)
    lrap = label_ranking_average_precision_score(all_labels, all_outputs)
    
    # Log results
    logging.info(f"Hamming Loss: {hl}")
    logging.info(f"Jaccard Score (Multi-label Accuracy): {js}")
    logging.info(f"Micro-averaged Precision: {precision_micro}")
    logging.info(f"Micro-averaged Recall: {recall_micro}")
    logging.info(f"Micro-averaged F1-score: {f1_micro}")
    logging.info(f"Macro-averaged Precision: {precision_macro}")
    logging.info(f"Macro-averaged Recall: {recall_macro}")
    logging.info(f"Macro-averaged F1-score: {f1_macro}")
    logging.info(f"Subset Accuracy (Exact Match Ratio): {subset_accuracy}")
    logging.info(f"Coverage Error: {coverage}")
    logging.info(f"Ranking Loss: {ranking_loss}")
    logging.info(f"Label Ranking Average Precision Score: {lrap}")
    
    return {
        'hamming_loss': hl,
        'jaccard_score': js,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'subset_accuracy': subset_accuracy,
        'coverage_error': coverage,
        'ranking_loss': ranking_loss,
        'lrap': lrap
    }

def main():
    # Load the saved model and MultiLabelBinarizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mlb = joblib.load('mlb.joblib')
    num_classes = len(mlb.classes_)
    
    model = ProteinClassifier(num_classes).to(device)
    model.load_state_dict(torch.load('protein_classifier.pth', map_location=device))
    
    # Set up data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_csv = 'datasets/mm/musculus.csv'
    test_img_dir = 'datasets/mm/images'
    test_dataset = ProteinDataset(test_csv, test_img_dir, transform, mlb=mlb)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Evaluate the model
    logging.info("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device, mlb)
    
    # You can now use the 'metrics' dictionary for further analysis or reporting

if __name__ == "__main__":
    main()