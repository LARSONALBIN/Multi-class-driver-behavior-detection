# Multi-Class Driver Behavior Classification

This repository implements training, evaluation, and visualization of a ResNet-50–based model to classify driver behaviors into five categories:  
- **other_activities**- **safe_driving**  
- **talking_phone**  
- **texting_phone**  
- **turning**  

## Features

- Data loading with augmentation (random flips, rotations)  
- Transfer learning using a ResNet-50 backbone  
- Custom classifier head for embedding → 5-class logits  
- Cosine annealing learning-rate schedule  
- Early stopping on validation loss  
- Training/validation loss & accuracy curves  
- Confusion matrix and classification report  
- t-SNE visualization of validation embeddings  

## Repository Structure

```
├── driver-1.ipynb           # End-to-end training & evaluation notebook  
├── best_model_no_gravity.pth# Checkpoint of the best model  
├── training_log_no_gravity.json  # Per-epoch loss/accuracy history  
└── README.md                # This file  
```

## Requirements

- Python 3.8+  
- PyTorch  
- torchvision  
- scikit-learn  
- matplotlib  
- seaborn  
- numpy  

Install via:
```
pip install torch torchvision scikit-learn matplotlib seaborn numpy
```

## Usage

1. **Dataset**  
   Download and place the “Multi-Class Driver Behavior Image Dataset” under:  
   `./data/Multi-Class Driver Behavior Image Dataset/`  

2. **Notebook**  
   Open `driver-1.ipynb` and update the `DATA_DIR` path if needed.  

3. **Training**  
   The notebook splits the dataset 80/20, trains for up to 50 epochs with early stopping (patience=5), and saves the best model to `best_model_no_gravity.pth`.  

4. **Evaluation**  
   After training, the notebook:
   - Prints validation accuracy, precision, recall, F1-score  
   - Displays a confusion matrix  
   - Plots training/validation loss & accuracy curves  
   - Generates a t-SNE plot of learned embeddings  

## Configuration Parameters

Adjust these in the notebook’s “User configuration” section:
```python
DATA_DIR = "./data/Multi-Class Driver Behavior Image Dataset"
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 5
LR = 3e-4
EMBED_DIM = 512
NUM_CLASS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Results

- **Validation Accuracy:** ~96.02%  
- **Precision / Recall / F1-score:** ~96% (weighted)  
- Detailed per-class metrics are printed in the classification report.  

## Visualization

- **Confusion Matrix:** shows class-wise performance  
- **Loss & Accuracy Curves:** tracks over epochs  
- **t-SNE Embeddings:** 2D projection of validation features  

## License

This project is released under the MIT License.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/9557945/c0596b79-c32d-47ae-9f0e-f57a66e038e6/driver-1.ipynb
