# Rice Leaf Disease Classification Using DenseNet121

**A deep learning-based image classification pipeline for identifying seven rice leaf disease categories using transfer learning, hyperparameter optimization, and test-time augmentation.**

## Overview

This project develops a multi-class rice leaf disease classification system using **DenseNet121 with ImageNet-pretrained weights**. The implementation focuses on improving classification performance through higher-resolution image processing, class imbalance handling, systematic hyperparameter optimization, and a two-phase fine-tuning strategy.

The pipeline also incorporates Test-Time Augmentation (TTA), mixed-precision training, and automated learning-rate scheduling to improve inference robustness and training efficiency.

## Key Features

- **Transfer Learning:** DenseNet121 pretrained on ImageNet, with a custom classification head.
- **High-Resolution Processing:** Uses 288 × 384 input images to preserve spatial details.
- **Data Augmentation:** Random flipping, rotation, zoom, translation, brightness, and contrast transformations.
- **Class Imbalance Handling:** Uses softened class weights based on balanced class frequencies.
- **Two-Phase Fine-Tuning:** Initially trains the classification head with a frozen backbone, followed by full-backbone fine-tuning.
- **Hyperparameter Optimization:** Evaluates eight configurations using validation performance.
- **Mixed-Precision Training:** Uses mixed float16 computation when a compatible GPU is available.
- **Test-Time Augmentation:** Averages predictions from original, horizontally flipped, vertically flipped, and doubly flipped images.

## Dataset

**Source:** [Kaggle - Paddy Disease Classification](https://www.kaggle.com/competitions/paddy-disease-classification/data)

The project uses seven disease categories from the training dataset.

| Disease Category | Number of Images |
|---|---:|
| Blast | 1,738 |
| Tungro | 1,088 |
| Brown Spot | 965 |
| Downy Mildew | 620 |
| Bacterial Leaf Blight | 479 |
| Bacterial Leaf Streak | 380 |
| Bacterial Panicle Blight | 337 |
| **Total** | **5,607** |

### Dataset Splitting

The dataset is divided using stratified sampling:

| Subset | Images | Percentage |
|---|---:|---:|
| Training | 4,036 | 72% |
| Validation | 449 | 8% |
| Testing | 1,122 | 20% |
| **Total** | **5,607** | **100%** |

The validation set is separated from the training portion, keeping the held-out test set independent of training and model-selection callbacks.

## Model Architecture

The classification architecture is based on DenseNet121 with ImageNet-pretrained weights.

The custom classification head consists of:

1. Global Average Pooling
2. Dropout
3. Fully Connected Layer (512 neurons, ReLU activation)
4. Dropout
5. Output Layer (7 neurons, Softmax activation)

## Training Strategy

### Phase 1: Classification Head Training

- Freeze the pretrained DenseNet121 backbone.
- Train the newly added classification head.
- Use the Adam optimizer with the selected learning rate.

### Phase 2: Fine-Tuning

- Unfreeze the complete DenseNet121 backbone.
- Fine-tune the network using a lower learning rate.
- Apply early stopping and adaptive learning-rate reduction.

### Hyperparameter Optimization

Eight hyperparameter configurations are evaluated using the validation set. The selected configuration includes learning rates, dropout probabilities, dense-layer size, and label-smoothing settings.

## Experimental Results

The recorded final experiment achieved the following results on the held-out test set:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 93.58% |
| Sensitivity | 95.24% |
| Specificity | 98.89% |
| Precision | 93.94% |
| F1-score | 94.50% |

The best hyperparameter configuration achieved 95.55% validation accuracy during the tuning experiments.

*These values correspond to the recorded notebook run and are not a cross-validation estimate.*

## Technologies Used

- Python
- TensorFlow
- Keras
- DenseNet121
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- SciPy

## Repository Structure

```text
rice-leaf-disease-classification/
│
├── README.md
├── rice_leaf_disease_classification.ipynb
└── LICENSE
```

## How to Run

### Recommended Environment: Kaggle Notebooks

1. Open the Kaggle Paddy Disease Classification competition.
2. Accept the competition rules and ensure dataset access.
3. Create a Kaggle notebook.
4. Attach the Paddy Disease Classification dataset.
5. Upload or import this repository's notebook.
6. Run the notebook cells sequentially.

### Local Environment

Install the required libraries:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn scipy kaggle
```

For local execution, download the dataset through Kaggle and configure the Kaggle API credentials. Ensure that the dataset directory is accessible to the notebook.

A compatible GPU is recommended for efficient training.

## Future Improvements

- Conduct five-fold stratified cross-validation.
- Evaluate generalization on independent rice disease datasets.
- Investigate model explainability using Grad-CAM.
- Compare DenseNet121 with alternative CNN and vision transformer architectures.
- Explore model compression and deployment for real-world agricultural applications.

## Author

**Syed Nazmus Sakib**

Machine Learning | Deep Learning | Computer Vision | Artificial Intelligence

[GitHub Profile](https://github.com/sn-sakib)
