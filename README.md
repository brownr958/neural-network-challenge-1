# neural-network-challenge-1

# Student Loan Repayment Prediction with Deep Learning

## Overview
This project applies deep learning to predict student loan repayment likelihood based on borrower characteristics. Using TensorFlow/Keras, we trained a neural network model on historical student loan data to classify credit ranking. The model helps financial institutions determine risk levels and set appropriate interest rates for borrowers.

## Dataset
The dataset contains information about student loan recipients, including:
- Demographics (e.g., age, income)
- Financial history (e.g., existing loans, credit ranking)
- Loan details (e.g., amount, term length)
  
The target variable (y) is "credit_ranking," while all other columns form the feature set (X).

## Setup Instructions

### Clone the Repository
```bash
git clone https://github.com/your-username/neural-network-challenge-1.git
cd neural-network-challenge-1
```

### Install Dependencies
```bash
pip install pandas tensorflow scikit-learn
```

### Run the Jupyter Notebook
- Open `student_loans_with_deep_learning.ipynb` in Google Colab or Jupyter Notebook.
- Execute all cells to train and evaluate the model.

## Methodology

### Data Preprocessing
1. Loaded dataset from:  
   [student-loans.csv](https://static.bc-edx.com/ai/ail-v-1-0/m18/lms/datasets/student-loans.csv)
2. Separated features (X) and target (y).
3. Split data into training (80%) and testing (20%) sets.
4. Scaled features using StandardScaler to normalize inputs.

### Model Architecture
- Implemented a deep neural network using TensorFlow/Keras:
  - Two hidden layers (ReLU activation)
  - Binary classification output (Sigmoid activation)
  - Loss function: binary_crossentropy
  - Optimizer: adam
  - Metrics: accuracy
- Trained model for 50 epochs.

### Model Evaluation
- Calculated loss and accuracy on the test set.
- Generated a classification report to assess performance.

## Results
- Model achieved competitive accuracy on test data.
- Model successfully classified loan repayment likelihood.
- Model saved as `student_loans.keras` for reuse.

## Recommendation System Discussion

### Data Needed
- Borrower demographics, income, existing loans, repayment history, and other financial indicators.

### Best Filtering Method
- Content-based filtering is ideal, as it personalizes recommendations based on borrower characteristics.

### Real-World Challenges
1. Data privacy and security – Protecting sensitive financial data.
2. Bias in loan recommendations – Ensuring fairness in predictions.

## Files in This Repository
| File Name | Description |
|-----------|-------------|
| `student_loans_with_deep_learning.ipynb` | Jupyter Notebook with full implementation |
| `student_loans.keras` | Saved deep learning model |
| `README.md` | Project documentation |

## Next Steps
- Optimize the model by fine-tuning hyperparameters.
- Experiment with different architectures, such as adding dropout layers.
- Deploy the model for real-time loan risk assessment.