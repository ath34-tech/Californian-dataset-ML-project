# Machine Learning Pipeline with Scikit-Learn

This project implements a comprehensive machine learning pipeline in Python using Scikit-Learn. It includes classes for data preprocessing, model training, evaluation, and fine-tuning.

## Classes Overview

### 1. `trainTestSplitter`

- **Purpose**: Splits data into training and testing sets.
- **Parameters**: 
  - `test_size`: Proportion of the dataset to include in the test split.
  - `random_state`: Seed used by the random number generator for reproducibility.

### 2. `DataProcessor`

- **Purpose**: Processes both numeric and categorical data:
  - Imputes missing values.
  - Scales numeric features using Min-Max scaling.
  - Encodes categorical features using one-hot encoding.
- **Parameters**: 
  - `imputer_strategy`: Strategy used for imputing missing values ('mean', 'median', 'most_frequent').

### 3. `PredictingTransformer`

- **Purpose**: Fits and predicts using specified regression models:
  - Linear Regression
  - Decision Tree Regression
  - Random Forest Regression
  - Support Vector Regression (SVM)
- **Parameters**: 
  - `model_name`: Name of the regression model to use ('linear', 'tree', 'forest', 'svm').

### 4. `FineTunerTransformer`

- **Purpose**: Performs model evaluation and fine-tuning:
  - Computes cross-validation scores (RMSE).
  - Fine-tunes hyperparameters using Grid Search.
- **Parameters**: 
  - `param`: Hyperparameter grid for Grid Search.
  - `cv_finetune`: Number of folds for cross-validation during hyperparameter tuning.
  - `cv`: Number of folds for cross-validation during model evaluation.

### 5. `CompleteTransformer`

- **Purpose**: Integrates all components into an end-to-end workflow:
  - Splits data into training and testing sets.
  - Processes data using `DataProcessor`.
  - Trains models using `PredictingTransformer`.
  - Evaluates and fine-tunes models using `FineTunerTransformer`.
- **Parameters**: 
  - `df`: Input DataFrame containing both features and target.
  - `target`: Name of the target column in `df`.
  - `finetune_params`: Hyperparameter grid for Grid Search in `FineTunerTransformer`.
  - `imputer_strategy`: Strategy used for imputing missing values.
  - `splitting`: Proportion of the dataset to include in the test split.
  - `random_state`: Seed used by the random number generator for reproducibility.
  - `model_name`: Name of the regression model to use ('linear', 'tree', 'forest', 'svm').
  - `cv`: Number of folds for cross-validation during model evaluation.
  - `finetune_cv`: Number of folds for cross-validation during hyperparameter tuning.

## Usage Example

```python
import pandas as pd
from CompleteTransformer import CompleteTransformer

# Load your dataset
df = pd.read_csv("your_dataset.csv")
target_column = "target"

# Initialize and fit CompleteTransformer
transformer = CompleteTransformer(df, target_column)
transformer.fit()

# Transform new data and perform model evaluation
X_new = ...  # New data to predict or transform
train_pred, test_pred, grid_search_results, model = transformer.transform(X_new)

# Display results or further analysis
print("Train predictions:", train_pred)
print("Test predictions:", test_pred)
print("Best parameters found by grid search:", grid_search_results.best_params_)

```

## Dependencies

This project relies on the following Python libraries:

- **scikit-learn**: For machine learning models and tools.
- **pandas**: For data manipulation and handling.
- **numpy**: For numerical operations and arrays.
- **matplotlib**: For visualization of data (optional, depending on your needs).

You can install these dependencies using pip:

```bash
pip install scikit-learn pandas numpy matplotlib
