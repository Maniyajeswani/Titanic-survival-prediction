# Titanic-survival-prediction
# Titanic Survival Prediction

## Project Overview
This project aims to predict the survival of passengers aboard the Titanic using machine learning techniques. The dataset includes features such as age, sex, passenger class, and fare. The model is trained using a Random Forest Classifier to classify whether a passenger survived or not.

## Dataset
The dataset used is stored in a CSV file named `tested.csv`, which contains passenger details and survival status.

### Features Used:
- `Pclass`: Passenger class (1st, 2nd, 3rd)
- `Sex`: Gender of the passenger
- `Age`: Age of the passenger
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Fare`: Ticket fare
- `Embarked`: Port of embarkation (C, Q, S)

## Preprocessing Steps
1. **Handling Missing Values**: Missing ages were filled with the median, and missing embarkation values were filled with the mode.
2. **Dropping Unnecessary Columns**: Features like `Cabin`, `Ticket`, and `Name` were dropped as they did not contribute significantly to predictions.
3. **Encoding Categorical Variables**: Gender was label-encoded, and embarkation was converted into dummy variables.
4. **Feature Scaling**: `Age` and `Fare` were standardized using `StandardScaler`.
5. **Feature Selection**: Highly correlated features (above 0.95 correlation) were removed to prevent data leakage.

## Model Training
- A **Random Forest Classifier** was used with parameters:
  - `n_estimators=100`
  - `max_depth=3`
  - `min_samples_split=10`
  - `random_state=42`
- The dataset was split into 80% training and 20% testing.

## Evaluation
The model was evaluated using:
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-score)
- **Confusion Matrix**

## Usage
### Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn joblib
```

### Running the Code
1. Place `tested.csv` in the appropriate directory.
2. Run the Python script:
```bash
python Growthlinktask.py
```
3. The model will train and display accuracy and evaluation metrics.
4. The trained model is saved as `titanic_model.pkl` for future predictions.

## Future Improvements
- Experiment with different ML algorithms such as XGBoost.
- Perform hyperparameter tuning to improve accuracy.
- Engineer new features from existing data.

## Author
Developed as part of a Titanic Survival Prediction task using machine learning techniques.

