# Python-predict-survive-not-survived-from-titanic_dataset-
This project is a complete machine learning pipeline for predicting Titanic passenger survival based on the classic Titanic dataset.

The code includes:
- Data cleaning and feature engineering
- Text feature processing (`Ticket`, `Cabin`) using TF-IDF
- Numeric feature scaling
- Hyperparameter tuning (GridSearchCV) for the Random Forest model
- Visualization of results (confusion matrix)
- The ability to predict survival based on user-provided passenger data

## Technologies Used

- Python 3
- Pandas, NumPy
- scikit-learn (RandomForest, GridSearchCV, ColumnTransformer, TfidfVectorizer)
- Matplotlib, Seaborn

## How to Run the Project

1. **Clone the repository:**
    ```
    git clone https://github.com/your-username/titanic-ml-pipeline.git
    cd titanic-ml-pipeline
    ```

2. **Install the required libraries:**
    ```
    pip install -r requirements.txt
    ```

3. **Run the main script:**
    ```
    python index.py
    ```

4. **Follow the console instructions to enter your own passenger data and see the survival prediction!**

## Project Structure

- `index.py` – main machine learning pipeline script
- `Titanic-Dataset.csv` – input data
- `requirements.txt` – list of required libraries

## Example Usage

Enter passenger data:
Age: 25
Sex (male/female): female
Number of siblings/spouses aboard: 0
Number of parents/children aboard: 0
Fare: 7.25
Ticket number: A/5 21171
Cabin number: missing_cabin

Predicted result: Survived

## 🛠️ My Custom Changes

This project builds on the original ["Build Your First Machine Learning Model in Python"](https://youtu.be/SW0YGA9d8y8) tutorial by Code with Josh.  
**Here are my main improvements:**

- Combined `Ticket` and `Cabin` into a single text feature and used TF-IDF vectorization.
- Used a `ColumnTransformer` pipeline for joint numeric and text preprocessing.
- Switched from KNN to `RandomForestClassifier` with GridSearchCV for better results.
- Added interactive user input: you can enter your own passenger data and get a survival prediction.
- Improved feature engineering and missing value handling (e.g., robust binning for `Fare` and `Age`).
- Enhanced confusion matrix visualization.
