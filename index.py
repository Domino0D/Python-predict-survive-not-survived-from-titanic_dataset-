import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing


from sklearn.preprocessing import MaxAbsScaler 

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Titanic-Dataset.csv")
data.info()

preprocessor = ColumnTransformer(
        transformers=[
            ('num', MaxAbsScaler(), ['Age', 'Fare', 'FamilySize', 'IsAlone', 'FareBin', 'AgeBin']),
            ('ticketAndCabin', TfidfVectorizer(max_features=500), 'Text'),
            # ('cabin_tfidf', TfidfVectorizer(max_features=800), 'Cabin')
        ],
        sparse_threshold=0  # Wymuś zwracanie gęstej macierzy
    )

#data cleaning and feature engineering
def preprocess_data(df):
    # Konwersja kolumn na string i wypełnienie braków
    df['Ticket'] = df['Ticket'].fillna('missing_ticket').astype(str)
    df['Cabin'] = df['Cabin'].fillna('missing_cabin').astype(str)
    
    # Usuwanie kolumn (dostosuj według potrzeb)
    df.drop(columns=["Name", "Embarked"], inplace=True, errors='ignore')
    
    fill_missing_ages(df)
    
    # Konwersja płci
    df["Sex"] = df["Sex"].map({'male':1, 'female':0})
    
    # Feature Engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)
    df["AgeBin"] = pd.cut(df["Age"], bins=[0,12,20,40,60, np.inf], labels=False)
    
    # Definicja transformacji z MaxAbsScaler
    df['Text'] = df['Ticket'] + df['Cabin']

    
    df_processed = preprocessor.fit_transform(df)
    
    return df, df_processed

# Fill in missing ages
def fill_missing_ages(df):
    age_fill_map = {}
    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df['Pclass'] == pclass]["Age"].median()
            
    df["Age"] = df.apply(lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"], axis=1)
    # df["Ticket"] = df.apply(lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"], axis=1)
    # df["Cabin"] = df.apply(lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"], axis=1)
    

original_df, X_processed = preprocess_data(data)

# create features /Target Variables

original_df, X = preprocess_data(data.copy())
# X = np.delete(X, original_df.columns.get_loc("Survived"), axis=1)

y = original_df["Survived"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ML Preprocessing
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Hyperparameter Tuning - KNN
def tune_model(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200],      # Zmniejsz z 3 do 2
        "max_depth": [5, 10, 20],     # Zmniejsz z 4 do 3
        "min_samples_split": [2, 5],     # Zmniejsz z 3 do 2
        "min_samples_leaf": [1, 2, 4],      # Zmniejsz z 3 do 2
        "max_features": ["sqrt", "log2"],# Usuń None (zostaw 2)
        "bootstrap": [True]              # Usuń False (zostaw 1)
    }
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

best_model = tune_model(X_train, y_train)

# Predictions and evaulate
def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)
    return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, X_test, y_test)

print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Confusion Matrix:')
print(matrix)



def plot_model(matrix):
    plt.figure(figsize=(10,7))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=["Survived","Not Survived"], yticklabels=["Not Survived", "Survived"])
    plt.title("confusion matrix")
    plt.xlabel("Predicted value")
    plt.ylabel("True values")
    
    matplb= input("czy chcesz zobaczyć wykres?: yes/no")
    if matplb == "yes":
        plt.show()
        predict = input("Czy chcesz zrobić predykcje zgonu? yes/no: ")
        if predict == "no":
            exit()
        else:
            print("Podaj dane na swój temat:")
    else:
        predict = input("Czy chcesz zrobić predykcje zgonu? yes/no: ")
        if predict == "no":
            exit()
        else:
            print("Podaj dane na swój temat:")
        
        
    
plot_model(matrix)

def predict_survival():
    # Zbierz dane od użytkownika
    age = int(input("wiek: "))
    sex = input("płeć (male/female): ").strip().lower()
    sibsp = int(input("Liczba rodzeństwa/małżonków na pokładzie: "))
    parch = int(input("liczba rodziców/dzieci: "))
    fare = float(input("cena biletu: "))
    ticket = input("numer biletu: ").strip()
    cabin = input("kabina: ").strip()
    
    # Przygotuj DataFrame
    input_data = pd.DataFrame({
        'Age': age,
        
        'sex': [1 if sex == 'male' else 0 ],

        'sibsp': sibsp,

        'parch': parch,

        'Fare': fare,

        'ticket': ticket,

        'cabin': cabin

    })

    # df["FamilySize"] = df["SibSp"] + df["Parch"]
    # Oblicz cechy pochodne
    input_data["FamilySize"] = input_data["parch"] + input_data["sibsp"]
    input_data['IsAlone'] = np.where(input_data['FamilySize'] == 0, 1, 0)
    
    # Użyj tych samych przedziałów co w treningu (uwaga: to uproszczenie!)
    input_data['FareBin'] = pd.qcut(input_data['Fare'], 4, labels=False, duplicates='drop')
    input_data['AgeBin'] = pd.cut(input_data['Age'], bins=[0,20,40,60,80, np.inf], labels=False)
    
    # Połącz cechy tekstowe
    input_data['Text'] = input_data['cabin'] = input_data['ticket']

    # Przetwórz dane
 
    input_preprocessed = preprocessor.transform(input_data)
    
    # Prognoza
    prediction = best_model.predict(input_preprocessed)
    print("\nPrzewidywany wynik:", "Przeżył" if prediction[0] == 1 else "Nie przeżył")


# Przykład użycia:
predict_survival()