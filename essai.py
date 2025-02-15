import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Chargement du fichier CSV

df = pd.read_csv('iris.csv',delimiter=";")

# Définition des colonnes des caractéristiques et de la cible
feature_columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
target_column = 'Species'
        
# Vérification que les colonnes existent dans le dataset
if not all(col in df.columns for col in feature_columns + [target_column]):
            st.error("Les colonnes doivent inclure : SepalLength, SepalWidth, PetalLength, PetalWidth et Species.")
else:
            # Conversion de la colonne cible en entiers si elle est catégorielle
            if df[target_column].dtype == 'object':
                df[target_column] = df[target_column].astype('category')
                class_mapping = dict(enumerate(df[target_column].cat.categories))
                df[target_column] = df[target_column].cat.codes  # Conversion en entiers
            
            # Séparer les caractéristiques et la cible
            X = df[feature_columns]
            y = df[target_column]
            
            # Paramètres du modèle
            test_size = st.slider("Taille du jeu de test (%)", 10, 50, 20, step=5) / 100
            n_neighbors = st.slider("Nombre de voisins (k)", 1, 15, 3)
            
            # Division des données
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Normalisation des données
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Création et entraînement du modèle
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(X_train, y_train)
            
            # Prédictions
            y_pred = knn.predict(X_test)
            
            # Affichage des résultats
            st.subheader("Résultats du modèle KNN")
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"**Exactitude du modèle :** {accuracy * 100:.2f}%")
            
            # Prédiction sur une nouvelle entrée utilisateur
            st.subheader("Prédire une espèce d'Iris")
            sepal_length = st.slider("Longueur du sépale", 0.0, 8.0, 5.5, 0.1)
            sepal_width = st.slider("Largeur du sépale", 0.0, 8.0, 3.0, 0.1)
            petal_length = st.slider("Longueur du pétale", 0.0, 8.0, 4.0, 0.1)
            petal_width = st.slider("Largeur du pétale", 0.0, 8.0, 1.2, 0.1)
            
            if st.button("Prédire l'espèce"):
                input_data = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
                prediction = knn.predict(input_data)[0]
                predicted_species = class_mapping[prediction]
                st.write(f"L'espèce prédite est : **{predicted_species}**")
