import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Charger les données d'iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Conversion en DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[i] for i in y]

# Titre de l'application
st.title("API de Prédiction des Espèces d'Iris")

# Affichage des données
st.write("Données d'Iris :")
st.dataframe(df)

# Séparation des données
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des caractéristiques
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entraînement du modèle
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Affichage de l'exactitude
st.write(f"Exactitude du modèle : {accuracy * 100:.2f}%")

# Interface utilisateur pour la saisie de nouvelles données
st.sidebar.header("Prédictions")
sepal_length = st.sidebar.slider("Longueur du sépale (cm)", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Largeur du sépale (cm)", 2.0, 5.0, 3.0)
petal_length = st.sidebar.slider("Longueur du pétale (cm)", 1.0, 7.0, 1.5)
petal_width = st.sidebar.slider("Largeur du pétale (cm)", 0.1, 2.5, 0.2)

# Préparation des données pour la prédiction
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                           columns=feature_names)

# Normalisation des nouvelles données
input_data_scaled = scaler.transform(input_data)

# Prédiction
predicted_species = model.predict(input_data_scaled)
predicted_species_name = target_names[predicted_species][0]

# Affichage du résultat de la prédiction
st.write(f"Espèce prédite : {predicted_species_name}")
