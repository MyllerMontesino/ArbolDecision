import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Cargar el dataset de razas de gatos
df = pd.read_csv('cat.csv')  # Asegúrate de que este archivo tenga las columnas adecuadas

# Eliminar la columna 'Id' si existe
if 'Id' in df.columns:
    df.drop('Id', axis=1, inplace=True)

# Título de la aplicación
st.title('Clasificación de Razas de Gatos')

# Sidebar para los parámetros de entrada
st.sidebar.header('Parámetros de entrada')

# Parámetros numéricos
age = st.sidebar.slider('Edad (años)', int(df['Age (Years)'].min()), int(df['Age (Years)'].max()))
weight = st.sidebar.slider('Peso (kg)', float(df['Weight (kg)'].min()), float(df['Weight (kg)'].max()))

# Parámetros categóricos
color = st.sidebar.selectbox('Color', df['Color'].unique())
gender = st.sidebar.selectbox('Género', df['Gender'].unique())

# Crear el DataFrame de entrada
input_data = pd.DataFrame({
    'Age (Years)': [age],
    'Weight (kg)': [weight],
    'Color': [color],
    'Gender': [gender]
})

# Convertir variables categóricas a numéricas (si es necesario)
df = pd.get_dummies(df, columns=['Color', 'Gender'], drop_first=True)
input_data = pd.get_dummies(input_data, columns=['Color', 'Gender'], drop_first=True)

# Asegurarse de que las columnas coincidan
input_data = input_data.reindex(columns=df.columns.drop('Breed'), fill_value=0)

# Dividir el conjunto de datos en características (X) y etiqueta (y)
X = df.drop('Breed', axis=1)
y = df['Breed']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar el modelo
modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)

# Realizar la predicción
prediccion = modelo.predict(input_data)
st.write('Resultado de la predicción:', prediccion[0])

# Mostrar imagen según la raza predicha
if prediccion[0] == 'Ragdoll':
    st.image('ragdoll.jpeg') 
elif prediccion[0] == 'Persian':
    st.image('persian.jpg')  
elif prediccion[0] == 'Chartreux':
    st.image('chartreux.jpg')  
