import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv('iris.csv')
df.drop('Id', axis=1, inplace=True)
st.title('Iris Predicción')
st.sidebar.header('Parametros de entrada')
sl = st.sidebar.slider('Sepal Length',df.SepalLengthCm.min(), df.SepalLengthCm.max())
sw = st.sidebar.slider('Sepal Width',df.SepalWidthCm.min(), df.SepalWidthCm.max())
pl = st.sidebar.slider('Petal Length',df.PetalLengthCm.min(), df.PetalLengthCm.max())
pw = st.sidebar.slider('Petal Width',df.PetalWidthCm.min(), df.PetalWidthCm.max())
btn_predecir = st.sidebar.button('Predecir')

st.write('Resultado de la predicción')
input = pd.DataFrame({  'SepalLengthCm': [sl],
                        'SepalWidthCm': [sw],
                        'PetalLengthCm': [pl],
                        'PetalWidthCm': [pw]})
X = df.drop('Species', axis=1)
y = df['Species']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3)
modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)

prediccion = modelo.predict(input)
st.write(prediccion[0])
if prediccion[0] == 'Iris-setosa':
    st.image('https://www.plant-world-seeds.com/images/item_images/000/007/023/large_square/iris_baby_blue.jpg?1500653527')
if prediccion[0] == 'Iris-virginica':
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS8D2E4Mkwy5IDwz8_mNY9nK84q14aalePSNMS_1TsOCbvyt5BpFXIkYZNtpLLLds795jNT')
if prediccion[0] == 'Iris-versicolor':
    st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ0UxLmRN_W0DgwDei4PjQRV9sKfGa-2kIncw&s')

