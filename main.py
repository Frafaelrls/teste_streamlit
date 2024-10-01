import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregar um conjunto de dados de exemplo (pode usar o Iris ou qualquer outro)
@st.cache_data
def load_data():
    # Criando um DataFrame simples
    data = {
        'idade': [22, 25, 47, 35, 46, 23, 24, 42, 39, 30],
        'salario': [1500, 2500, 3500, 3000, 4000, 1600, 1700, 3200, 2900, 2800],
        'comprou': [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
    }
    df = pd.DataFrame(data)
    return df

# Carregar os dados
df = load_data()

# Separar as features e o target
X = df[['idade', 'salario']]
y = df['comprou']

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo XGBoost
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Prever com o modelo
y_pred = model.predict(X_test)

# Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred)

# Criar a interface do Streamlit
st.title('Previsão de Compras com XGBoost')
st.write('Acurácia do modelo:', accuracy)

# Inputs para o usuário
idade = st.number_input('Idade', min_value=18, max_value=100, value=30)
salario = st.number_input('Salário', min_value=1000, max_value=10000, value=2500)

# Previsão com os inputs do usuário
if st.button('Prever'):
    input_data = pd.DataFrame([[idade, salario]], columns=['idade', 'salario'])
    prediction = model.predict(input_data)
    st.write('Previsão: ', 'Comprou' if prediction[0] == 1 else 'Não Comprou')
