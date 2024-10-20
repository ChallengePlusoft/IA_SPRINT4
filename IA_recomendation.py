from flask import Flask, request, jsonify
import oracledb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)

# CONFIGURAÇÃO DE CONEXÃO COM O BANCO DE DADOS ORACLE
connection = oracledb.connect(user='rm551763',
                              password='fiap23',
                              dsn='oracle.fiap.com.br:1521/orcl')


# Função para carregar os dados de produtos e características dos usuários
def load_data():
    query_produtos = "SELECT * FROM BT_PRODUTO"
    query_clientes = "SELECT * FROM BT_CLIENTE"
    query_historico = "SELECT * FROM BT_HISTORICO_PESQUISA"

    df_produtos = pd.read_sql(query_produtos, con=connection)
    df_clientes = pd.read_sql(query_clientes, con=connection)
    df_historico = pd.read_sql(query_historico, con=connection)

    return df_produtos, df_clientes, df_historico


# Carregar dados e preprocessar
df_produtos, df_clientes, df_historico = load_data()


def preprocess_data(df_produtos, df_clientes, df_historico):
    # Merge dos dados relevantes
    df = pd.merge(df_historico,
                  df_clientes,
                  left_on='ID_CLIENTE',
                  right_on='ID_CLIENTE')
    df = pd.merge(df, df_produtos, left_on='ID_PRODUTO', right_on='ID_PRODUTO')

    # Selecionar apenas as colunas relevantes para a recomendação
    features = [
        'PELE_CLIENTE', 'ESTADO_CIVIL_CLIENTE', 'CABELO_CLIENTE', 'NM_PRODUTO',
        'ID_PRODUTO'
    ]
    df = df[features]

    # Convertendo colunas categóricas em variáveis dummy
    df = pd.get_dummies(
        df, columns=['PELE_CLIENTE', 'ESTADO_CIVIL_CLIENTE', 'CABELO_CLIENTE'])

    return df


df = preprocess_data(df_produtos, df_clientes, df_historico)

# DIVIDINDO OS DADOS EM TREINO E TESTE
X = df.drop('NM_PRODUTO', axis=1)
y = df['NM_PRODUTO']
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# TREINANDO UM MODELO DE CLASSIFICAÇÃO
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# FUNÇÃO PARA RECOMENDAR PRODUTOS
def recommend_product(client_data):
    client_df = pd.DataFrame([client_data])
    client_df = pd.get_dummies(client_df)

    # Alinhar as colunas do client_df com as colunas usadas no treinamento
    client_df = client_df.reindex(columns=X.columns, fill_value=0)

    recommended_product = model.predict(client_df)
    return recommended_product[0]


# Buscar dados de um cliente específico no banco de dados
def get_client_data(client_id):
    query_cliente = f"SELECT * FROM BT_CLIENTE WHERE ID_CLIENTE = {client_id}"
    df_cliente = pd.read_sql(query_cliente, con=connection)

    if df_cliente.empty:
        raise ValueError("Cliente não encontrado.")

    # Mapeando os dados do cliente para o formato esperado pela função recommend_product
    client_data = {
        'PELE_CLIENTE': df_cliente.iloc[0]['PELE_CLIENTE'],
        'ESTADO_CIVIL_CLIENTE': df_cliente.iloc[0]['ESTADO_CIVIL_CLIENTE'],
        'CABELO_CLIENTE': df_cliente.iloc[0]['CABELO_CLIENTE']
    }

    return client_data


# Buscar dados de um produto específico no banco de dados
def get_product_by_name(product_name):
    query_produto = f"SELECT * FROM BT_PRODUTO WHERE NM_PRODUTO = '{product_name}'"
    df_produto = pd.read_sql(query_produto, con=connection)

    if df_produto.empty:
        raise ValueError("Produto não encontrado.")

    return df_produto.iloc[0].to_dict()


@app.route('/recommend', methods=['GET'])
def recommend():
    client_id = request.args.get('client_id', type=int)

    if client_id is None:
        return jsonify({'error': 'client_id parameter is required'}), 400

    try:
        client_data = get_client_data(client_id)
        recommended_product = recommend_product(client_data)
        product_details = get_product_by_name(recommended_product)
        return jsonify({
            'name': recommended_product,
            'details': product_details
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 404


@app.route('/helloworld', methods=['GET'])
def helloworld():
    return 'Hello World!'


port = int(os.environ.get('PORT', 5000))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)
