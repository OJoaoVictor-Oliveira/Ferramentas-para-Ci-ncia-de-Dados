import pandas as pd

path = "./data/"

orders = pd.read_csv(path + "olist_orders_dataset.csv")
items = pd.read_csv(path + "olist_order_items_dataset.csv")
products = pd.read_csv(path + "olist_products_dataset.csv")
reviews = pd.read_csv(path + "olist_order_reviews_dataset.csv")
customers = pd.read_csv(path + "olist_customers_dataset.csv")

df = orders.merge(customers, on='customer_id', how='left') \
        .merge(items, on='order_id', how='inner') \
        .merge(products, on='product_id', how='left') \
        .merge(reviews, on='order_id', how='left')

df.info()
print(df.columns)

# Excluindo colunas que não fazem sentido para a análise
colunas_para_remover = ['order_approved_at', 'order_delivered_carrier_date', 'order_status', 'shipping_limit_date', 'product_name_lenght', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'review_comment_title', 'review_comment_message', 'review_answer_timestamp', 'customer_unique_id', 'customer_zip_code_prefix']
df.drop(columns=colunas_para_remover, inplace=True)

#Colocando valores corretos nos dados
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
df['order_estimated_delivery_date'] = pd.to_datetime(df['order_estimated_delivery_date'])
df['review_creation_date'] = pd.to_datetime(df['review_creation_date'])

df.info()
print(df.columns)

#Ver valores nulos
nulos = df.isnull().sum()
print(nulos)

#Em comparação com o total de linhas os valores nulos são muito poucos, podendo ser removidos
#Excluir valores nulos

df = df.dropna()
df.to_csv("./data/df.csv", index=False)
print("Arquivo criado!")


# Ver duplicados completos (linhas 100% iguais)
duplicados = df.duplicated().sum()
print("Duplicados exatos:", duplicados)

df.info()