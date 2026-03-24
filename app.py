import streamlit as st
import pandas as pd
import plotly.express as px

# ===================== CONFIG =====================
st.set_page_config(page_title="Dashboard de Vendas", layout="wide")

# ===================== LOAD =====================
df = pd.read_csv("./data/df.csv")
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

# Bases
df_items = df.copy()
df_orders = df.drop_duplicates(subset='order_id')

# ===================== SIDEBAR =====================
st.sidebar.header("🔎 Filtros")

estados = ["Todos"] + sorted(df['customer_state'].dropna().unique())
estado = st.sidebar.selectbox("Estado", estados)

if estado != "Todos":
    categorias = ["Todos"] + sorted(
        df[df['customer_state'] == estado]['product_category_name'].dropna().unique()
    )
else:
    categorias = ["Todos"] + sorted(df['product_category_name'].dropna().unique())

categoria = st.sidebar.selectbox("Categoria", categorias)

# ===================== FILTROS =====================
df_items_filtrado = df_items.copy()
df_orders_filtrado = df_orders.copy()

# FILTRO DE STATUS (ANTES DE TUDO)
df_orders_filtrado = df_orders_filtrado[df_orders_filtrado['order_status'] == 'delivered']

# ⚠️ garantir que df_items só tenha pedidos entregues
df_items_filtrado = df_items_filtrado[
    df_items_filtrado['order_id'].isin(df_orders_filtrado['order_id'])
]

# 🔹 FILTRO DE ESTADO
if estado != "Todos":
    df_items_filtrado = df_items_filtrado[df_items_filtrado['customer_state'] == estado]
    df_orders_filtrado = df_orders_filtrado[df_orders_filtrado['customer_state'] == estado]

# 🔹 FILTRO DE CATEGORIA
if categoria != "Todos":
    df_items_filtrado = df_items_filtrado[df_items_filtrado['product_category_name'] == categoria]

# ===================== KPIs =====================
total_vendas = df_orders_filtrado['order_id'].nunique()
faturamento = df_items_filtrado['price'].sum()
ticket_medio = faturamento / total_vendas if total_vendas > 0 else 0
nota_media = df_orders_filtrado['review_score'].mean()

# ===================== HEADER =====================
st.title("📊 Dashboard de Vendas")
st.markdown("Visão geral de desempenho")

# ===================== KPIs =====================
col1, col2, col3, col4 = st.columns(4)

col1.metric("🛒 Vendas", total_vendas)
col2.metric("💰 Faturamento", f"R$ {faturamento:,.0f}")
col3.metric("🎯 Ticket Médio", f"R$ {ticket_medio:,.2f}")
col4.metric("⭐ Nota Média", f"{nota_media:.2f}")

st.markdown("---")

# ===================== GRÁFICOS / FOCO =====================
col1, col2 = st.columns(2)

# 🔵 ESTADO
with col1:
    if estado == "Todos":
        vendas_estado = df_orders['customer_state'].value_counts().reset_index()
        vendas_estado.columns = ['Estado', 'Vendas']

        fig_estado = px.bar(
            vendas_estado,
            x='Estado',
            y='Vendas',
            title="Vendas por Estado"
        )

        fig_estado.update_layout(transition_duration=400)

        st.plotly_chart(fig_estado, use_container_width=True)

    else:
        valor_estado = df_orders_filtrado['order_id'].nunique()

        st.markdown(f"""
        <div style='background-color: #d4c7bf; padding:20px; border-radius:12px; text-align:center'>
            <p style='color:black; font-size:18px;'>Estado selecionado</p>
            <p style='font-size:28px; color:white; font-weight:bold;'>{estado}</p>
            <p style='font-size:30px; color:#095169; font-weight:bold;'>{valor_estado}</p>
            <p style='color:black;'>vendas</p>
        </div>
        """, unsafe_allow_html=True)

# 🟣 CATEGORIA
with col2:
    if categoria == "Todos":
        fat_categoria = df_items_filtrado.groupby('product_category_name')['price'].sum().reset_index()
        fat_categoria = fat_categoria.sort_values(by='price', ascending=False)

        fig_categoria = px.bar(
            fat_categoria,
            x='product_category_name',
            y='price',
            title="Faturamento por Categoria"
        )

        fig_categoria.update_layout(transition_duration=400)

        st.plotly_chart(fig_categoria, use_container_width=True)

    else:
        valor_categoria = df_items_filtrado['price'].sum()

        st.markdown(f"""
        <div style='background-color: #d4c7bf; padding:20px; border-radius:12px; text-align:center'>
            <p style='color:black; font-size:18px;'>Categoria selecionada</p>
            <p style='font-size:28px; color:white; font-weight:bold;'>{categoria}</p>
            <p style='font-size:30px; color:#095169; font-weight:bold;'>R$ {valor_categoria:,.0f}</p>
            <p style='color: black;'>faturamento</p>
        </div>
        """, unsafe_allow_html=True)

# ===================== EVOLUÇÃO =====================
st.subheader("📈 Evolução de Faturamento")

df_items_filtrado['mes'] = df_items_filtrado['order_purchase_timestamp'].dt.to_period('M').astype(str)

evolucao = df_items_filtrado.groupby('mes')['price'].sum().reset_index()

fig_tempo = px.line(
    evolucao,
    x='mes',
    y='price',
    markers=True,
    title="Faturamento ao longo do tempo"
)

fig_tempo.update_layout(transition_duration=400)

st.plotly_chart(fig_tempo, use_container_width=True)
