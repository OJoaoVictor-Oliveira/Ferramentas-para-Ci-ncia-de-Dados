import streamlit as st
from src.data_loader import load_data
from src.metrics import calculate_kpis
from src.charts import create_charts

# Load the data
data = load_data()

# Sidebar filters
estado = st.sidebar.selectbox('Escolha o Estado:', data['estado'].unique())
categoria = st.sidebar.selectbox('Escolha a Categoria:', data['categoria'].unique())

# KPI calculations
kpis = calculate_kpis(data, estado, categoria)
st.metric("Total de Pedidos", kpis['pedidos'])
st.metric("Receita Total", kpis['receita'])
st.metric("Ticket Médio", kpis['ticket_medio'])
st.metric("Nota Média", kpis['nota_media'])

# Display charts
create_charts(data, estado, categoria)