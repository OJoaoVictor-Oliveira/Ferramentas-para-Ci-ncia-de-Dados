import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import nltk
from nltk.corpus import stopwords
import re
import warnings
warnings.filterwarnings('ignore')
 
# ── Configuração da página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="SIPA - Sistema de Inteligência de Preços e Análise",
    page_icon="📊",
    layout="wide"
)
 
# ── Download NLTK ───────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
download_nltk()
 
# ── Carregamento dos dados ──────────────────────────────────────────────────
@st.cache_data
def carregar_dados():
    orders      = pd.read_csv('olist_data/olist_orders_dataset.csv')
    items       = pd.read_csv('olist_data/olist_order_items_dataset.csv')
    products    = pd.read_csv('olist_data/olist_products_dataset.csv')
    reviews     = pd.read_csv('olist_data/olist_order_reviews_dataset.csv')
    customers   = pd.read_csv('olist_data/olist_customers_dataset.csv')
    translation = pd.read_csv('olist_data/product_category_name_translation.csv')
 
    # Unificação
    df = (orders
          .merge(items,       on='order_id',   how='inner')
          .merge(products,    on='product_id', how='left')
          .merge(reviews,     on='order_id',   how='left')
          .merge(customers,   on='customer_id',how='left')
          .merge(translation, on='product_category_name', how='left'))
 
    # Datas
    date_cols = ['order_purchase_timestamp','order_approved_at',
                 'order_delivered_customer_date','order_estimated_delivery_date']
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors='coerce')
 
    # Limpeza
    df.drop_duplicates(inplace=True)
    df['review_comment_message'].fillna('', inplace=True)
    df['product_category_name_english'].fillna('outros', inplace=True)
    df['price'].fillna(df['price'].median(), inplace=True)
    df['customer_state'].fillna('Desconhecido', inplace=True)
 
    # Colunas auxiliares
    df['mes_ano']   = df['order_purchase_timestamp'].dt.to_period('M').astype(str)
    df['estado']    = df['customer_state']
    df['categoria'] = df['product_category_name_english']
 
    # Tempo de entrega em dias
    df['dias_entrega'] = (
        df['order_delivered_customer_date'] - df['order_purchase_timestamp']
    ).dt.days
 
    return df
 
# ── Carrega ─────────────────────────────────────────────────────────────────
with st.spinner('Carregando e processando os dados...'):
    df = carregar_dados()
 
# ── Sidebar / Filtros ────────────────────────────────────────────────────────
st.sidebar.title("🎛️ Filtros")
estados    = ['Todos'] + sorted(df['estado'].dropna().unique().tolist())
categorias = ['Todas'] + sorted(df['categoria'].dropna().unique().tolist())
 
estado_sel    = st.sidebar.selectbox("Estado", estados)
categoria_sel = st.sidebar.selectbox("Categoria", categorias)
 
df_filtrado = df.copy()
if estado_sel != 'Todos':
    df_filtrado = df_filtrado[df_filtrado['estado'] == estado_sel]
if categoria_sel != 'Todas':
    df_filtrado = df_filtrado[df_filtrado['categoria'] == categoria_sel]
 
# ══════════════════════════════════════════════════════════════════════════════
# TÍTULO
# ══════════════════════════════════════════════════════════════════════════════
st.title("📊 SIPA — Sistema de Inteligência de Preços e Análise de Sentimento")
st.markdown("**Dataset:** Olist Brazilian E-Commerce  |  **Fonte:** Kaggle")
st.divider()
 
# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 — PAINEL DE CONTROLE
# ══════════════════════════════════════════════════════════════════════════════
st.header("📦 Fase 1 — Painel de Controle")
 
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total de Pedidos",       f"{df_filtrado['order_id'].nunique():,}")
col2.metric("Receita Total",          f"R$ {df_filtrado['price'].sum():,.2f}")
col3.metric("Ticket Médio",           f"R$ {df_filtrado['price'].mean():,.2f}")
col4.metric("Avaliação Média",        f"{df_filtrado['review_score'].mean():.2f} ⭐")
col5.metric("Entrega Média (dias)",   f"{df_filtrado['dias_entrega'].mean():.1f} dias")
 
# Receita Mensal
st.subheader("Vendas por Mês")
vendas_mes = (df_filtrado.groupby('mes_ano')['price']
              .sum().reset_index().sort_values('mes_ano'))
fig1 = px.line(vendas_mes, x='mes_ano', y='price',
               labels={'mes_ano':'Mês','price':'Receita (R$)'},
               title='Receita Mensal')
fig1.update_traces(line_color='#3498db', line_width=2.5)
st.plotly_chart(fig1, use_container_width=True)
 
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Top 10 Categorias por Receita")
    top_cat = (df_filtrado.groupby('categoria')['price']
               .sum().nlargest(10).reset_index())
    fig2 = px.bar(top_cat, x='price', y='categoria', orientation='h',
                  color='price', color_continuous_scale='Blues',
                  labels={'price':'Receita (R$)','categoria':'Categoria'})
    st.plotly_chart(fig2, use_container_width=True)
 
with col_b:
    st.subheader("Pedidos por Estado")
    por_estado = (df_filtrado.groupby('estado')['order_id']
                  .nunique().reset_index()
                  .rename(columns={'order_id':'pedidos'})
                  .sort_values('pedidos', ascending=False))
    fig3 = px.bar(por_estado, x='estado', y='pedidos',
                  color='pedidos', color_continuous_scale='Teal',
                  labels={'estado':'Estado','pedidos':'Pedidos'})
    st.plotly_chart(fig3, use_container_width=True)
 
# Ticket médio por categoria
st.subheader("Ticket Médio por Categoria (Top 10)")
ticket_cat = (df_filtrado.groupby('categoria')['price']
              .mean().nlargest(10).reset_index()
              .rename(columns={'price':'ticket_medio'}))
fig_ticket = px.bar(ticket_cat, x='ticket_medio', y='categoria',
                    orientation='h', color='ticket_medio',
                    color_continuous_scale='Oranges',
                    labels={'ticket_medio':'Ticket Médio (R$)','categoria':'Categoria'})
st.plotly_chart(fig_ticket, use_container_width=True)
 
st.divider()
 
# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 — ENGENHARIA DE ATRIBUTOS + NLP
# ══════════════════════════════════════════════════════════════════════════════
st.header("⚙️ Fase 2 — Engenharia de Atributos e NLP")
 
# ── Pipeline OneHotEncoder + Scaler ─────────────────────────────────────────
st.subheader("🔧 Pipeline de Pré-Processamento (OneHotEncoder + Scaler)")
 
with st.expander("ℹ️ O que é este pipeline?", expanded=False):
    st.markdown("""
    O pipeline de engenharia de atributos transforma os dados brutos em variáveis 
    que os algoritmos de Machine Learning conseguem interpretar:
    - **OneHotEncoder**: converte categorias (ex: estado, categoria do produto) em colunas binárias (0 ou 1)
    - **StandardScaler**: normaliza variáveis numéricas (ex: preço, frete) para ficarem na mesma escala
    - **Objetivo**: preparar os dados para modelos preditivos de forma padronizada
    """)
 
@st.cache_data
def rodar_pipeline(df_base):
    sample = df_base[['categoria','estado','price','freight_value',
                       'review_score']].dropna().sample(
                           min(5000, len(df_base)), random_state=42)
 
    # Variáveis categóricas
    cat_features = ['categoria', 'estado']
    num_features = ['price', 'freight_value']
 
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    scaler = StandardScaler()
 
    encoded = ohe.fit_transform(sample[cat_features])
    scaled  = scaler.fit_transform(sample[num_features])
 
    encoded_cols = ohe.get_feature_names_out(cat_features)
    df_encoded   = pd.DataFrame(encoded, columns=encoded_cols)
    df_scaled    = pd.DataFrame(scaled,  columns=[f'{c}_scaled' for c in num_features])
 
    # Stats do scaler
    scaler_stats = pd.DataFrame({
        'variável': num_features,
        'média original': scaler.mean_,
        'desvio padrão':  np.sqrt(scaler.var_)
    })
 
    # Top categorias geradas pelo OHE
    top_ohe = pd.DataFrame({
        'coluna gerada': encoded_cols,
        'soma': df_encoded.sum().values
    }).sort_values('soma', ascending=False).head(15)
 
    return scaler_stats, top_ohe, sample
 
scaler_stats, top_ohe, sample_pipe = rodar_pipeline(df_filtrado)
 
col_p1, col_p2 = st.columns(2)
with col_p1:
    st.markdown("**StandardScaler — Estatísticas das Variáveis Numéricas**")
    st.dataframe(scaler_stats.style.format({
        'média original': 'R$ {:.2f}',
        'desvio padrão':  'R$ {:.2f}'
    }), use_container_width=True)
 
with col_p2:
    st.markdown("**OneHotEncoder — Top 15 Colunas Geradas**")
    fig_ohe = px.bar(top_ohe, x='soma', y='coluna gerada', orientation='h',
                     color='soma', color_continuous_scale='Purples',
                     labels={'soma':'Frequência','coluna gerada':'Atributo'})
    fig_ohe.update_layout(height=400)
    st.plotly_chart(fig_ohe, use_container_width=True)
 
st.markdown("**Amostra dos dados após transformação (primeiras 5 linhas):**")
amostra_display = sample_pipe[['categoria','estado','price','freight_value','review_score']].head()
st.dataframe(amostra_display, use_container_width=True)
 
st.divider()
 
# ── NLP / TF-IDF ────────────────────────────────────────────────────────────
st.subheader("💬 Monitor de Qualidade (TF-IDF nos Reviews)")
 
@st.cache_data
def processar_nlp(df_reviews):
    stop_pt = set(stopwords.words('portuguese'))
    stop_extra = {'produto','entrega','chegou','recebi','comprei',
                  'pedido','loja','muito','bem','bom','boa','mais',
                  'para','que','com','uma','por','foi','mas','nao',
                  'sim','ate','ser','ter','vai','vou','nem'}
    stop_all = stop_pt | stop_extra
 
    def limpar(txt):
        txt = str(txt).lower()
        txt = re.sub(r'[^a-záàãâéêíóôõúç\s]', '', txt)
        tokens = txt.split()
        return ' '.join([t for t in tokens if t not in stop_all and len(t) > 2])
 
    positivos = df_reviews[df_reviews['review_score'] >= 4]['review_comment_message']
    negativos = df_reviews[df_reviews['review_score'] <= 2]['review_comment_message']
 
    def top_termos(series, n=15):
        corpus = series.apply(limpar)
        corpus = corpus[corpus.str.strip() != '']
        if len(corpus) < 5:
            return pd.DataFrame(columns=['termo','score'])
        tfidf = TfidfVectorizer(max_features=200, min_df=2)
        tfidf.fit(corpus)
        scores = tfidf.idf_
        termos = tfidf.get_feature_names_out()
        return (pd.DataFrame({'termo': termos, 'score': scores})
                .sort_values('score').head(n))
 
    return top_termos(positivos), top_termos(negativos)
 
pos_df, neg_df = processar_nlp(df_filtrado)
 
col_nlp1, col_nlp2 = st.columns(2)
with col_nlp1:
    st.markdown("**✅ Termos mais relevantes — Avaliações Positivas (4-5 ⭐)**")
    if not pos_df.empty:
        fig_pos = px.bar(pos_df, x='score', y='termo', orientation='h',
                         color_discrete_sequence=['#2ecc71'],
                         labels={'score':'Score TF-IDF','termo':'Termo'})
        st.plotly_chart(fig_pos, use_container_width=True)
    else:
        st.info("Dados insuficientes.")
 
with col_nlp2:
    st.markdown("**❌ Termos mais relevantes — Avaliações Negativas (1-2 ⭐)**")
    if not neg_df.empty:
        fig_neg = px.bar(neg_df, x='score', y='termo', orientation='h',
                         color_discrete_sequence=['#e74c3c'],
                         labels={'score':'Score TF-IDF','termo':'Termo'})
        st.plotly_chart(fig_neg, use_container_width=True)
    else:
        st.info("Dados insuficientes.")
 
# Distribuição de notas
st.subheader("Distribuição das Avaliações")
dist_score = (df_filtrado['review_score'].value_counts()
              .sort_index().reset_index())
dist_score.columns = ['nota','quantidade']
fig_score = px.bar(dist_score, x='nota', y='quantidade',
                   color='nota', color_continuous_scale='RdYlGn',
                   labels={'nota':'Nota','quantidade':'Quantidade'})
st.plotly_chart(fig_score, use_container_width=True)
 
st.divider()
 
# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 — SÉRIES TEMPORAIS
# ══════════════════════════════════════════════════════════════════════════════
st.header("🔮 Fase 3 — Máquina do Tempo (Séries Temporais)")
 
@st.cache_data
def series_temporais(df_base):
    serie = (df_base.groupby('mes_ano')['order_id']
             .nunique().reset_index()
             .rename(columns={'order_id':'pedidos'})
             .sort_values('mes_ano'))
    serie_ts = serie['pedidos'].astype(float).values
    return serie, serie_ts
 
serie, serie_ts_arr = series_temporais(df_filtrado)
 
# Decomposição
st.subheader("Decomposição da Série Temporal")
if len(serie_ts_arr) >= 12:
    try:
        import pandas as pd
        serie_ts = pd.Series(serie_ts_arr)
        decomp   = seasonal_decompose(serie_ts, model='additive', period=12)
 
        fig_decomp = go.Figure()
        fig_decomp.add_trace(go.Scatter(y=decomp.trend,    name='Tendência',
                                        line=dict(color='#3498db', width=2)))
        fig_decomp.add_trace(go.Scatter(y=decomp.seasonal, name='Sazonalidade',
                                        line=dict(color='#f39c12', width=2)))
        fig_decomp.add_trace(go.Scatter(y=decomp.resid,    name='Resíduo',
                                        line=dict(color='#e74c3c', width=1.5)))
        fig_decomp.update_layout(
            title='Decomposição da Série (Tendência, Sazonalidade, Resíduo)',
            xaxis_title='Período', yaxis_title='Pedidos')
        st.plotly_chart(fig_decomp, use_container_width=True)
    except Exception as e:
        st.warning(f"Decomposição indisponível: {e}")
else:
    st.info("Mínimo de 12 meses necessário. Selecione 'Todos' nos filtros.")
 
# ── Validação Train/Test Split ───────────────────────────────────────────────
st.subheader("📊 Validação do Modelo — Train/Test Split")
 
with st.expander("ℹ️ O que é Train/Test Split?", expanded=False):
    st.markdown("""
    Para validar se o modelo realmente aprende os padrões dos dados, dividimos a série em:
    - **Treino (80%)**: dados que o modelo usa para aprender
    - **Teste (20%)**: dados que o modelo nunca viu — usados para medir o erro real
    
    As métricas **MAE** e **RMSE** calculadas sobre o conjunto de teste são mais confiáveis
    do que as calculadas sobre os dados de treino.
    """)
 
if len(serie_ts_arr) >= 8:
    try:
        n        = len(serie_ts_arr)
        split    = int(n * 0.8)
        treino   = pd.Series(serie_ts_arr[:split])
        teste    = pd.Series(serie_ts_arr[split:])
 
        modelo_val = ExponentialSmoothing(
            treino, trend='add',
            seasonal='add'    if len(treino) >= 12 else None,
            seasonal_periods=12 if len(treino) >= 12 else None)
        fit_val    = modelo_val.fit(optimized=True)
        prev_teste = fit_val.forecast(len(teste))
 
        mae_test  = mean_absolute_error(teste, prev_teste)
        rmse_test = np.sqrt(mean_squared_error(teste, prev_teste))
        mape_test = np.mean(np.abs((teste.values - prev_teste.values) /
                                    np.where(teste.values == 0, 1, teste.values))) * 100
 
        col_v1, col_v2, col_v3 = st.columns(3)
        col_v1.metric("MAE  — Teste",  f"{mae_test:.1f} pedidos",
                      help="Erro médio absoluto no conjunto de teste")
        col_v2.metric("RMSE — Teste",  f"{rmse_test:.1f} pedidos",
                      help="Raiz do erro quadrático médio no conjunto de teste")
        col_v3.metric("MAPE — Teste",  f"{mape_test:.1f}%",
                      help="Erro percentual médio absoluto")
 
        # Gráfico train/test
        meses = serie['mes_ano'].tolist()
        fig_val = go.Figure()
        fig_val.add_trace(go.Scatter(
            x=meses[:split], y=treino.values,
            name='Treino (80%)', line=dict(color='#3498db', width=2)))
        fig_val.add_trace(go.Scatter(
            x=meses[split:], y=teste.values,
            name='Teste Real (20%)', line=dict(color='#2ecc71', width=2)))
        fig_val.add_trace(go.Scatter(
            x=meses[split:], y=prev_teste.values,
            name='Previsão no Teste', line=dict(color='#e74c3c', width=2, dash='dash')))
        fig_val.update_layout(
            title='Validação: Treino vs Teste vs Previsão',
            xaxis_title='Mês', yaxis_title='Nº de Pedidos')
        st.plotly_chart(fig_val, use_container_width=True)
 
    except Exception as e:
        st.warning(f"Validação indisponível: {e}")
else:
    st.info("Dados insuficientes para validação. Selecione 'Todos' nos filtros.")
 
# ── Previsão Futura ──────────────────────────────────────────────────────────
st.subheader("🚀 Previsão de Demanda Futura")
meses_prev = st.slider("Quantos meses prever?", 1, 6, 3)
 
if len(serie_ts_arr) >= 6:
    try:
        serie_ts_full = pd.Series(serie_ts_arr)
        modelo_final  = ExponentialSmoothing(
            serie_ts_full, trend='add',
            seasonal='add'    if len(serie_ts_full) >= 12 else None,
            seasonal_periods=12 if len(serie_ts_full) >= 12 else None)
        fit_final  = modelo_final.fit(optimized=True)
        previsao   = fit_final.forecast(meses_prev)
 
        meses_futuros = [f"Prev +{i+1}m" for i in range(meses_prev)]
 
        fig_prev = go.Figure()
        fig_prev.add_trace(go.Scatter(
            x=serie['mes_ano'], y=serie_ts_arr,
            name='Histórico', line=dict(color='#3498db', width=2)))
        fig_prev.add_trace(go.Scatter(
            x=meses_futuros, y=previsao.values,
            name='Previsão', line=dict(color='#f39c12', width=2.5, dash='dash'),
            mode='lines+markers', marker=dict(size=8)))
        fig_prev.update_layout(
            title='Histórico de Pedidos + Previsão Futura',
            xaxis_title='Mês', yaxis_title='Nº de Pedidos')
        st.plotly_chart(fig_prev, use_container_width=True)
 
        # Tabela de previsão
        st.markdown("**Valores previstos:**")
        df_prev = pd.DataFrame({
            'Mês': meses_futuros,
            'Pedidos Previstos': previsao.values.astype(int)
        })
        st.dataframe(df_prev, use_container_width=True)
 
    except Exception as e:
        st.warning(f"Previsão indisponível: {e}")
else:
    st.info("Dados insuficientes para previsão.")
 
st.divider()
st.caption("SIPA — Desenvolvido para a disciplina Ferramentas para Ciência de Dados | EAD Unifor")
 