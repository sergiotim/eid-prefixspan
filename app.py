import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

# ==========================================
# 1 e 2. CONFIGURAÇÃO E CONEXÃO EM NUVEM (NEON)
# ==========================================
st.set_page_config(page_title="StreamTudo - OLAP & Pattern Discovery", layout="wide")

@st.cache_data
def carregar_dados_olap():
    # 1. Puxando a URL do banco de forma segura através dos Secrets do Streamlit
    url_neon = st.secrets["DATABASE_URL"]
    
    # 2. Conectando usando SQLAlchemy
    engine = create_engine(url_neon)
    
    # 3. Extração otimizada com JOIN e aspas duplas nas tabelas (Regra do Postgres)
    query_eventos = """
        SELECT 
            u.userid, 
            s.sessaoid, 
            e.timestamp, 
            e.event
        FROM "EVENTOS" e
        JOIN "SESSOES" s ON e.sessaoid = s.sessaoid
        JOIN "USUARIOS" u ON s.userid = u.userid
    """
    
    # Executa a leitura da tabela de eventos
    df = pd.read_sql_query(query_eventos, engine)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    df['data'] = df['timestamp'].dt.date
    
    # Executa a leitura dos padrões minerados pela IA (também com aspas duplas)
    df_padroes = pd.read_sql_query('SELECT * FROM "PADROES_CHURN"', engine)
    
    return df, df_padroes

df_olap, df_padroes_completo = carregar_dados_olap()

# ==========================================
# 3. FUNÇÃO DO GRAFO (NETWORKX + MATPLOTLIB)
# ==========================================
def plotar_grafo_rede_matplotlib(df_dados):
    G = nx.DiGraph()
    for index, row in df_dados.iterrows():
        jornada = row['Jornada Descoberta'].split(' ➔ ')
        ocorrencias = row['Ocorrencias']
        
        for i in range(len(jornada) - 1):
            origem = jornada[i]
            destino = jornada[i+1]
            if G.has_edge(origem, destino):
                G[origem][destino]['weight'] += ocorrencias
            else:
                G.add_edge(origem, destino, weight=ocorrencias)
                
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Mapeamento em Rede da Jornada de Churn", fontsize=14, fontweight='bold')
    pos = nx.spring_layout(G, k=0.8, seed=42)
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3000, node_color='#E50914', edgecolors='black', alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold', font_color='white')
    
    pesos = [G[u][v]['weight'] for u, v in G.edges()]
    max_peso = max(pesos) if pesos else 1
    espessuras = [(p / max_peso) * 5 for p in pesos]
    
    nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='-|>', arrowsize=15, width=espessuras, edge_color='gray')
    edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7, label_pos=0.3)
    
    ax.axis('off')
    plt.tight_layout()
    return fig

# ==========================================
# 4. INTERFACE DO DASHBOARD E FILTROS OLAP
# ==========================================
st.title("🎬 StreamTudo: Painel Executivo 360º")
st.markdown("Monitoramento de Eventos e Descoberta de Padrões de Evasão (Churn).")
st.markdown("---")

st.subheader("⚙️ Filtros OLAP (Slice)")
st.write("Selecione os Eventos de Navegação para analisar:")

acoes_disponiveis = df_olap['event'].unique().tolist()
filtro_acao = []
cols = st.columns(len(acoes_disponiveis))

for i, acao in enumerate(acoes_disponiveis):
    if cols[i].checkbox(acao, value=True):
        filtro_acao.append(acao)

if len(filtro_acao) > 0:
    df_filtrado = df_olap[df_olap['event'].isin(filtro_acao)]
else:
    st.warning("⚠️ Selecione pelo menos um evento acima.")
    df_filtrado = df_olap.head(0)

st.markdown("<br>", unsafe_allow_html=True)

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Usuários Únicos", f"{df_filtrado['userid'].nunique():,}".replace(',', '.'))
col2.metric("📱 Sessões Geradas", f"{df_filtrado['sessaoid'].nunique():,}".replace(',', '.'))
col3.metric("🖱️ Total de Interações", f"{len(df_filtrado):,}".replace(',', '.'))
cancelamentos = len(df_filtrado[df_filtrado['event'] == 'cancelar_assinatura'])
col4.metric("🚨 Total de Cancelamentos", f"{cancelamentos:,}".replace(',', '.'))
st.markdown("---")

# ==========================================
# 5. GRÁFICOS OLAP
# ==========================================
st.subheader("📊 Análise Descritiva OLAP")
col_graf1, col_graf2 = st.columns(2)

with col_graf1:
    st.write("**Top Eventos mais realizados (Frequência)**")
    graf_barras = df_filtrado['event'].value_counts().reset_index()
    graf_barras.columns = ['Evento', 'Volume']
    st.bar_chart(graf_barras.set_index('Evento'), x_label="Ação Realizada", y_label="Volume")

with col_graf2:
    st.write("**Evolução Temporal de Interações (Série Histórica)**")
    # OPERAÇÃO OLAP: Roll-up de Tempo
    agrupamento = st.radio("Agrupar tempo por (Roll-up):", ["Diário", "Semanal", "Mensal"], horizontal=True)
    df_tempo = df_filtrado.copy()
    df_tempo.set_index('timestamp', inplace=True)
    
    if agrupamento == "Diário":
        graf_linha = df_tempo.resample('D').size()
    elif agrupamento == "Semanal":
        graf_linha = df_tempo.resample('W').size()
    else:
        graf_linha = df_tempo.resample('ME').size()
        
    st.line_chart(graf_linha, x_label=f"Linha do Tempo ({agrupamento})", y_label="Volume")

st.markdown("---")

# ==========================================
# 6. PREFIXSPAN DINÂMICO E GRAFO DE REDE
# ==========================================
st.header("🧠 Inteligência Artificial: Padrões Sequenciais (PrefixSpan)")

opcao_ps = st.radio(
    "Ajuste da Lente do Algoritmo (Hiperparâmetros):",
    ["🔎 Visão de Gatilho (Suporte: 30% | Minlen: 2)", 
     "🧭 Visão Média (Suporte: 20% | Minlen: 3)", 
     "🗺️ Visão Raiz/Profunda (Suporte: 10% | Minlen: 4)"],
    horizontal=True
)

if "30%" in opcao_ps:
    dados_ps = df_padroes_completo[df_padroes_completo['Visao'] == '30%'].drop(columns=['Visao'])
    st.error("🚨 **Ação Imediata (ROI):** Redirecionar a equipe de TI para auditar a funcionalidade 'Adicionar à Lista'. Corrigir este gargalo técnico estancará a perda de milhares de assinaturas.")
elif "20%" in opcao_ps:
    dados_ps = df_padroes_completo[df_padroes_completo['Visao'] == '20%'].drop(columns=['Visao'])
    st.warning("⚠️ **Alerta:** O erro acontece no meio da jornada logo após a exploração inicial.")
else:
    dados_ps = df_padroes_completo[df_padroes_completo['Visao'] == '10%'].drop(columns=['Visao'])
    st.success("✅ **Insight Completo:** Identificamos exatamente o caminho de ponta a ponta desde a home até o cancelamento.")

if not dados_ps.empty:
    st.write("**Tabela de Ocorrências Absolutas**")
    st.table(dados_ps.assign(Index=range(1, len(dados_ps)+1)).set_index('Index'))
    
    st.markdown("---")
    st.write("**Visualização do Volume de Churn por Jornada**")
    
    # Voltando ao gráfico de barras nativo do Streamlit, com a cor da marca (Vermelho)
    st.bar_chart(
        data=dados_ps, 
        x="Jornada Descoberta", 
        y="Ocorrencias", 
        color="#E50914"
    )
    
else:
    st.write("Nenhum padrão encontrado para esta configuração no banco de dados.")