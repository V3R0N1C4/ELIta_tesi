import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from Fase1.emotion_config import BASIC_EMOTIONS, EMOTION_COLORS

def plot_semantic_structure(df_recalc, df_centroids, target_emotion, top_n=10):
    """
    Genera i due grafici (Network e Scatter) basati sui punteggi ricalcolati.
    """
    # 1. Selezioniamo le parole più rappresentative per l'emozione target
    # (Prendiamo le top N parole basandoci sul punteggio ricalcolato)
    top_words = df_recalc[target_emotion].sort_values(ascending=False).head(top_n * 3).index

    # Prepariamo i dati per il t-SNE: Vettori delle parole + Vettore del Centroide
    vectors = df_recalc.loc[top_words].values
    centroid_vector = df_centroids.loc[target_emotion].values.reshape(1, -1)

    combined_vectors = np.vstack([centroid_vector, vectors])
    labels = [f"CENTROID_{target_emotion.upper()}"] + list(top_words)
    types = ["Reference"] + ["Word"] * len(top_words)

    # 2. Riduzione dimensionale t-SNE
    # Usiamo una perplexity bassa perché abbiamo pochi punti
    tsne = TSNE(n_components=2, perplexity=min(10, len(labels) - 1), random_state=42, init='pca')
    pos_2d = tsne.fit_transform(combined_vectors)

    # --- GRAFICO 1: MAPPA SEMANTICA (NETWORK) ---
    G = nx.Graph()
    for i, label in enumerate(labels):
        G.add_node(label, pos=pos_2d[i], type=types[i])

    # Aggiungiamo archi se la similarità coseno è > 0.95 (regola molto stretta)
    # --- NUOVA LOGICA PER GLI ARCHI ---
    sim_matrix = cosine_similarity(combined_vectors)    # Calcola tutte le similarità coseno tra i nodi.
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            sim = sim_matrix[i, j]

            # Regola 1: Se uno dei due nodi è il CENTROIDE, usiamo una soglia più bassa per mostrare il legame con le sue parole
            if labels[i].startswith("CENTROID") or labels[j].startswith("CENTROID"):
                if sim > 0.95:  # Soglia più permissiva per il centroide
                    G.add_edge(labels[i], labels[j], weight=sim)

            # Regola 2: Per le connessioni tra parole blu, teniamo la soglia alta
            else:
                if sim > 0.98:
                    G.add_edge(labels[i], labels[j], weight=sim)

    plt.figure(figsize=(12, 8))
    pos = nx.get_node_attributes(G, 'pos')
    colors = ['red' if G.nodes[n]['type'] == "Reference" else '#1f77b4' for n in G.nodes]

    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=800,
            edge_color="#dddddd", font_size=9, font_weight='bold')
    plt.title(f"Mappa Semantica: Struttura di '{target_emotion.capitalize()}' (Ricalcolata)")
    plt.show()

    # --- GRAFICO 2: SCATTER PLOT (t-SNE) con sfondo BIANCO ---
    plt.figure(figsize=(10, 8))
    ax2 = plt.gca()
    ax2.set_facecolor('white')
    plt.gcf().patch.set_facecolor('white')

    # Plot dei punti
    for i, label in enumerate(labels):
        color = 'red' if types[i] == "Reference" else '#1f77b4'
        plt.scatter(pos_2d[i, 0], pos_2d[i, 1], c=color, s=150, edgecolors='white')
        plt.text(pos_2d[i, 0] + 0.1, pos_2d[i, 1] + 0.1, label, fontsize=10, color='black')  # <-- TESTO NERO

    plt.title(f"Scatter Plot t-SNE: Vicinanza al Centroide '{target_emotion.capitalize()}'", color='black')
    plt.axis('off')
    plt.show()


def compare_pca_layouts(pca_list, titles):
    """
    Genera un grafico a subplot per confrontare i layout PCA dei diversi dataset (Originale, Ricalcolato).
     - pca_list: Lista di dizionari con chiave 'df' che contiene il DataFrame con le colonne 'PC1', 'PC2' e 'Emozione Dominante'.
     - titles: Lista di titoli per ogni subplot (es. ["Originale", "Ricalcolato", "Ibrido"]).
    """
    fig = make_subplots(rows=1, cols=len(pca_list), subplot_titles=titles, horizontal_spacing=0.05)

    for i, pca_data in enumerate(pca_list):
        df = pca_data['df']
        for emo in BASIC_EMOTIONS:
            emo_data = df[df['Emozione Dominante'] == emo]
            fig.add_trace(
                go.Scatter(
                    x=emo_data['PC1'], y=emo_data['PC2'],
                    mode='markers',
                    marker=dict(color=EMOTION_COLORS[emo], size=5, opacity=0.6),
                    name=emo,
                    showlegend=(i == 0),  # Mostra legenda solo per il primo grafico
                    legendgroup=emo
                ),
                row=1, col=i + 1
            )

    fig.update_layout(title_text="Confronto della compattezza degli spazi emotivi (Originale -> Ibrido)")
    fig.show()


def save_alpha_csvs(alpha_dataframes, output_dir=".", base_name="ELIta_ricalcolata_alpha", encoding="utf-8-sig"):
    """
    Salva un CSV per ogni coppia {alpha: dataframe}.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for alpha, df in alpha_dataframes.items():
        alpha_tag = str(alpha).replace(".", "_")
        csv_path = out / f"{base_name}_{alpha_tag}.csv"
        df.to_csv(csv_path, encoding=encoding)
        saved_files.append(str(csv_path))

    return saved_files