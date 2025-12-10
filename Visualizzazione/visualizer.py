import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def plot_pca_emotions(pca_data, colors_dict, title, plot_loadings=False, feature_names=None):
    """
    Crea il grafico comparativo (Emoji vs Markers) per la PCA.

    Args:
        pca_data (dict): Il dizionario restituito da perform_pca_analysis.
        colors_dict (dict): Dizionario colore {emozione: hex}.
        title (str): Titolo del grafico.
        plot_loadings (bool): Se True, disegna i vettori (frecce) delle feature originali (utile per VAD).
        feature_names (list): Nomi delle feature originali (necessario se plot_loadings=True).
    """

    df = pca_data['df']
    var_ratio = pca_data['variance_ratio']

    # Creazione subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribuzione con Emoji', 'Distribuzione con Marker'),
        horizontal_spacing=0.12
    )

    # Iteriamo sui colori disponibili
    for emotion, color in colors_dict.items():
        emotion_data = df[df['Emozione Dominante'] == emotion]
        if emotion_data.empty:
            continue

        # 1. Grafico Emoji (Sinistra)
        fig.add_trace(
            go.Scatter(
                x=emotion_data['PC1'], y=emotion_data['PC2'],
                mode='text',
                text=emotion_data['Emoji'],
                textfont=dict(size=18),
                textposition='middle center',
                legendgroup=emotion,
                showlegend=False,
                hovertemplate=f'<b>{emotion}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>'
            ), row=1, col=1
        )

        # 2. Grafico Marker (Destra)
        fig.add_trace(
            go.Scatter(
                x=emotion_data['PC1'], y=emotion_data['PC2'],
                mode='markers',
                marker=dict(
                    color=color,
                    size=12, opacity=0.8,
                    line=dict(width=1.5, color='white')
                ),
                name=emotion,
                legendgroup=emotion,
                showlegend=True,
                hovertemplate=f'<b>{emotion}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>'
            ), row=1, col=2
        )

    # Aggiunta vettori di caricamento (Loadings) per VAD
    if plot_loadings and feature_names:
        loadings = pca_data['components'].T * np.sqrt(pca_data['explained_variance'])

        for i, feature in enumerate(feature_names):
            fig.add_annotation(
                x=loadings[i, 0] * 3,  # Scalato per visibilità
                y=loadings[i, 1] * 3,
                text=feature,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#555",
                font=dict(color="#555", size=12),
                ax=0, ay=0,
                row=1, col=2
            )

    # Layout Assi
    if var_ratio is not None:
        # Etichette per PCA (con varianza)
        x_title = f"PC1 ({var_ratio[0]:.2%} var)"
        y_title = f"PC2 ({var_ratio[1]:.2%} var)"
    else:
        # Etichette generiche per t-SNE
        x_title = "Dimensione 1 (t-SNE)"
        y_title = "Dimensione 2 (t-SNE)"

    fig.update_xaxes(title_text=x_title, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text=y_title, showgrid=True, gridcolor='lightgray')

    # Rimuovi i tick (numeri) dagli assi per t-SNE, perché non hanno significato metrico
    if var_ratio is None:
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

    # Layout Generale
    fig.update_layout(
        title_text=title,
        title_font_size=16,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        legend=dict(
            title=dict(text='Emozione Dominante', font=dict(size=14)),
            font=dict(size=12),
            x=1.02, y=0.5,
            xanchor='left', yanchor='middle'
        )
    )

    return fig

def plot_pca_single(pca_data, colors_dict, title, plot_loadings=False, feature_names=None):
    """
    Crea un grafico singolo per le Parole (solo Marker con hover).
    Evita i subplot affiancati per una visualizzazione più pulita.
    """
    df = pca_data['df']
    var_ratio = pca_data['variance_ratio']

    fig = go.Figure()

    # Iteriamo sui colori disponibili
    for emotion, color in colors_dict.items():
        emotion_data = df[df['Emozione Dominante'] == emotion]
        if emotion_data.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=emotion_data['PC1'], y=emotion_data['PC2'],
                mode='markers',
                marker=dict(
                    color=color,
                    size=8, # Un po' più piccoli per le parole
                    opacity=0.7,
                    line=dict(width=0.5, color='white')
                ),
                name=emotion,
                text=emotion_data['Emoji'], # Qui 'Emoji' contiene in realtà la Parola (dall'indice)
                legendgroup=emotion,
                showlegend=True,
                hovertemplate=f'<b>{emotion}</b><br>Parola: %{{text}}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>'
            )
        )

    # Aggiunta vettori di caricamento (Loadings) per VAD
    if plot_loadings and feature_names:
        loadings = pca_data['components'].T * np.sqrt(pca_data['explained_variance'])

        for i, feature in enumerate(feature_names):
            fig.add_annotation(
                x=loadings[i, 0] * 3,  # Scalato per visibilità
                y=loadings[i, 1] * 3,
                text=feature,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#333",
                font=dict(color="#333", size=14, family="Arial Black"),
                ax=0, ay=0
            )

    # Layout Assi
    if var_ratio is not None:
        x_title = f"PC1 ({var_ratio[0]:.2%} var)"
        y_title = f"PC2 ({var_ratio[1]:.2%} var)"
    else:
        x_title = "Dimensione 1 (t-SNE)"
        y_title = "Dimensione 2 (t-SNE)"

    fig.update_xaxes(title_text=x_title, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text=y_title, showgrid=True, gridcolor='lightgray')

    # Rimuovi i tick per t-SNE
    if var_ratio is None:
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

    # Layout Generale
    fig.update_layout(
        title_text=title,
        title_font_size=16,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=700, # Più alto per vedere meglio i cluster
        legend=dict(
            title=dict(text='Emozione Dominante', font=dict(size=14)),
            font=dict(size=12)
        )
    )

    return fig

def plot_pca_3d(pca_data, colors_dict, title):
    """
    Crea un grafico interattivo 3D (PC1, PC2, PC3).
    """
    df = pca_data['df']
    var_ratio = pca_data['variance_ratio']

    # Controlliamo se abbiamo almeno 3 componenti
    if 'PC3' not in df.columns:
        raise ValueError("Per il grafico 3D devi eseguire la PCA con n_components=3")

    fig = go.Figure()

    # Iteriamo per ogni emozione per assegnare i colori
    for emotion, color in colors_dict.items():
        emotion_data = df[df['Emozione Dominante'] == emotion]
        if emotion_data.empty:
            continue

        fig.add_trace(
            go.Scatter3d(
                x=emotion_data['PC1'],
                y=emotion_data['PC2'],
                z=emotion_data['PC3'],
                mode='markers',  # Usa 'text+markers' se vuoi vedere anche le emoji (può essere pesante)
                marker=dict(
                    size=5,
                    color=color,
                    opacity=0.8,
                    line=dict(width=0.5, color='white')  # Bordo sottile per visibilità
                ),
                text=emotion_data['Emoji'],  # Mostra l'emoji quando passi col mouse
                name=emotion,
                hovertemplate=f'<b>{emotion}</b><br>Emoji: %{{text}}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>'
            )
        )

    # Layout Assi
    fig.update_layout(
        title=title,
        height=700,  # Più alto per godersi il 3D
        scene=dict(
            xaxis_title=f"PC1 ({var_ratio[0]:.2%} var)",
            yaxis_title=f"PC2 ({var_ratio[1]:.2%} var)",
            zaxis_title=f"PC3 ({var_ratio[2]:.2%} var)",
            bgcolor='white'
        ),
        margin=dict(l=0, r=0, b=0, t=50),  # Riduci margini
        legend=dict(x=0, y=1)
    )

    return fig


def plot_top_words_per_emotion(df, emotions_list, colors_dict, top_n=15):
    """
    Crea una griglia di grafici a barre orizzontali mostrati le parole con
    il punteggio più alto per ogni emozione.
    """
    # Calcoliamo quante righe servono (2 colonne fisse)
    rows = (len(emotions_list) + 1) // 2

    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=[f"Top {top_n} - {e.capitalize()}" for e in emotions_list],
        horizontal_spacing=0.15,
        vertical_spacing=0.08
    )

    for i, emotion in enumerate(emotions_list):
        # Calcolo posizione griglia
        row = (i // 2) + 1
        col = (i % 2) + 1

        # 1. Prendi i dati e ordina
        # Ordiniamo e prendiamo le top_n
        top_data = df.sort_values(by=emotion, ascending=True).tail(
            top_n)  # tail perché le barre orizzontali partono dal basso

        # 2. Aggiungi traccia
        fig.add_trace(
            go.Bar(
                x=top_data[emotion],
                y=top_data.index,  # Le parole sono nell'indice
                orientation='h',
                marker=dict(color=colors_dict.get(emotion, '#888')),
                name=emotion,
                showlegend=False,
                text=top_data[emotion].apply(lambda x: f"{x:.2f}"),  # Mostra il valore sulla barra
                textposition='auto'
            ),
            row=row, col=col
        )

    fig.update_layout(
        title_text=f"Parole con maggiore intensità per ogni Emozione (Top {top_n})",
        height=300 * rows,  # Altezza dinamica in base al numero di emozioni
        plot_bgcolor='white'
    )

    return fig


def plot_top_words_per_emotion(df, emotions_list, colors_dict, top_n=15):
    """
    Crea una griglia di grafici a barre orizzontali mostrati le parole con
    il punteggio più alto per ogni emozione.
    """
    # Calcoliamo quante righe servono (2 colonne fisse)
    rows = (len(emotions_list) + 1) // 2

    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=[f"Top {top_n} - {e.capitalize()}" for e in emotions_list],
        horizontal_spacing=0.15,
        vertical_spacing=0.08
    )

    for i, emotion in enumerate(emotions_list):
        # Calcolo posizione griglia
        row = (i // 2) + 1
        col = (i % 2) + 1

        # 1. Prendi i dati e ordina
        # Ordiniamo e prendiamo le top_n
        top_data = df.sort_values(by=emotion, ascending=True).tail(
            top_n)  # tail perché le barre orizzontali partono dal basso

        # 2. Aggiungi traccia
        fig.add_trace(
            go.Bar(
                x=top_data[emotion],
                y=top_data.index,  # Le parole sono nell'indice
                orientation='h',
                marker=dict(color=colors_dict.get(emotion, '#888')),
                name=emotion,
                showlegend=False,
                text=top_data[emotion].apply(lambda x: f"{x:.2f}"),  # Mostra il valore sulla barra
                textposition='auto'
            ),
            row=row, col=col
        )

    fig.update_layout(
        title_text=f"Parole con maggiore intensità per ogni Emozione (Top {top_n})",
        height=300 * rows,  # Altezza dinamica in base al numero di emozioni
        plot_bgcolor='white'
    )

    return fig