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
    fig.update_xaxes(title_text=f"PC1 ({var_ratio[0]:.2%} var)", showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text=f"PC2 ({var_ratio[1]:.2%} var)", showgrid=True, gridcolor='lightgray')

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