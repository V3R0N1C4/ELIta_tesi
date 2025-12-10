import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

try:
    from .pca_utils import perform_pca_analysis, perform_svd_analysis, trova_parole_complesse
    from .emotion_config import EMOTION_COLORS, BASIC_EMOTIONS
except ImportError:
    from pca_utils import  perform_pca_analysis, perform_svd_analysis, trova_parole_complesse
    from emotion_config import EMOTION_COLORS, BASIC_EMOTIONS

def plot_pca_emotions(pca_data, colors_dict, title, plot_loadings=False, feature_names=None):
    """
    Crea il grafico comparativo (Emoji vs Markers) per la PCA.
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
                x=loadings[i, 0] * 3,
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
        x_title = f"PC1 ({var_ratio[0]:.2%} var)"
        y_title = f"PC2 ({var_ratio[1]:.2%} var)"
    else:
        x_title = "Dimensione 1 (t-SNE)"
        y_title = "Dimensione 2 (t-SNE)"

    fig.update_xaxes(title_text=x_title, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text=y_title, showgrid=True, gridcolor='lightgray')

    if var_ratio is None:
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

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
    """
    df = pca_data['df']
    var_ratio = pca_data['variance_ratio']

    fig = go.Figure()

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
                    size=8,
                    opacity=0.7,
                    line=dict(width=0.5, color='white')
                ),
                name=emotion,
                text=emotion_data['Emoji'],
                legendgroup=emotion,
                showlegend=True,
                hovertemplate=f'<b>{emotion}</b><br>Parola: %{{text}}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>'
            )
        )

    if plot_loadings and feature_names:
        loadings = pca_data['components'].T * np.sqrt(pca_data['explained_variance'])
        for i, feature in enumerate(feature_names):
            fig.add_annotation(
                x=loadings[i, 0] * 3,
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

    if var_ratio is not None:
        x_title = f"PC1 ({var_ratio[0]:.2%} var)"
        y_title = f"PC2 ({var_ratio[1]:.2%} var)"
    else:
        x_title = "Dimensione 1 (t-SNE)"
        y_title = "Dimensione 2 (t-SNE)"

    fig.update_xaxes(title_text=x_title, showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text=y_title, showgrid=True, gridcolor='lightgray')

    if var_ratio is None:
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

    fig.update_layout(
        title_text=title,
        title_font_size=16,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=700,
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

    if 'PC3' not in df.columns:
        raise ValueError("Per il grafico 3D devi eseguire la PCA con n_components=3")

    fig = go.Figure()

    for emotion, color in colors_dict.items():
        emotion_data = df[df['Emozione Dominante'] == emotion]
        if emotion_data.empty:
            continue

        fig.add_trace(
            go.Scatter3d(
                x=emotion_data['PC1'],
                y=emotion_data['PC2'],
                z=emotion_data['PC3'],
                mode='markers',
                marker=dict(
                    size=5, color=color, opacity=0.8,
                    line=dict(width=0.5, color='white')
                ),
                text=emotion_data['Emoji'],
                name=emotion,
                hovertemplate=f'<b>{emotion}</b><br>Emoji: %{{text}}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>'
            )
        )

    fig.update_layout(
        title=title,
        height=700,
        scene=dict(
            xaxis_title=f"PC1 ({var_ratio[0]:.2%} var)",
            yaxis_title=f"PC2 ({var_ratio[1]:.2%} var)",
            zaxis_title=f"PC3 ({var_ratio[2]:.2%} var)",
            bgcolor='white'
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(x=0, y=1)
    )
    return fig

def plot_soglie_affiancate(df, emozione_target):
    """
    Genera 3 grafici affiancati per le soglie 0.25, 0.50, 0.75
    per una specifica emozione target.

    Args:
        df: DataFrame contenente i dati (es. df_words)
        emozione_target: stringa (es. 'gioia', 'tristezza')
    """
    soglie = [0.25, 0.50, 0.75]
    titles = [f"Soglia > {s}" for s in soglie]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=titles,
        horizontal_spacing=0.1
    )

    show_legend_tracker = set()

    for i, soglia in enumerate(soglie):
        # 1. Filtro
        subset = df[df[emozione_target] > soglia].copy()
        n_parole = len(subset)

        # Aggiorna il titolo del subplot
        fig.layout.annotations[i].text = f"Soglia > {soglia} (n={n_parole})"

        if n_parole < 3:
            continue

        # 2. PCA (Calcolata internamente)
        dominant = subset[BASIC_EMOTIONS].idxmax(axis=1)
        pca_res = perform_pca_analysis(subset, BASIC_EMOTIONS, dominant, scale=True)
        df_pca = pca_res['df']

        # 3. Plot
        for emozione in df_pca['Emozione Dominante'].unique():
            df_emo = df_pca[df_pca['Emozione Dominante'] == emozione]

            show_leg = False
            if emozione not in show_legend_tracker:
                show_leg = True
                show_legend_tracker.add(emozione)

            fig.add_trace(
                go.Scatter(
                    x=df_emo['PC1'], y=df_emo['PC2'],
                    mode='markers',
                    marker=dict(
                        color=EMOTION_COLORS.get(emozione, '#888'),
                        size=6, opacity=0.7
                    ),
                    name=emozione,
                    text=df_emo['Emoji'],  # Indice (parola)
                    legendgroup=emozione,
                    showlegend=show_leg,
                    hovertemplate=f'<b>{emozione}</b><br>Parola: %{{text}}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>'
                ),
                row=1, col=i + 1
            )

    fig.update_layout(
        title_text=f"Analisi Sensibilità Soglia: <b>{emozione_target.capitalize()}</b>",
        height=500,
        width=1200,
        plot_bgcolor='white'
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgray', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgray', mirror=True, showticklabels=False)

    return fig


def plot_complesse_confronto(df):
    """
    Genera un grafico affiancato (0.5 vs 0.75) per le parole ambigue.
    """
    soglie = [0.5, 0.75]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"Parole Ambigue (Soglia > {s})" for s in soglie],
        horizontal_spacing=0.12
    )

    show_legend_tracker = set()

    for i, soglia in enumerate(soglie):
        # 1. Usa la funzione importata da pca_utils
        complesse_df = trova_parole_complesse(df, soglia=soglia)
        n_parole = len(complesse_df)

        fig.layout.annotations[i].text = f"Soglia > {soglia} (n={n_parole})"

        if n_parole < 3:
            fig.add_annotation(
                text="Dati insufficienti (<3 parole)",
                xref=f"x{i + 1}", yref=f"y{i + 1}",
                showarrow=False, row=1, col=i + 1
            )
            continue

        # 2. Calcolo PCA
        dominant = complesse_df[BASIC_EMOTIONS].idxmax(axis=1)
        pca_res = perform_pca_analysis(complesse_df, BASIC_EMOTIONS, dominant, scale=True)
        df_pca = pca_res['df']

        # 3. Plot
        for emozione in df_pca['Emozione Dominante'].unique():
            df_emo = df_pca[df_pca['Emozione Dominante'] == emozione]

            show_leg = False
            if emozione not in show_legend_tracker:
                show_leg = True
                show_legend_tracker.add(emozione)

            fig.add_trace(
                go.Scatter(
                    x=df_emo['PC1'], y=df_emo['PC2'],
                    mode='markers',
                    marker=dict(
                        color=EMOTION_COLORS.get(emozione, '#888'),
                        size=7, opacity=0.7,
                        line=dict(width=0.5, color='white')
                    ),
                    name=emozione,
                    text=df_emo['Emoji'],
                    legendgroup=emozione,
                    showlegend=show_leg,
                    hovertemplate=f'<b>{emozione}</b><br>Parola: %{{text}}<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>'
                ),
                row=1, col=i + 1
            )

    fig.update_layout(
        title_text="Analisi Parole Ambigue: Confronto Soglie (0.5 vs 0.75)",
        height=500,
        width=1100,
        plot_bgcolor='white'
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgray', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgray', mirror=True, showticklabels=False)

    return fig


def plot_svd_soglie_affiancate(df, emozione_target):
    """
    Genera 3 grafici affiancati per le soglie 0.25, 0.50, 0.75
    utilizzando TruncatedSVD per una specifica emozione target.
    """
    soglie = [0.25, 0.50, 0.75]
    titles = [f"Soglia > {s}" for s in soglie]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=titles,
        horizontal_spacing=0.1
    )

    show_legend_tracker = set()

    for i, soglia in enumerate(soglie):
        # 1. Filtro
        subset = df[df[emozione_target] > soglia].copy()
        n_parole = len(subset)

        fig.layout.annotations[i].text = f"Soglia > {soglia} (n={n_parole})"

        if n_parole < 3:
            continue

        # 2. Calcolo SVD (Invece di PCA)
        dominant = subset[BASIC_EMOTIONS].idxmax(axis=1)

        # SVD lavora sui dati grezzi, quindi preserva l'intensità (0 è 0)
        svd_res = perform_svd_analysis(subset, BASIC_EMOTIONS, dominant)
        df_svd = svd_res['df']

        # 3. Plot
        for emozione in df_svd['Emozione Dominante'].unique():
            df_emo = df_svd[df_svd['Emozione Dominante'] == emozione]

            show_leg = False
            if emozione not in show_legend_tracker:
                show_leg = True
                show_legend_tracker.add(emozione)

            fig.add_trace(
                go.Scatter(
                    x=df_emo['PC1'], y=df_emo['PC2'],
                    mode='markers',
                    marker=dict(
                        color=EMOTION_COLORS.get(emozione, '#888'),
                        size=6, opacity=0.7,
                        symbol='diamond',  # Usiamo un rombo per distinguerlo a colpo d'occhio dalla PCA
                        line=dict(width=0.5, color='white')
                    ),
                    name=emozione,
                    text=df_emo['Emoji'],  # Parola
                    legendgroup=emozione,
                    showlegend=show_leg,
                    hovertemplate=f'<b>{emozione}</b><br>Parola: %{{text}}<br>SVD1: %{{x:.2f}}<br>SVD2: %{{y:.2f}}<extra></extra>'
                ),
                row=1, col=i + 1
            )

    fig.update_layout(
        title_text=f"Analisi SVD (Intensità): <b>{emozione_target.capitalize()}</b>",
        height=500,
        width=1200,
        plot_bgcolor='white'
    )

    # Rimuoviamo i tick labels perché SVD produce coordinate di intensità non standardizzate
    fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgray', mirror=True, showticklabels=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgray', mirror=True, showticklabels=False)

    return fig