import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Fase1.emotion_config import EMOTION_COLORS, BASIC_EMOTIONS

# costanti
SEVEN_EMOTIONS = ['gioia', 'tristezza', 'rabbia', 'paura', 'disgusto', 'fiducia', 'sorpresa']
POSITIVE       = {'gioia', 'fiducia', 'sorpresa', 'aspettativa'}
NEGATIVE       = {'tristezza', 'rabbia', 'paura', 'disgusto'}
POS_FILTER     = {'NOUN', 'VERB', 'ADJ', 'ADV', 'EMOJI', 'PRONP', 'PRON', 'ADP', 'AUX', 'X'}


# caricamento csv
def load_corpus(corpus_csv, tokens_csv):
    """Carica corpus e token con normalizzazione standard (lower/strip)."""
    df_corpus = pd.read_csv(corpus_csv)
    df_tokens = pd.read_csv(tokens_csv)
    df_tokens['lemma'] = df_tokens['lemma'].astype(str).str.lower().str.strip()
    df_tokens['pos']   = df_tokens['pos'].astype(str).str.upper().str.strip()
    return df_corpus, df_tokens

def load_elita_matrix(path, emotions=BASIC_EMOTIONS):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.lower().str.strip()
    return df[emotions].fillna(0)

def load_all_matrices(elita_csv, alpha_02_csv, alpha_05_csv, alpha_08_csv):
    """Restituisce un dict ordinato con le 4 versioni del lessico ELIta."""
    return {
        'Originale (α=0)': load_elita_matrix(elita_csv),
        'Ibrido (α=0.2)':  load_elita_matrix(alpha_02_csv),
        'Ibrido (α=0.5)':  load_elita_matrix(alpha_05_csv),
        'Ibrido (α=0.8)':  load_elita_matrix(alpha_08_csv),
    }


# analisi
def detect_emotions(df_corpus, df_tokens, df_elita, emotions=BASIC_EMOTIONS):
    """
    Aggrega i punteggi ELIta per ogni documento (score grezzo).
    Filtra solo POS in POS_FILTER (ADJ, NOUN, VERB, EMOJI).
    Restituisce un DataFrame con una riga per documento.
    """
    df_f    = df_tokens[df_tokens['pos'].isin(POS_FILTER)].copy()
    eidx    = set(df_elita.index)
    tok     = df_f.groupby('doc_id')['lemma'].apply(list).to_dict()
    results = []
    for _, row in df_corpus.iterrows():
        did   = row['doc_id']
        lemmi = tok.get(did, [])
        sc    = {e: 0.0 for e in emotions}
        found = 0
        for lemma in lemmi:
            if lemma in eidx:
                found += 1
                for e in emotions:
                    sc[e] += df_elita.loc[lemma, e]
        results.append({
            'doc_id':            did,
            'type':              row.get('type', ''),
            'n_tokens_matched':  found,
            **sc,
            'dominant_emotion':  max(sc, key=sc.get) if found > 0 else 'neutrale',
        })
    return pd.DataFrame(results)

def apply_corpus_mean_norm(df_results, emotions=BASIC_EMOTIONS):
    """Corpus_mean ItEm (Formula 3.5): divide ogni score per la media di corpus"""
    mu      = df_results[emotions].mean()
    df_norm = df_results.copy()
    for e in emotions:
        if mu[e] > 0:
            df_norm[e] = df_results[e] / mu[e]
    df_norm['dominant_emotion'] = df_norm[emotions].apply(
        lambda r: r.idxmax() if r.sum() > 0 else 'neutrale', axis=1
    )
    return df_norm, mu

def compute_mu_e(df_corpus, df_tokens, df_elita, emotions=BASIC_EMOTIONS):
    df_raw = detect_emotions(df_corpus, df_tokens, df_elita, emotions)
    return df_raw[emotions].mean()

def score_single_document(doc_id, tok_dict, elita_idx, df_elita, emotions=BASIC_EMOTIONS):
    """Restituisce (vettore_grezzo, n_token_trovati) per un singolo documento"""
    lemmi = tok_dict.get(doc_id, [])
    sc    = {e: 0.0 for e in emotions}
    found = 0
    for lemma in lemmi:
        if lemma in elita_idx:
            found += 1
            for e in emotions:
                sc[e] += df_elita.loc[lemma, e]
    return sc, found

def normalizza(sc, mu_e, emotions=BASIC_EMOTIONS):
    """Corpus_mean normalisation"""
    return {e: (sc[e] / mu_e[e] if mu_e[e] > 0 else 0.0) for e in emotions}

def emozione_dominante(sc, emotions=BASIC_EMOTIONS):
    """Restituisce l'emozione con score massimo, o neutrale se tutto zero"""
    if sum(sc[e] for e in emotions) == 0:
        return 'neutrale'
    return max(emotions, key=lambda e: sc[e])


# grafici
def plot_emotion_bars(counts, total, title, emotions=BASIC_EMOTIONS, colors=EMOTION_COLORS, height=450):
    """grafico a barre: distribuzione emozioni dominanti in percentuale"""
    fig = go.Figure()
    for e in emotions + ['neutrale']:
        n = counts.get(e, 0)
        fig.add_trace(go.Bar(
            name=e, x=[e], y=[round(n / total * 100, 1)],
            marker_color=colors.get(e, '#999'),
            text=['{:.0f}%'.format(n / total * 100)],
            textposition='outside',
        ))
    fig.update_layout(title=title, barmode='group', height=height, showlegend=False)
    return fig

def plot_emotion_grid(panels, total, title, emotions=BASIC_EMOTIONS, colors=EMOTION_COLORS, height=None):
    """
    Griglia a barre: 1×2 o 2×2
    panels: lista di (label, df_results)
    """
    n = len(panels)
    if n <= 2:
        rows, cols = 1, 2
        positions = [(1, 1), (1, 2)]
        height = height or 400
    else:
        rows, cols = 2, 2
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        height = height or 700
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[label for label, _ in panels],
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )
    for idx, (_, df_r) in enumerate(panels):
        counts = df_r['dominant_emotion'].value_counts()
        r, c   = positions[idx]
        for e in emotions + ['neutrale']:
            n = counts.get(e, 0)
            fig.add_trace(go.Bar(
                name=e, x=[e], y=[round(n / total * 100, 1)],
                marker_color=colors.get(e, '#999'),
                showlegend=(idx == 0), legendgroup=e,
                text=['{:.0f}%'.format(n / total * 100)],
                textposition='outside',
            ), row=r, col=c)
    fig.update_layout(title=title, barmode='group', height=height)
    return fig

def plot_radar_single(sc_norm, emo_dominant, title, emotions=BASIC_EMOTIONS, colors=EMOTION_COLORS, height=480):
    emos_loop = emotions + [emotions[0]]
    fig = go.Figure(go.Scatterpolar(
        r=[sc_norm[e] for e in emos_loop],
        theta=emos_loop,
        fill='toself',
        line_color=colors.get(emo_dominant, '#999'),
        opacity=0.7,
        name=emo_dominant,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title=title,
        height=height,
    )
    return fig

def plot_radar_comparison(traces_data, title, emotions=BASIC_EMOTIONS, height=520):
    """
    Radar con tracce sovrapposte.
    traces_data: lista di (label, sc_norm, color, opacity, line_dash).
    """
    emos_loop = emotions + [emotions[0]]
    fig = go.Figure()
    for label, sc, color, opacity, dash in traces_data:
        fig.add_trace(go.Scatterpolar(
            r=[sc[e] for e in emos_loop],
            theta=emos_loop,
            fill='toself',
            opacity=opacity,
            name=label,
            line_color=color,
            line_dash=dash,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title=title,
        height=height,
    )
    return fig

def plot_raw_vs_norm_bars(totals, sc_norm, doc_id, emotions=BASIC_EMOTIONS, colors=EMOTION_COLORS, height=450):
    bar_colors = [colors.get(e, '#999') for e in emotions]
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Score grezzo S_e', 'Corpus_mean normalizzato S_e_norm'],
        horizontal_spacing=0.12,
    )
    fig.add_trace(go.Bar(
        x=emotions, y=[totals[e] for e in emotions],
        marker_color=bar_colors,
        text=[f'{totals[e]:.2f}' for e in emotions],
        textposition='outside', name='Raw',
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=emotions, y=[sc_norm[e] for e in emotions],
        marker_color=bar_colors,
        text=[f'{sc_norm[e]:.2f}' for e in emotions],
        textposition='outside', name='Normalizzato',
    ), row=1, col=2)
    fig.update_layout(
        title=f'Score emotivi — {doc_id} (ELIta α=0.5)',
        showlegend=False,
        height=height,
    )
    return fig