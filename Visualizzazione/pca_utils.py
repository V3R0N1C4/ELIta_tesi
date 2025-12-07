import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD


def perform_pca_analysis(df, features_columns, dominant_emotions, scale=False, n_components=2):
    """
    Esegue la PCA (supporta 2D o 3D).
    """
    X = df[features_columns].fillna(0)

    if scale:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
    else:
        X_processed = X

    # PCA dinamica (2 o 3 componenti)
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(X_processed)

    # Nomi colonne dinamici (PC1, PC2, PC3...)
    cols = [f'PC{i + 1}' for i in range(n_components)]

    pca_df = pd.DataFrame(data=principalComponents, columns=cols)
    pca_df['Emoji'] = X.index
    pca_df['Emozione Dominante'] = dominant_emotions.values

    return {
        'df': pca_df,
        'variance_ratio': pca.explained_variance_ratio_,
        'components': pca.components_,
        'pca_object': pca,
        'explained_variance': pca.explained_variance_
    }

def perform_tsne_analysis(df, features_columns, dominant_emotions, perplexity=30, scale=False):
    """
    Esegue t-SNE sui dati forniti.
    Args:
        perplexity (int): Parametro cruciale t-SNE.
                          5-30 per dettagli locali, 30-50 per struttura globale.
    """
    X = df[features_columns].fillna(0)

    # Scaling (consigliato per t-SNE)
    if scale:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
    else:
        X_processed = X

    # Esecuzione t-SNE
    # init='pca' rende il risultato più stabile
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    embedded = tsne.fit_transform(X_processed)

    # Creazione DF (Usiamo nomi colonne compatibili con il visualizer esistente)
    tsne_df = pd.DataFrame(data=embedded, columns=['PC1', 'PC2'])
    tsne_df['Emoji'] = X.index
    tsne_df['Emozione Dominante'] = dominant_emotions.values

    return {
        'df': tsne_df,
        'variance_ratio': None,  # t-SNE non ha varianza spiegata
        'components': None  # t-SNE non ha vettori di caricamento lineari
    }

def perform_svd_analysis(df, features_columns, dominant_emotions, n_iter=7):
    """
    Esegue TruncatedSVD sui dati.
    Nota: SVD lavora bene sui dati grezzi (senza sottrarre la media),
    quindi cattura sia la varianza che l'intensità (magnitudo) dei vettori.
    """
    X = df[features_columns].fillna(0)

    # SVD standard (senza scaling/centramento preventivo, altrimenti diventa PCA)
    svd = TruncatedSVD(n_components=2, n_iter=n_iter, random_state=42)
    components_transformed = svd.fit_transform(X)

    # Creiamo il DataFrame
    # Usiamo i nomi 'PC1' e 'PC2' per compatibilità con la funzione di visualizzazione esistente
    svd_df = pd.DataFrame(data=components_transformed, columns=['PC1', 'PC2'])
    svd_df['Emoji'] = X.index
    svd_df['Emozione Dominante'] = dominant_emotions.values

    return {
        'df': svd_df,
        'variance_ratio': svd.explained_variance_ratio_,
        'components': svd.components_,
        'pca_object': svd,  # Passiamo l'oggetto SVD
        'explained_variance': svd.explained_variance_
    }