import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


def perform_pca_analysis(df, features_columns, dominant_emotions, scale=False):
    """
    Esegue la PCA sui dati forniti.

    Args:
        df (pd.DataFrame): Il dataframe originale.
        features_columns (list): Le colonne su cui fare PCA.
        dominant_emotions (pd.Series): La serie con l'emozione dominante per ogni riga (per colorare).
        scale (bool): Se True, applica StandardScaler prima della PCA.

    Returns:
        dict: Contiene il df risultante, variance_ratio, components e l'oggetto pca.
    """

    X = df[features_columns].fillna(0)

    # 2. Scaling (opzionale)
    if scale:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
    else:
        X_processed = X

    # 3. PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X_processed)

    # 4. Creazione DF risultati
    pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    pca_df['Emoji'] = X.index
    # Assicuriamo che l'indice corrisponda per assegnare l'emozione corretta
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