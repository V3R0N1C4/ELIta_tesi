import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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