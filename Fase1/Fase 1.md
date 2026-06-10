# Fase 1 - Visualizzazione dei dati

- `AnalisiDati.ipynb`: Notebook con descrizione ELIta, prime visualizzazioni delle emoji.
  - **PCA**, **TruncatedSVD**, **t-SNE** per visualizzare i dati in 2D e 3D.
  - osservazioni per ogni grafico prodotto.
- `AnalisiParole.ipynb`: Notebook con le stesse visualizzazioni del notebook precedente ma focalizzato sulle parole.
  - **PCA**, **TruncatedSVD**, **t-SNE** per visualizzare i dati in 2D e 3D.
  - osservazioni per ogni grafico prodotto.
- `AnalisiPerEmozioni.ipynb`: Notebook con visualizzazioni focalizzate sulle emozioni.
  - analisi: soglia e parole/emojie ambigue, pearson.
  - osservazioni per ogni grafico prodotto.

- File .py:
  - `emotion_config.py`: palette colori per le emozioni di Plutchik, liste emozioni (BASIC, ALL, VAD).
  - `sklearn.py`: ha funzioni per eseguire PCA, TruncatedSVD e t-SNE, con parametri configurabili.
    - `perform_pca_analysis`
    - `perform_tsne_analysis`
    - `perform_svd_analysis`
    - `trova_parole_complesse`
  - `visualizer.py`: ha funzioni per creare grafici 2D e 3D, con parametri configurabili.
    - `plot_pca_emotions`
    - `plot_pca_single`
    - `plot_pca_3d`
    - `plot_soglie_affiancate` (PCA)
    - `plot_complesse_confronto` (PCA)
    - `plot_svd_soglie_affiancate`