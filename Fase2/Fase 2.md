# Fase 2 - Ricalcolo punteggi

 - `Valutazione e Ricalcolo.ipynb`: 
   - Dataset iniziale: `df_emotions`.
     1. calcolata la **distintività** di ogni parola per ogni emozione, `((max1 - max2) / (max1)) * (max1 - mean)`.
       - ottenuto `df_distinctiveness`
     2. calcolati i **centroidi** delle emozioni usando le 50 parole più distintive come semi.
       - ottenuto `df_centroids`
     3. calcolata la `cosine_similarity` tra i centroidi e i vettori di ogni parola, ottenendo un nuovo punteggio per ogni parola.
       - ottenuto `df_cos_sim`
     4. applicata la formula `e = 𝛼cos + (1 - 𝛼)e`, (cos = `df_cos_sim`, e = `df_emotions`)
       - ottenuto `df_alpha_02`, `df_alpha_05`, `df_alpha_08`
   - analisi cambiamento tra i punteggi originali e quelli ricalcolati
   - visualizzazione dei risultati con grafici pca
   - mappa semantica con NetworkX
   - salvataggio dei nuovi dataset `df_alpha_02`, `df_alpha_05`, `df_alpha_08`