# Fase 3 - Confronto risultati

 - `Raccolta_Reddit.ipynb`: raccolta dati da Reddit, nel subreddit `r/italia` cerchiamo i post con keyword: notizie, 
    film e sport. Per ogni keyword raccogliamo 100 post, per ogni post 50 commenti.

   I dati raccolti vengono salvati in:
   - `posts_Italia_multi.csv` — un post per riga, con colonna `keyword`
   - `comments_Italia_multi.csv` — un commento per riga, con `post_id` come chiave di join
   - `corpus_Italia_multi.csv` — long format: una riga per documento (post o commento), con `keyword` per filtrare per topic
   
   Dopo la raccolta avviene la fase di tokenizzazione, lemmatizzazione e di POS-taggin utilizzando `it_core_news_sm` di spaCy.
   
   Il risultato di questa fase è salvato in `tokens_Italia_notizie.csv`. Con le colonne `doc_id` (post o commento), `keyword`, 
     `token`, `lemma` e `pos`.
 - `Confronto_Corpus.ipynb`: applichiamo elita al corpus di Reddit.
   1. Elita raw (`ELITA_CSV` - `ELIta_INTENSITY_Matrix.csv`) applicato a `df_corpus` (`corpus_Italia_multi.csv`)
      1. nel risultato **aspettativa** è l'emozione dominante per ogni documento (post o commento), 57%
   2. Per cercare di migliorere i risultati, si esegue lo stesso procedimento del passo precedente, escludendo **aspettativa**
      come emozione in analisi.
      1. il risulto `corpus_7emo` ora mostra **gioia** come emozione dominante per ogni documento, 50%. 
      2. questo metodo non è quindi efficace per migliorare i risultati.
   3. ItEm con la **formula 3.5** normalizzava il corpus, quando un emozione ha un valore medio alto su tutto il corpus
        (es. aspettativa) dividerla per la sua media, riporta le emozioni ad essere più confrontabili tra di loro.
      1. il risulto `df_corpus_mean` mostra emozioni più bilanciate, tutte con valori intorno al 10%, con **gioia** e **disgusto** 
         leggermente più rappresentate.
      2. tutte le versioni di elita (alpha = [0, 0.2, 0.5, 0.8]) vengono applicate a`df_corpus_mean`
      3. confronto presenza emozioni positive e negative nei 2 corpus (raw, normalizzato)
 - `Commento_Singolo.ipynb`: applichiamo elita al corpus di Reddit, ma solo per quanto riguarda un commento specifico.
   - vediamo la differenza tra elita raw e normalizzato, e come cambia la distribuzione delle emozioni.
 - `Analisi_Discussione.ipynb`: applichiamo elita normalizzato a un post specifico, poi lo applichiamo anche ai sui commenti,
    per vedere se c'è uno shift emozionale tra post e commenti.
   - Calcoliamo anche quante volte un post positivo genra in una disccussione negativa e viceversa.