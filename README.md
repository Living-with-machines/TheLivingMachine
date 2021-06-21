# BERTMachine
[BERT as a/and the animate machine]

## Code

Notebook `prepare_datasets.ipynb` does the following (for the JSA and RSC datasets):
1. Converts the dataset into:
    * A metadata `.tsv` file (stored in `data/xxx_processed/XXX_metadata.tsv`).
    * One file per document in the collection (stored as `data/xxx_processed/XXX_content/yyyyyyy.txt`
2. Gets all sentences which contain certain query tokens (machine, machines, engine, engines), in the date range 1783-1908. The output is a `.tsv` file with one sentece per row (and relevant metadata in the other columns), stored as `data/xxx_processed/XXX_machines.tsv`.
3. Parses the sentences, finds the syntactic role of the query token, and filters out sentences which do not look like a sentence (e.g. no verb). The output is `data/xxx_processed/XXX_synparsed.pkl`, which is basically the same `.tsv` from step 2, with one additional column, which has information on the syntactic role of the query token in the sentence.
4. Using BERT to predict the most likely word instead of the query word. The output is `data/xxx_processed/XXX_synparsed_pred_bert.pkl`, which is basically the dataframe from step 3, with four additional columns: one for each language model (1760-1850, 1850-1875, 1875-1890, 1890-1900), each with the predictions using the different language models.

Notebook `explore_rsc.ipynb` explores the Royal Society Corpus metadata.