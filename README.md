# The Living Machine

This repository has the code for the experiments used for the paper 'The Living Machine'

## Code

Python script `prepare_datasets.py`:
1. Processes the different datasets (HMD, JSA, RSC, BLB).
2. Gets all sentences which contain certain query tokens (machine, machines, etc), in the date range 1783-1908. The output is a `.tsv` file with one sentece per row (and relevant metadata in the other columns), stored as `data/xxx_processed/XXX_machine.tsv`.
3. Parses the sentences, finds the syntactic role of the query token, and filters out sentences which do not look like a sentence (e.g. no verb). The output is `data/xxx_processed/XXX_synparsed.pkl`, which has the same structure as the `.tsv` from step 2, with one additional column, which has information on the syntactic role of the query token in the sentence.
4. Using BERT to predict the most likely word instead of the query word. The output is `data/xxx_processed/XXX_synparsed_pred_bert.pkl`, which is basically the dataframe from step 3, with two additional columns: one for each language model (1760-1900 and contemporary), each with the predictions using the different language models.

Notebook `explore_predictions.ipynb` uses the predictions produced by the BERT models to find the plausibility that the masked expression is instead filled by one of the following concepts: "machine", "woman", "girl", "slave", "artisan", and "boy", using word2vec.

Notebook `explore_data.ipynb` provides stats about the datasets more generally (number of sentences per dataset, etc).
