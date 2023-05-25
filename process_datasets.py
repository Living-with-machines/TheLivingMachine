from pathlib import Path
import pandas as pd
import spacy
from utils import explore_preds, prepare_sents

nlp = spacy.load('en_core_web_sm', disable=['ner'])

# Specify the query tokens here. Change the query to see the results for a different
# target word:
query = "machine" 
min_year = 1783
max_year = 1908

# This dictionary maps the query to the name that will be displayed in the output file.
generic = {"machine": "machine"}

# This dictionary maps the query to the tokens it will be expanded to.
query_tokens = dict()
query_tokens["machine"] = ["machine", "machines"]

#### ----------------------------------
#### Linguistic filtering
print("Linguistic filtering")

for dataset in ["jsa", "hmd", "blb", "example"]:
    print("*", dataset)
    syndf = pd.read_csv("data/" + dataset + "_processed/" + dataset + "_" + query + ".tsv", sep="\t")
    # If we have more than 100000 sentences, downsample to 100000:
    if query != "machine" and syndf.shape[0] > 65000:
        syndf = syndf.sample(n=65000, random_state=42)
    # Get a sentence ID
    syndf['sentId'] = list(syndf.index.values)
    # Process and filter sentences through syntactic parsing:
    syndf['currentSentence'] = syndf.apply(lambda x: prepare_sents.remove_punctspaces(x["currentSentence"]), axis=1)
    syndf['synt'] = prepare_sents.preprocess_pipe(syndf['currentSentence'], nlp)
    syndf = syndf[syndf.apply(lambda x: prepare_sents.filter_sents_synt(x.synt, x.maskedSentence, x.currentSentence, x.targetExpression), axis=1)]
    syndf["query_label"] = syndf.apply(lambda x: prepare_sents.find_query_deplabel(x.synt, x.maskedSentence, x.targetExpression), axis=1)
    syndf.to_pickle("data/" + dataset + "_processed/" + dataset + "_" + query + "_synparsed.pkl")

#### ----------------------------------
#### BERT masking

print("BERT masking")

for dataset in ["data/jsa_processed/JSA_" + query + "_synparsed.pkl",
                "data/hmd_processed/HMD_" + query + "_synparsed.pkl",
                "data/blb_processed/BLB_" + query + "_synparsed.pkl",
                "data/example_processed/example_" + query + "_synparsed.pkl"]:
    
    if not Path(dataset.split(".pkl")[0] + "_pred_bert.pkl").is_file():

        # Load dataframe where to apply this:
        pred_df = pd.read_pickle(dataset)
        for epoch in ["1760_1900", "contemporary"]:

            print("*", epoch)

            # Create pipeline depending on the BERT model of the specified period
            # and the number of expected predictions:
            pred_toks = 20
            model_rd = explore_preds.create_mask_pipeline(epoch, pred_toks)

            # Use BERT to find most likely predictions for a mask:
            pred_df["pred_bert_" + epoch] = pred_df.apply(lambda x: explore_preds.bert_masking(x, model_rd), axis=1)

        pred_df.to_pickle(dataset.split(".pkl")[0] + "_pred_bert.pkl")
