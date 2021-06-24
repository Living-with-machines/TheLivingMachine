import pandas as pd
from pathlib import Path
import spacy
import glob

nlp = spacy.load('en_core_web_sm', disable=['ner'])

from utils import process_jsa
from utils import process_rsc
from utils import process_hmd
from utils import process_blbooks
from utils import prepare_sents
from utils import explore_preds

# Specify the query tokens here:
query = "machine"
min_year = 1783
max_year = 1908

#### ----------------------------------
#### Process the JSA corpus

print("Process the JSA corpus")

input_path = "../../workspace/data/" # Path where JSA data is located
output_path = "data/jsa_processed/"
overwrite = False # If False, run the code only if output has not been created.
                  # If True, run the code regardless.

process_jsa.parse_corpus(input_path, output_path, overwrite)

jsa_sents_df = prepare_sents.filter_sents_query("JSA", query)
jsa_sents_df = jsa_sents_df[(jsa_sents_df["year"] >= min_year) & (jsa_sents_df["year"] <= max_year)]
jsa_sents_df.to_csv("data/jsa_processed/JSA_" + query + ".tsv", sep="\t")

#### ----------------------------------
#### Process the RSC corpus

# Data downloaded from https://fedora.clarin-d.uni-saarland.de/rsc_v6/access.html#download.

# We are using:
# * TEI-formatted corpus [v6.0.4](https://fedora.clarin-d.uni-saarland.de/rsc_v6/data/texts/Royal_Society_Corpus_open_v6.0.4_texts_tei.zip) (as separate files).
# * Corresponding metadata [v6.0.4](https://fedora.clarin-d.uni-saarland.de/rsc_v6/data/Royal_Society_Corpus_open_v6.0.4_meta.tsv.zip).

print("Process the RSC corpus")

input_path = "../../workspace/data/RSC/" # Path where RSC data is located
output_path = "data/rsc_processed/"
overwrite = False # If False, run the code only if output has not been created.
                  # If True, run the code regardless.

process_rsc.parse_corpus(input_path, output_path, overwrite)

rsc_sents_df = prepare_sents.filter_sents_query("RSC", query)
rsc_sents_df = rsc_sents_df[(rsc_sents_df["year"] >= min_year) & (rsc_sents_df["year"] <= max_year)]
rsc_sents_df.to_csv("data/rsc_processed/RSC_" + query + ".tsv", sep="\t")

#### ----------------------------------
#### Process the HMD dataset

print("Process the HMD corpus")

hmd_metadata_df = process_hmd.read_metadata("data/hmd_processed/HMD_metadata_all.csv")

hmd_dfs = []
for i in glob.glob("data/hmd_processed/hmd_data/*.csv"): # Files exported by Kaspar
    hmd_dfs.append(process_hmd.process_content(i, query, hmd_metadata_df))
    
hmd_main_df = pd.concat(hmd_dfs, axis=0, ignore_index=True)
hmd_main_df = hmd_main_df[(hmd_main_df["year"] >= min_year) & (hmd_main_df["year"] <= max_year)]
hmd_main_df.to_csv("data/hmd_processed/HMD_" + query + ".tsv", sep="\t")

#### ----------------------------------
#### Process the BLBooks corpus

print("Process the BLB corpus")

blb_metadata_df = process_blbooks.read_metadata("data/blb_processed/book_data.json") # File provided by Kaspar
blb_main_df = process_blbooks.process_content("data/blb_processed/bl_books.csv", query, blb_metadata_df)  # File provided by Kaspar

# Filter by date:
blb_main_df = blb_main_df[blb_main_df["date"].str.isnumeric()]
blb_main_df = blb_main_df[(blb_main_df["date"].astype(int) >= 1783) & (blb_main_df["date"].astype(int) <= 1908)]

blb_main_df.to_csv("data/blb_processed/BLB_" + query + ".tsv", sep="\t")

#### ----------------------------------
#### Linguistic filtering

print("Linguistic filtering")

for dataset in ["jsa", "rsc", "hmd", "blb"]:
    print("*", dataset)
    syndf = pd.read_csv("data/" + dataset + "_processed/" + dataset.upper() + "_" + query + ".tsv", sep="\t")
    syndf['synt'] = prepare_sents.preprocess_pipe(syndf['currentSentence'], nlp)
    syndf = syndf[syndf.apply(lambda x: prepare_sents.filter_sents_synt(x.synt, x.maskedSentence, x.targetExpression), axis=1)]
    syndf["query_label"] = syndf.apply(lambda x: prepare_sents.find_query_deplabel(x.synt, x.maskedSentence, x.targetExpression), axis=1)
    syndf.to_pickle("data/" + dataset + "_processed/" + dataset.upper() + "_" + query + "_synparsed.pkl")

#### ----------------------------------
#### BERT masking

print("BERT masking")

for dataset in ["data/rsc_processed/RSC_" + query + "_synparsed.pkl",
                "data/jsa_processed/JSA_" + query + "_synparsed.pkl",
                "data/hmd_processed/HMD_" + query + "_synparsed.pkl",
                "data/blb_processed/BLB_" + query + "_synparsed.pkl"]:
    
    if not Path(dataset.split(".pkl")[0] + "_pred_bert.pkl").is_file():
        
        print(dataset)

        # Load dataframe where to apply this:
        pred_df = pd.read_pickle(dataset)
        for epoch in ["1760_1850", "1890_1900"]:

            print("*", epoch)

            # Create pipeline depending on the BERT model of the specified period
            # and the number of expected predictions:
            pred_toks = 20
            model_rd = explore_preds.create_mask_pipeline(epoch, pred_toks)

            # Use BERT to find most likely predictions for a mask:
            pred_df["pred_bert_" + epoch] = pred_df.apply(lambda x: explore_preds.bert_masking(x, model_rd), axis=1)
            
            pred_df.to_pickle(dataset.split(".pkl")[0] + "_" + epoch + "_pred_bert.pkl")

        pred_df.to_pickle(dataset.split(".pkl")[0] + "_pred_bert.pkl")