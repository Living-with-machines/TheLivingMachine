import pandas as pd
from pathlib import Path
import spacy
import glob

nlp = spacy.load('en_core_web_sm', disable=['ner'])

from utils import process_jsa
from utils import process_hmd
from utils import process_blbooks
from utils import prepare_sents

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
#### Process the JSA corpus

print("Process the JSA corpus")

input_path = "../../workspace/data/" # Path where JSA data is located
output_path = "data/jsa_processed/"
overwrite = False # If False, run the code only if output has not been created.
                  # If True, run the code regardless.

process_jsa.parse_corpus(input_path, output_path, overwrite)

jsa_sents_df = prepare_sents.filter_sents_query("JSA", query_tokens[query])
jsa_sents_df = jsa_sents_df[(jsa_sents_df["year"] >= min_year) & (jsa_sents_df["year"] <= max_year)]
jsa_sents_df.to_csv("data/jsa_processed/JSA_" + query + ".tsv", sep="\t", index=False)

#### ----------------------------------
#### Process the HMD dataset

print("Process the HMD corpus")

hmd_metadata_df = process_hmd.read_metadata("data/hmd_processed/HMD_metadata_all.csv")

hmd_dfs = []
for i in glob.glob("data/hmd_processed/hmd_data_" + generic[query] + "_words/*.csv"): # Files exported by Kaspar
    hmd_dfs.append(process_hmd.process_content(i, query_tokens[query], hmd_metadata_df))
    
hmd_main_df = pd.concat(hmd_dfs, axis=0, ignore_index=True)
hmd_main_df = hmd_main_df[(hmd_main_df["year"] >= min_year) & (hmd_main_df["year"] <= max_year)]
hmd_main_df.to_csv("data/hmd_processed/HMD_" + query + ".tsv", sep="\t", index=False)

#### ----------------------------------
#### Process the BLBooks corpus
print("Process the BLB corpus")

blb_metadata_df = process_blbooks.read_metadata("data/blb_processed/book_data.json") # File provided by Kaspar
blb_main_df = process_blbooks.process_content("data/blb_processed/bl_books_" + generic[query] + "_words.csv", query_tokens[query], blb_metadata_df)  # File provided by Kaspar

# Filter by date:
blb_main_df = blb_main_df[blb_main_df["date"].str.isnumeric()]
blb_main_df = blb_main_df[(blb_main_df["date"].astype(int) >= 1783) & (blb_main_df["date"].astype(int) <= 1908)]

blb_main_df.to_csv("data/blb_processed/BLB_" + query + ".tsv", sep="\t", index=False)