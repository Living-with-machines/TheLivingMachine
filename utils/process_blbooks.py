import pandas as pd
import json
import re

def read_metadata(f):
    with open('data/blb_processed/book_data.json') as fr:
        blb_metadata = json.load(fr)
    
    dBLBmetadata = []
    for k in blb_metadata:
        identifier = k["identifier"]
        date = k["date"]
        if date == "":
            date = k["datefield"]
        shelfmarks = "/".join(k["shelfmarks"])
        publisher = k["publisher"]
        title = "/".join(k["title"])
        edition = k["edition"]
        contributors = ""
        creators = ""
        if "contributor" in k["authors"]:
            contributors = "/".join(k["authors"]["contributor"])
        if "creator" in k["authors"]:
            creators = "/".join(k["authors"]["creator"])
        dBLBmetadata.append([identifier, date, shelfmarks, publisher, title, edition, contributors, creators])
        
    blb_mtdt = pd.DataFrame.from_records(dBLBmetadata, columns = ["identifier", "date", "shelfmarks", "publisher",
                                                                  "title", "edition", "contributors", "creators"])
    
    blb_mtdt.to_csv("data/blb_processed/BLB_metadata.csv")
    
    return blb_mtdt

def mask_target(match, currentSentence):
    # Mark the target expression in the sentence:
    markedSentence = re.sub(r"\b%s\b" % match, "$***" + match + "***$", currentSentence, 1, flags=re.IGNORECASE)
    
    # Mask the target expression in the sentence:
    maskedSentence = re.sub(r"\b%s\b" % match, " [MASK] ", currentSentence, 1, flags=re.IGNORECASE)
    maskedSentence = re.sub(" +", " ", maskedSentence)
    return markedSentence, maskedSentence

def process_content(i, query, blb_metadata_df):
    
    # Read csv with sentences:
    one_publ = pd.read_csv(i)
    
    # Drop dummy "Unnamed: 0" column
    one_publ =  one_publ.drop(columns=["Unnamed: 0"])
    
    # Multiple target expressions are separated by <SEP>. Split them
    # and create one row per sentence/targetExpression pair:
    one_publ = one_publ.assign(hits=one_publ.hits.str.split("<SEP>")).explode('hits')

    # Rename columns to match other datasets:
    one_publ = one_publ.rename(columns = {"hits": "targetExpression", "previous_sentence": "prevSentence",
                                          "target_sentence": "currentSentence", "next_sentence": "nextSentence"})

    # Filter by targetExpression/query:
    one_publ = one_publ[one_publ["targetExpression"].str.contains(query, flags=re.IGNORECASE)]

    # Mark and mask the target expression in the current sentence:
    one_publ[["markedSentence", "maskedSentence"]] = one_publ.apply(lambda x: pd.Series(mask_target(x.targetExpression, x.currentSentence)), axis=1)
    
    one_publ['mtdt_id'] = one_publ['article_path'].apply(lambda x: x.split("_")[0])
    one_publ = one_publ.merge(blb_metadata_df, left_on="mtdt_id", right_on="identifier")

    # Reorder the columns:
    one_publ = one_publ[['article_path', 'identifier', 'date', 'shelfmarks', 'publisher',
                         'title', 'edition', 'contributors', 'creators', 
                         'prevSentence', 'currentSentence', 'markedSentence',
                         'maskedSentence', 'nextSentence', 'targetExpression']]
    
    return one_publ