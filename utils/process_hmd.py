import pandas as pd
import re

def parse_path(ap):
    ap_split = ap.split("/")[-1].split(".txt")[0].split("_")
    publ_code = int(ap_split[0])
    issue_code = int(ap_split[1])
    item_code = ap_split[2]
    return publ_code, issue_code, item_code

def mask_target(match, currentSentence):
    # Mark the target expression in the sentence:
    markedSentence = re.sub(r"\b%s\b" % match, "$***" + match + "***$", currentSentence, 1, flags=re.IGNORECASE)
    
    # Mask the target expression in the sentence:
    maskedSentence = re.sub(r"\b%s\b" % match, " [MASK] ", currentSentence, 1, flags=re.IGNORECASE)
    maskedSentence = re.sub(" +", " ", maskedSentence)
    return markedSentence, maskedSentence

def read_metadata(f):
    # This metadata file has been created from the PostgreSQL DB
    # with HMD, and has the following columns (with info from the
    # DB table.column problenance). The DB is `newspapers_hmd_staging`:
    # * item_id: item.id
    # * item_code: item.item_code
    # * issue_id: issue.id
    # * issue_code: issue.issue_code
    # * publication_id: publication.id
    # * publication_code: publication.code
    # * issue_date: issue.issue_date
    # * item_type: item.item_type
    # * word_count: item.word_count
    # * ocr_quality_mean: item.ocr_quality_mean
    # * title: publication.title
    # * location: publication.location
    mtdt_df = pd.read_csv(f)
    mtdt_df["issue_code"] = mtdt_df["issue_code"].astype(int)
    mtdt_df["publication_code"] = mtdt_df["publication_code"].astype(int)
    return mtdt_df

def process_content(i, query_tokens, hmd_metadata_df):
    
    query_tokens = [r"\b" + x + r"\b" for x in query_tokens]
    query_tokens = '|'.join(query_tokens)
    
    # Read csv with sentences:
    one_publ = pd.read_csv(i)
    
    # Drop dummy "Unnamed: 0" column
    one_publ =  one_publ.drop(columns=["Unnamed: 0"])

    # Find the publication code, the issue code, and the item code,
    # based on the article path.
    one_publ[["publication_code", "issue_code", "item_code"]] = one_publ.apply(lambda x: pd.Series(parse_path(x.article_path)), axis=1)
    
    # Multiple target expressions are separated by <SEP>. Split them
    # and create one row per sentence/targetExpression pair:
    one_publ = one_publ.assign(hits=one_publ.hits.str.split("<SEP>")).explode('hits')

    # Rename columns to match other datasets:
    one_publ = one_publ.rename(columns = {"hits": "targetExpression", "prev_sentence": "prevSentence",
                                          "target_sentence": "currentSentence", "next_sentence": "nextSentence",
                                          "previous_sentence": "prevSentence"})

    # Mark and mask the target expression in the current sentence:
    one_publ[["markedSentence", "maskedSentence"]] = one_publ.apply(lambda x: pd.Series(mask_target(x.targetExpression, x.currentSentence)), axis=1)

    # Reorder the columns:
    one_publ = one_publ[['item_code', 'issue_code', 'publication_code', 'prevSentence',
                         'currentSentence', 'markedSentence', 'maskedSentence', 'nextSentence',
                         'targetExpression', 'article_path']]

    # Merge metadata into the sentences dataframe:
    one_publ = one_publ.merge(hmd_metadata_df, how="left", left_on=["publication_code", "issue_code", "item_code"], right_on=["publication_code", "issue_code", "item_code"])
    
    one_publ['year'] = one_publ['issue_date'].apply(lambda x: x.split("-")[0]).astype(int)
    
    # Drop idential sentences and expressions, while keeping the sentence from the oldest issue:
    one_publ = one_publ.sort_values(by='issue_date')
    one_publ = one_publ.drop_duplicates(subset=["issue_code", "publication_code", "currentSentence", "targetExpression"])

    # Filter by targetExpression/query:
    one_publ = one_publ[one_publ["targetExpression"].str.contains(query_tokens, flags=re.IGNORECASE)]
    
    return one_publ
