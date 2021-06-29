from langdetect import detect
import pandas as pd
import glob
import re


def process_content(query_tokens, txtf):
    with open(txtf) as fr:
        lines = fr.readlines()
        lines = [line.replace("\n", "").strip() for line in lines]
        lSentenceTuples = []
        for iline in range(len(lines)):
            prevSentence = ""
            nextSentence = ""
            if iline > 0:
                prevSentence = lines[iline - 1]
            if iline < len(lines) - 1:
                nextSentence = lines[iline + 1]
            currentSentence = lines[iline]
            
            matches = re.findall(r"(?=("+'|'.join(query_tokens)+r"))", currentSentence, re.IGNORECASE)
            matches = list(set(matches))
            
            for match in matches:
                targetExpression = match
                markedSentence = re.sub(r"\b%s\b" % match, "$***" + match + "***$", currentSentence, 1, flags=re.IGNORECASE)
                maskedSentence = re.sub(r"\b%s\b" % match, " [MASK] ", currentSentence, 1, flags=re.IGNORECASE)
                maskedSentence = re.sub(" +", " ", maskedSentence)
                    
                lSentenceTuples.append((prevSentence, currentSentence, markedSentence, maskedSentence, nextSentence, targetExpression))
    
    return lSentenceTuples


def filter_sents_query(corpus, query):
    
    query_tokens = dict()
    query_tokens["machine"] = ["machine", "machines"]
    query_tokens["engine"] = ["engine", "engines"]

    metadata_fields = []
    metadata_file = ""

    if corpus == "JSA":
        metadata_fields = ["filename", "journal_id", "journal_title", "publisher_name", "volume", "issue_id", "article_type", "title_group", "date", "year", "contributors", "subjects"]
        metadata_file = pd.read_csv("data/" + corpus.lower() + "_processed/" + corpus + "_metadata.tsv", sep="\t", index_col="article_id")
        metadata_file.index = metadata_file.index.astype(str)
    
    if corpus == "RSC":
        metadata_fields = ["issn", "title", "year", "volume", "journal", "author", "type", "language", "primaryTopic", "secondaryTopic"]
        metadata_file = pd.read_csv("data/rsc_processed/RSC_metadata.tsv", sep="\t", index_col="id")

    query_tokens = [r"\b" + x + r"\b" for x in query_tokens[query]]

    df_cols = metadata_fields + ["prevSentence", "currentSentence", "markedSentence", "maskedSentence", "nextSentence", "targetExpression"]
    df_rows = []

    for txtf in glob.glob("data/" + corpus.lower() + "_processed/" + corpus + "_content/*"):
        fileid = txtf.split(".txt")[0].split("/")[-1]
    
        filemtdt = metadata_file.loc[fileid][metadata_fields]
    
        for t in process_content(query_tokens, txtf):
            prevSentence, currentSentence, markedSentence, maskedSentence, nextSentence, targetExpression = t
            df_rows.append(list(filemtdt) + [prevSentence, currentSentence, markedSentence, maskedSentence, nextSentence, targetExpression])

    df_sentences = pd.DataFrame(df_rows, columns=df_cols)

    return df_sentences


# ===============================
# Syntactic parsing and filtering

def remove_punctspaces(text):
    # Remove spaces before punctuation:
    return re.sub(r'\s([?.,;!](?:\s|$))', r'\1', text)

def parse_pipe(doc):
    lemma_list = [(
        tok.text, # Text form
        tok.pos_, # POS tag
        tok.dep_, # Dep label
        tok.head.text, # Head of dependency
    ) for tok in doc]
    
    return lemma_list


def preprocess_pipe(texts, nlp):
    preproc_pipe = []
    for doc in nlp.pipe(texts, batch_size=20):
        preproc_pipe.append(parse_pipe(doc))
    return preproc_pipe


def filter_sents_synt(processed, maskedSent, currentSent, query):

    # Because we're not always working with clean data, some times the
    # sentences are not correctly parsed. We minimize this through
    # the following steps, which we apply to all datasets.
    
    # Filter out sentences not identified as being in English: 
    if detect(currentSent) != "en":
        return False

    # Our BERT is only able to process 512 characters. Any sentence
    # longer than that is filtered out:
    if len(maskedSent) >= 512:
        return False

    query_exists = False
    verb_is_root = False

    for t in processed:

        # The query is identifiable:
        if t[0] == query:
            query_exists = True

        # If the query is not a noun, filter out: ### ADAPT IF THIS CHANGES
        if t[0] == query and t[1] != "NOUN":
            return False

        # The query is identifiable:
        if (t[1] == "VERB" or t[1] == "AUX") and t[2] == "ROOT":
            verb_is_root = True

        # If the query is the root, filter out:
        if t[2] == "ROOT" and t[0] == query:
            return False

    # If the query is identifiable, filter in:
    if query_exists == True and verb_is_root == True:
        return True

    # Otherwise, filter out:
    return False

def find_query_deplabel(processed, masked, query):

    # There may be more than one match of the query word, make sure
    # we're taking the one we've masked!
    premasked = masked.split("[MASK]")[0]
    notQueries = len(re.findall(r'\b' + query + r'\b', premasked))

    for t in processed:
        if t[0] == query:
            if notQueries == 0:
                return t[2]
            else:
                notQueries -= 1
