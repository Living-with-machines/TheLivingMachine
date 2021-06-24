import re
import glob
import datetime
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from sentence_splitter import split_text_into_sentences
from tqdm import tqdm
tqdm.pandas()


# ===================================================
# Parse metadata
# ===================================================

def parse_metadata(input_path):

    rows = []

    re_jsa_id = "j[0-9]+"
    for i in glob.glob(input_path + "/metadata/*"):
        
        filename = i.split("/")[-1].split(".xml")[0]
        
        # journal metadata
        journal_id = ""
        journal_id_text = ""
        journal_title = ""
        publisher_name = ""
        
        # article metadata
        volume = ""
        issue = ""
        issue_id = ""
        article_id = ""
        first_page = ""
        last_page = ""
        uri = ""
        title_group = ""
        date = ""
        contributors = []
        categories = []
        year = ""
        
        root = ET.parse(i).getroot()
        article_type = root.attrib["article-type"]
        for child in root.findall('front/journal-meta/'):
            if child.tag == "journal-id":
                if re.match(re_jsa_id, child.text):
                    journal_id = child.text
                else:
                    journal_id_text = child.text
            elif child.tag == "journal-title-group":
                journal_title = child.find("journal-title").text
            elif child.tag == "publisher":
                publisher_name = child.find("publisher-name").text
        
        for child in root.findall('front/article-meta/'):
            if child.tag == "volume":
                volume = child.text
            elif child.tag == "issue":
                issue = child.text
            elif child.tag == "issue-id":
                issue_id = child.text
            elif child.tag == "article-id":
                article_id = child.text
            elif child.tag == "fpage":
                first_page = child.text
            elif child.tag == "lpage":
                last_page = child.text
            elif child.tag == "self-uri":
                uri = child.attrib["{http://www.w3.org/1999/xlink}href"]
            elif child.tag == "title-group":
                title_group = child.find("article-title").text
            elif child.tag == "contrib-group":
                for subchild in child.findall("contrib/"):
                    contrib = dict()
                    if subchild.tag == "string-name":
                        for subsubchild in subchild:
                            contrib[subsubchild.tag] = subsubchild.text
                    else:
                        contrib[subchild.tag] = subchild.text
                    contributors.append(contrib)
            elif child.tag == "article-categories":
                for subchild in child.findall("subj-group/"):
                    categories.append(subchild.text)
            elif child.tag == "pub-date":
                day = int(child.find("day").text)
                month = int(child.find("month").text)
                year = int(child.find("year").text)
                date = datetime.date(year, month, day)
        rows.append([filename, journal_id, journal_id_text, journal_title, publisher_name,
                     volume, issue, issue_id, article_id, article_type, first_page, last_page, uri,
                     title_group, date, year, contributors, categories])

    df = pd.DataFrame(rows, columns=["filename", "journal_id", "journal_id_text", "journal_title", "publisher_name", "volume",
             "issue", "issue_id", "article_id", "article_type", "first_page", "last_page", "uri", "title_group",
             "date", "year", "contributors", "subjects"])

    return df


# ===================================================
# Parse the full text
# ===================================================

def parse_fulltext(fulltext_path, input_path):
    fulltext_path = input_path + "/ocr/" + fulltext_path + ".txt"

    sentences = []
    if Path(fulltext_path).exists():
        with open(fulltext_path) as fr:
            re_pageheader = r".*\, [A-Z][A-Za-z]+ [0-9]{1,2}\, [0-9]{4}\.? ?[0-9]*"
            fulltext = fr.read()
            fulltext = fulltext.split("<plain_text>")[1]
            fulltext = fulltext.split("</plain_text>")[0]
            fulltext = fulltext.split("</page>")
            fulltext = [x for x in fulltext if x]
            text = ""
            for pageseq in fulltext:
                pageseq_num = re.match("\<page sequence\=\"([0-9]+)\"\>", pageseq).group(1)
                pageseq = re.sub("\<page sequence\=\"([0-9]+)\"\>", "", pageseq).strip()
                pageseq = re.sub(re_pageheader, "", pageseq).strip()
                text += " " + pageseq
            sentences = split_text_into_sentences(text=text, language='en')
    return sentences


# ===================================================
# Main code
# ===================================================

def parse_corpus(input_path, output_path, overwrite):
    batches = ["JSA_before1854", "JSA_1854-1908"]

    if not Path(output_path + "JSA_content").is_dir() or overwrite == True:

        Path(output_path + "JSA_content").mkdir(parents=True, exist_ok=True)

        metadata_dfs = []
        content_dfs = []

        for jsa_batch in batches:
            input_path = "../../workspace/data/" + jsa_batch

            # Process metadata into a dataframe:
            metadata_tmp_df = parse_metadata(input_path)

            # Process the plain text:
            content_tmp_df = metadata_tmp_df.copy()
            
            content_tmp_df['sentences'] = content_tmp_df.progress_apply(lambda x: parse_fulltext(x["filename"], input_path),axis=1)

            metadata_dfs.append(metadata_tmp_df)

            # Store content:
            for i, row in content_tmp_df.iterrows():
                with open(output_path + "JSA_content/" + str(row["article_id"]) + ".txt", "w") as fw:
                    for sent in row["sentences"]:
                        fw.write(sent + "\n")


        # Store metadata:
        pd.concat(metadata_dfs).to_csv(output_path + "JSA_metadata.tsv", sep="\t", index=False)
