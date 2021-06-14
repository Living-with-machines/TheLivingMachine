import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import glob


def parse_content(xml):
	# ----- Code by Giorgia Tolfo ----- #

    tree = ET.parse(xml)
    root = tree.getroot()
    body = root.find('.//{http://www.tei-c.org/ns/1.0}text')
    sentences = []
    for s in tree.findall('.//{http://www.tei-c.org/ns/1.0}s'):
        sentence = ' '.join(''.join(s.itertext()).split())
        sentences.append(sentence)

    return sentences


def parse_corpus(input_path, output_path, overwrite):

    if not Path(output_path + "RSC_content").is_dir() or overwrite == True:

        Path(output_path + "RSC_content").mkdir(parents=True, exist_ok=True)

        metadata = input_path + "Royal_Society_Corpus_open_v6.0_meta.tsv"
        mddf = pd.read_csv(metadata, sep="\t")

        mddf.to_csv(output_path + "RSC_metadata.tsv", sep="\t", index=False)

        for xml in glob.glob(input_path + 'Royal_Society_Corpus_open_v6.0_texts_tei/*.xml'):
            sentences = parse_content(xml)

            formatted_filepath = xml.split("text_")[-1].split(".")[0] + ".txt"
            
            with open(output_path + "RSC_content/" + formatted_filepath, 'w') as fw:
                for sentence in sentences:
                    fw.write(sentence + "\n")