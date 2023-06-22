# The Living Machine

This repository contains the code for the experiments used for the paper 'The Living Machine: A Computational Approach to the Nineteenth Century Language of Technology', currently in Press with _Technology and Culture_, by Daniel CS Wilson, Mariona Coll Ardanuy, Kaspar Beelen et al. and available Open Access from July 2023. Contact @dcsw2 for info.

### Abstract

The article examines a long-standing question in the history of technology concerning the trope of the living machine. We do this using a cutting-edge computational method, which we apply to large collections of digitized texts. In particular, we demonstrate the affordances of a neural language model for historical research. In a self-conscious maneuver, we use a type of model, often portrayed as sentient today, to detect figures of speech in nineteenth-century texts that portrayed machines as self-acting, automatic, or alive. Our method uses a masked language model to detect unusual or surprising turns of phrase, which could not be discovered using simple keyword search. We collect and close-read such sentences to explore how figurative language produced a context in which humans and machines were conceived as interchangeable in complicated ways. We conclude that, used judiciously, language models have the potential to open new avenues of historical research. 

All of our experiments can be reproduced using the code below, which is presented in Python and Jupyter Notebooks. The same methods can also be used on different text datasets, or with different models.

## Installation

- Clone this Repository to your local machine with one of the methods using the `code` button, above.
- Navigate to this directory in your Terminal using `cd` TheLivingMachine

- Ensure you have Anaconda installed (https://docs.anaconda.com/anaconda/install/)

Then, run the following commands:
- To create an environment `conda create -n py39tlm python=3.9`

- Activate your environment with conda
`conda activate py39tlm`

- To install the requirements
 `pip install -r requirements.txt`
 
- To download the Spacy pipeline we use
`python -m spacy download en_core_web_sm`

- To enable the kernel on Jupyter Notebooks
  `python -m ipykernel install --user --name py39tlm --display-name "Python (py39tlm)"`

- You should then be able to run the Jupyter Notebooks provided in the repository using `jupyter notebook`.
 

## Code

Python script `prepare_datasets.py`:
1. Processes the different datasets (HMD, JSA, BLB).
2. Gets all sentences which contain certain query tokens (machine, machines, etc), in the date range 1783-1908. The output is a `.tsv` file with one sentence per row (and relevant metadata in the other columns), stored as `data/xxx_processed/XXX_machine.tsv`.
3. Parses the sentences, finds the syntactic role of the query token, and filters out sentences which do not look like a sentence (e.g. no verb). The output is `data/xxx_processed/XXX_synparsed.pkl`, which has the same structure as the `.tsv` from step 2, with one additional column, which has information on the syntactic role of the query token in the sentence.
4. Using BERT to predict the most likely word instead of the query word. The output is `data/xxx_processed/XXX_synparsed_pred_bert.pkl`, which is the dataframe from step 3, with two additional columns: one for each model (e.g., limited by date range), with predictions given by the respective models.

Notebook `explore_by_concept.ipynb` uses aggregated predictions produced by the models to explore the affinity (using word2vec) between the masked expression and one of a number of concepts of historical interest.

Notebook `explore_data.ipynb` provides stats about the datasets more generally (number of sentences per dataset, etc).
