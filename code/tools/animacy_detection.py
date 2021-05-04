import re
import unidecode
import torch
from tools import wordnet_utils, processing
from pathlib import Path
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from collections import Counter
pd.options.mode.chained_assignment = None


# ---------------------------------------------------
# BERT masking approach
def bert_masking(masked_sent,cutoff,model,tokenizer,wsd):
    
    model.eval()
    
    tokenized_text = tokenizer.tokenize(masked_sent)
    
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [0] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)

    masked_index = tokenized_text.index('[MASK]')

    predicted_index = torch.topk(predictions[0, masked_index], k=cutoff)

    predicted_index_score = predicted_index[0]
    predicted_index_tokindex = predicted_index[1]
    
    predicted_tokens = [[tokenizer.convert_ids_to_tokens([predicted_index_tokindex[x].item()])[0],predicted_index_score[x].item()] for x in range(len(predicted_index_score.tolist()))]
    
    return predicted_tokens


# ---------------------------------------------------
# Determine animacy of predicted tokens
def animate_prediction(predicted_tokens,sentence,wsd,tokenizer,wsdmodel):
    predicts = []
    scores = []

    for x in range(len(predicted_tokens)):
        predicted_token = predicted_tokens[x][0]
        predicted_token_score = predicted_tokens[x][1]
        try:
            wordnet_pred = wordnet_utils.wordnet_animacy(predicted_token,sentence,wsdmodel,tokenizer)
                
            predicts.append(wordnet_pred)
            scores.append(predicted_token_score)
            
        except IndexError:
            continue
            
    return predicts, scores


# ---------------------------------------------------
# Main mask prediction function
def predict_mask_animacy(results_path, dataset_df, context, words_cutoff, prediction_approach, wsd, wsdmodel, tokenizer, language_model, bert_models, time_period, overwrite=False):
    
    stored_df = Path(results_path)
    stored_df.parent.mkdir(parents=True, exist_ok=True)

    if stored_df.is_file() and overwrite==False:
        dataset_df = pd.read_pickle(stored_df)
    else:
        predicts_over_rows = []
        scores_over_rows = []
        tokens_over_rows = []

        # Select target sentence according to chosen "context":
        dataset_df['targetSentence'] = dataset_df.apply(lambda row: processing.determine_context("maskedSentence", row, context), axis=1)

        if prediction_approach == "bert_masking":
            if time_period in bert_models:
                language_model = bert_models[time_period]
            
        for i, row in tqdm(dataset_df.iterrows()):

            # If time sensitive, we select the language model that better fits the document date:
            if time_period == "timeSensitive":
                if "date" in row:
                    if int(row["date"]) <= 1850:
                        language_model = bert_models["before1850"]
                    elif int(row["date"]) > 1850 and int(row["date"]) <= 1875:
                        language_model = bert_models["from1850to1875"]
                    elif int(row["date"]) > 1875 and int(row["date"]) <= 1890:
                        language_model = bert_models["from1875to1890"]
                    elif int(row["date"]) > 1890 and int(row["date"]) <= 1900:
                        language_model = bert_models["from1890to1900"]

            preprocessed_sentence = row["targetSentence"]

            # Find predictions with their scores
    
            try:
                method_options = {'bert_masking':bert_masking}
                predicted_tokens = method_options[prediction_approach](preprocessed_sentence,words_cutoff,language_model,tokenizer,wsd)
                predicted, scores = animate_prediction(predicted_tokens,preprocessed_sentence,wsd,tokenizer,wsdmodel)
                tokens_alone = [tok[0] for tok in predicted_tokens]

                if predicted:
                    predicts_over_rows.append(predicted)
                    scores_over_rows.append(scores)
                    tokens_over_rows.append(tokens_alone)
                else:
                    predicts_over_rows.append([])
                    scores_over_rows.append([])
                    tokens_over_rows.append([])
            except IndexError:
                predicts_over_rows.append([])
                scores_over_rows.append([])
                tokens_over_rows.append([])

        dataset_df['predicted'] = predicts_over_rows
        dataset_df['scores'] = scores_over_rows
        dataset_df['predicted_tokens'] = tokens_over_rows
        dataset_df.to_pickle(stored_df)


# ---------------------------------------------------
# Find animacy score, given predictions
def animacy_score(predicted,scores,weighted,res_cutoff):

    tmp_pred = []
    tmp_scor = []
    for x in range(len(predicted)):
        if not predicted[x] == None: # Keep only 0-1 animacy values for the prediction
            tmp_pred.append(int(predicted[x]))
            tmp_scor.append(scores[x])
    predicted = tmp_pred[:res_cutoff]
    scores = tmp_scor[:res_cutoff]
    
    # Animate score per sentence
    animate = 0.0
    inanimate = 0.0
    
    for i in range(len(predicted)):
        if predicted[i] == 1:
            if weighted:
                animate += scores[i] # WEIGHTED
            else:
                animate += 1.0 # NOT WEIGHTED
        if predicted[i] == 0:
            if weighted:
                inanimate += scores[i] # WEIGHTED
            else:
                inanimate += 1.0 # NOT WEIGHTED
            
    if inanimate == 0.0 and animate == 0.0:
        animacy_score = 0.0
    else:
        animacy_score = animate / (animate + inanimate)    
        
    return round(animacy_score, 2)