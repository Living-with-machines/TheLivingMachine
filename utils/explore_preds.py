from transformers import pipeline
import numpy as np

def create_mask_pipeline(epoch, pred_toks):

    path2model = "khosseini/bert_" + epoch
    if epoch == "contemporary":
        path2model = "bert-base-uncased"
        model_rd = pipeline("fill-mask",
                            model=path2model,
                            top_k=pred_toks)
    else:
        tokenizer = "bert-base-uncased"
        # Create a pipeline, see transformers repo for details
        model_rd = pipeline("fill-mask",
                         model=path2model,
                         tokenizer=tokenizer,
                         top_k=pred_toks)
    return model_rd

def bert_masking(row, model_rd):
    masked = []
    
    prevSentence = ""
    nextSentence = ""
    if type(row["prevSentence"]) == str:
        prevSentence = row["prevSentence"]
    if type(row["nextSentence"]) == str:
        nextSentence = row["nextSentence"]
    to_predict = prevSentence + " " + row["maskedSentence"] + " " + nextSentence
    if len(to_predict) >= 512:
        to_predict = prevSentence + " " + row["maskedSentence"]
    if len(to_predict) >= 512:
        to_predict = row["maskedSentence"]

    output = model_rd(to_predict)
    for ind_output in output:
        masked.append((ind_output["token_str"], round(ind_output["score"], 2)))
    return masked

def w2v_avg_embedding(ltokens, emb_model):
    doc_embed = []
    
    for word in ltokens:
        try:
            embed_word = emb_model.wv.get_vector(word)
            doc_embed.append(embed_word)
        except KeyError:
            pass
                
    if len(doc_embed)>0:
        avg = [float(sum(col))/len(col) for col in zip(*doc_embed)]

        avg = np.array(avg)

        return avg
    else:
        return np.zeros(emb_model.vector_size)
