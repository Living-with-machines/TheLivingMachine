from transformers import pipeline
import numpy as np

def create_mask_pipeline(epoch, pred_toks):

	path2model = "./models/bert/bert_" + epoch
	tokenizer = "bert-base-uncased"

	# Create a pipeline, see transformers repo for details
	model_rd = pipeline("fill-mask",
	                     model=path2model,
	                     tokenizer=tokenizer,
	                     top_k=pred_toks)
	return model_rd

def bert_masking(row, model_rd):
	masked = []
	output = model_rd(row["maskedSentence"])
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