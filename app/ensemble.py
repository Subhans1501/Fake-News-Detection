
import torch
import numpy as np
from transformers import (BertTokenizer,BertForSequenceClassification,RobertaTokenizer,RobertaForSequenceClassification,DistilBertTokenizer,DistilBertForSequenceClassification)
from typing import List, Dict, Tuple
MODEL_PATHS={"bert":r"D:\Study\Semester 5\ML-LAB\Final Project (Fake News Detection)\models\Model_BERT","roberta":r"D:\Study\Semester 5\ML-LAB\Final Project (Fake News Detection)\models\roberta_model","distilbert":r"D:\Study\Semester 5\ML-LAB\Final Project (Fake News Detection)\models\Model_DistilBERT",}
_LOADED={}
def _softmax(x:np.ndarray)->np.ndarray:
    e=np.exp(x-np.max(x,axis=-1,keepdims=True))
    return e/e.sum(axis=-1,keepdims=True)
def load_models(model_names:List[str]=["bert","roberta","distilbert"])->Dict[str,Tuple]:
    global _LOADED
    for name in model_names:
        if name in _LOADED:
            continue
        path=MODEL_PATHS.get(name)
        if name=="bert":
            tokenizer=BertTokenizer.from_pretrained(path)
            model=BertForSequenceClassification.from_pretrained(path)
        elif name=="roberta":
            tokenizer=RobertaTokenizer.from_pretrained(path)
            model=RobertaForSequenceClassification.from_pretrained(path)
        else:
            tokenizer=DistilBertTokenizer.from_pretrained(path)
            model=DistilBertForSequenceClassification.from_pretrained(path)
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        _LOADED[name]=(tokenizer,model,device)
    return _LOADED
def predict_single(texts:List[str],tokenizer,model,device,batch_size:int=16):
    all_probs=[]
    all_preds=[]
    with torch.no_grad():
        for i in range(0,len(texts),batch_size):
            batch=texts[i:i+batch_size]
            enc=tokenizer(batch,truncation=True,padding=True,max_length=512,return_tensors="pt").to(device)
            outputs=model(**enc)
            logits=outputs.logits.cpu().numpy()
            probs=_softmax(logits)
            preds=np.argmax(probs, axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
    return np.vstack(all_probs),np.concatenate(all_preds)
def ensemble_predict(text: str,models:Dict[str,Tuple],strategy:str="majority")->Dict:
    texts=[text]
    per_model={}
    for name,(tokenizer,model,device) in models.items():
        probs, preds=predict_single(texts,tokenizer,model,device,batch_size=1)
        per_model[name]={"pred":int(preds[0]),"probs":probs[0].tolist()}
    if strategy=="avg_proba":
        prob_arrays=np.array([per_model[n]["probs"] for n in per_model])
        avg_probs=prob_arrays.mean(axis=0)
        ensemble_pred=int(np.argmax(avg_probs))
        ensemble_confidence=float(np.max(avg_probs))
        method="avg_proba"
    else:
        votes=[per_model[n]["pred"] for n in per_model]
        values,counts = np.unique(votes,return_counts=True)
        if len(values)==1:
            ensemble_pred=int(values[0])
        else:
            max_count=counts.max()
            candidates=values[counts==max_count]
            if len(candidates)==1:
                ensemble_pred=int(candidates[0])
            else:
                prob_arrays=np.array([per_model[n]["probs"] for n in per_model])
                avg_probs=prob_arrays.mean(axis=0)
                ensemble_pred=int(np.argmax(avg_probs))
        prob_arrays=np.array([per_model[n]["probs"] for n in per_model])
        avg_probs=prob_arrays.mean(axis=0)
        ensemble_confidence=float(avg_probs[ensemble_pred])
        method="majority_vote"
    return {
        "per_model":per_model,
        "ensemble":{
            "pred":int(ensemble_pred),
            "confidence":ensemble_confidence,
            "method":method
        }
    }