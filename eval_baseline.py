import argparse
import logging
import os
import sys
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader,TensorDataset
from transformers import BertForMultipleChoice,AdamW,get_linear_schedule_with_warmup

#Fix the seed.
SEED=42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

#Set up a logger.
logging_fmt = "%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_dataset(input_dir,num_examples=-1,num_options=4):
    input_ids=torch.load(os.path.join(input_dir,"input_ids.pt"))
    attention_mask=torch.load(os.path.join(input_dir,"attention_mask.pt"))
    token_type_ids=torch.load(os.path.join(input_dir,"token_type_ids.pt"))
    labels=torch.load(os.path.join(input_dir,"labels.pt"))

    input_ids=input_ids[:,:num_options,:]
    attention_mask=attention_mask[:,:num_options,:]
    token_type_ids=token_type_ids[:,:num_options,:]

    if num_examples>0:
        input_ids=input_ids[:num_examples,:,:]
        attention_mask=attention_mask[:num_examples,:,:]
        token_type_ids=token_type_ids[:num_examples,:,:]
        labels=labels[:num_examples]

    return TensorDataset(input_ids,attention_mask,token_type_ids,labels)

def create_pooled_text_embeddings(bert_model,batch_inputs):
    bert_model.eval()
    num_options=batch_inputs["input_ids"].size(0)
    ret=torch.empty(num_options,768).to(device)

    for i in range(num_options):
        option_inputs={
            "input_ids":batch_inputs["input_ids"][i].unsqueeze(0),
            "attention_mask":batch_inputs["attention_mask"][i].unsqueeze(0),
            "token_type_ids":batch_inputs["token_type_ids"][i].unsqueeze(0),
        }

        with torch.no_grad():
            outputs=bert_model(**option_inputs)
            ret[i]=outputs[1][0]

    return ret

def create_text_embeddings(bert_model,batch_inputs):
    bert_model.eval()
    num_options=batch_inputs["input_ids"].size(0)
    ret=torch.empty(num_options,512,768).to(device)

    for i in range(num_options):
        option_inputs={
            "input_ids":batch_inputs["input_ids"][i].unsqueeze(0),
            "attention_mask":batch_inputs["attention_mask"][i].unsqueeze(0),
            "token_type_ids":batch_inputs["token_type_ids"][i].unsqueeze(0),
        }

        with torch.no_grad():
            outputs=bert_model(**option_inputs)
            ret[i]=outputs[0][0]

    return ret

def simple_accuracy(pred_labels, correct_labels):
    return (pred_labels == correct_labels).mean()

def evaluate(classifier_model,dataloader):
    classifier_model.eval()

    preds = None
    correct_labels = None
    for batch_idx,batch in tqdm(enumerate(dataloader),total=len(dataloader)):
        with torch.no_grad():
            batch = tuple(t for t in batch)
            batch_size=batch[0].size(0)
            bert_inputs = {
                "input_ids": batch[0].to(device),
                "attention_mask": batch[1].to(device),
                "token_type_ids": batch[2].to(device),
                "labels": batch[3].to(device)
            }

            classifier_outputs=classifier_model(**bert_inputs)
            logits=classifier_outputs[1]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                correct_labels = bert_inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                correct_labels = np.append(
                    correct_labels, bert_inputs["labels"].detach().cpu().numpy(), axis=0
                )

    pred_labels = np.argmax(preds, axis=1)
    accuracy = simple_accuracy(pred_labels, correct_labels)

    return pred_labels,correct_labels,accuracy

def main(test_input_dir,model_filepath,result_save_dir):
    #Create a dataloader.
    test_dataset=create_dataset(test_input_dir,num_examples=-1,num_options=20)
    test_dataloader=DataLoader(test_dataset,batch_size=4,shuffle=False,drop_last=True)

    #Create a classifier model.
    logger.info("Load model parameters from {}.".format(model_filepath))
    classifier_model=BertForMultipleChoice.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    classifier_model.to(device)

    parameters=None
    if torch.cuda.is_available():
        parameters=torch.load(model_filepath)
    else:
        parameters=torch.load(model_filepath,map_location=torch.device("cpu"))

    classifier_model.load_state_dict(parameters)

    #Create a directory to save the results in.
    os.makedirs(result_save_dir,exist_ok=True)

    logger.info("Start model evaluation.")
    pred_labels,correct_labels,accuracy=evaluate(classifier_model,test_dataloader)
    logger.info("Accuracy: {}".format(accuracy))

    #Save results as text files.
    res_filepath=os.path.join(result_save_dir,"result_eval.txt")
    labels_filepath=os.path.join(result_save_dir,"labels_eval.txt")

    with open(res_filepath,"w") as w:
        w.write("Accuracy: {}\n".format(accuracy))

    with open(labels_filepath,"w") as w:
        for pred_label,correct_label in zip(pred_labels,correct_labels):
            w.write("{} {}\n".format(pred_label,correct_label))

    logger.info("Finished model evaluation.")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="AIO")

    parser.add_argument("--test_input_dir",type=str,default="~/EncodedCache/Dev2")
    parser.add_argument("--model_filepath",type=str,default="./OutputDir/Baseline/checkpoint_1.pt")
    parser.add_argument("--result_save_dir",type=str,default="./OutputDir/Baseline")

    args=parser.parse_args()

    main(
        args.test_input_dir,
        args.model_filepath,
        args.result_save_dir
    )
