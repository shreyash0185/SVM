#Import required Libraries 

import torch 
import numpy as np 
import os 
import random 
import pandas as pd 
from tqdm.notebook import tqdm 
from sklearn.model_selection import train_test_split 
from transformers import BertTokenizer 
from torch.utils.data import TensorDataset 
from transformers import BertForSequenceClassification 
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler 
from sklearn.metrics import f1_score 

#Loading data from Google drive 
from google.colab import drive 
drive.mount('/content/drive') 


os.chdir("ENTER LOCATION WHERE DATASET IS.") # EXAMPLE: /content/drive/My Drive/Sentiment_analysis_using_BERT 

df = pd.read_csv("smileannotationsfinal.csv", names = ['id', 'text', 'category'])

df.set_index('id', inplace = True) df.head()

df = df[~df.category.str.contains('\|')] # for removing values containing '|'

df = df[df.category != 'nocode']  #for removing nocode

df.category.value_counts()

possible_labels = df.category.unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index

df['label'] = df.category.replace(label_dict)
df.head()

X_train, X_val, y_train, y_val = train_test_split(df.index.values, df.label.values, test_size=0.15, random_state=17, stratify=df.label.values) 

df['data_type'] = ['not_set']*df.shape[0]   #CREATING A NEW COLUMN IN DATASET AND SETTING ALL VALUES TO 'not_set' 

df.loc[X_train, 'data_type'] ='train' #CHECKING AND SETTING data_type TO TRAIN 
df.loc[X_val, 'data_type'] = 'val' #CHECKING AND SETTING data_type TO VAL


df.groupby(['category', 'label', 'data_type']).count() #TO CHECK WHICH CATEGORY DATA IS IN WHICH data_type

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

#ENCODING DATA
encoded_data_train = tokenizer.batch_encode_plus(df[df.data_type=='train'].text.values,
                                                add_special_tokens=True,
                                                return_attention_mask=True,
                                                pad_to_max_length=True,
                                                max_length=256,
                                                 return_tensors='pt'
                                                )
encoded_data_val = tokenizer.batch_encode_plus(df[df.data_type=='val'].text.values,
                                                add_special_tokens=True,
                                                return_attention_mask=True,
                                                pad_to_max_length=True,
                                                max_length=256,
                                                 return_tensors='pt'
                                                )

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].label.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)

dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

#SETTING MODEL
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6, output_attentions=False, output_hidden_states=False )

#CREATING DATA LOADERS
dataloader_train = DataLoader(dataset_train,sampler = RandomSampler(dataset_train), batch_size= 32)
                             

dataloader_val = DataLoader(dataset_val, sampler = RandomSampler(dataset_val), batch_size= 32)
      

#SETTING OPTIMIZERS

op = AdamW(model.parameters(),lr=1e-5,eps=1e-8)

epochs = 10

scheduler = get_linear_schedule_with_warmup(op, num_warmup_steps=0, num_training_steps=len(dataloader_train)*epochs)

#FUNCTION TO CALCULATE F1 SCORE
def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds,axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average = 'weighted')

#FUNCTION FOR CALCULATING ACCURACY PER CLASS
def accuracy_per_class(preds, labels):
    label_dict_inverse = {v:k for k,v in label_dict.items()}
    
    preds_flat = np.argmax(preds,axis=1).flatten()
    labels_flat = labels.flatten()
    
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat==label]
        y_true = labels_flat[labels_flat==label]
        print(label_dict_inverse[label])
        print("accuracy ", len(y_preds[y_preds==label])/len(y_true))

#FUNCTION FOR MODEL EVALUATION
def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    
    loss_val_avg = loss_val_total/len(dataloader_val) 
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals

for epoch in tqdm(range(1, epochs+1)):
    model.train()
    
    loss_train_total = 0
    
    progress_bar = tqdm(dataloader_train, 
                        desc ='Epoch {:1d}'.format(epoch),
                        leave=False,
                       disable=False
                       )
    for batch in progress_bar:
        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = { 'input_ids' : batch[0],
                 'attention_mask' : batch[1],
                 'labels' : batch[2]
                 }
        outputs = model(**inputs)
        
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss':'{:.3f}'.format(loss.item()/len(batch))})
    
    # THIS SECTION OF CODE IS JUST FOR PRINTING VALUES AFTER EACH EPOCH.
    torch.save(model.state_dict(), f'BERT_ft_epoch{epoch}.model')
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_val)
    val_f1 = f1_score_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 score (weighted): {val_f1}')  

_, predictions, true_val = evaluate(dataloader_val)  #why _ ? reason behind this is evaluate function return 3 values and i don't require the 1st value i.e., loss_val_avg

accuracy_per_class(predictions, true_val)