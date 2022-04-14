import os
import json
from pathlib import Path

# Third-party packages
import numpy as np
import pandas as pd
from pytorch_pretrained_bert.tokenization import BertTokenizer
import torch
import torch.nn.functional as F

# Custom imports
import model
import data
import optim


# Model parameters setup
DATA_PATH = Path('data/')
MODEL_PATH = Path('model/')

args = {
    "data_dir": DATA_PATH,
    "output_dir": MODEL_PATH,
    
    "task_name": "news_cat_label",
    "bert_model": 'bert-base-uncased',  # оптимальная моделька с точки зрения веса, вряд ли стоит менять
    "do_lower_case": True,
    
    "max_seq_length": 32, # Сколько слов из колонки 'text' используется в модели
    "batch_size": 128, # размер батча на входе в модель

    "no_cuda": False, # Если есть cuda, но почему-то не хочешь юзать
    "seed": 42,
}

args['device'] = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")

args['label_list'] = ['ENTERTAINMENT', 'WORLD NEWS', 'OTHER', 'POLITICS', 'WOMEN', 'SPORTS',
       'BUSINESS', 'TRAVEL', 'TECH', 'RELIGION', 'SCIENCE', 'ARTS', 'STYLE',
       'ENVIRONMENT', 'FOOD & DRINK', 'HEALTHY LIVING', 'HOME & LIVING',
       'MONEY']
args['num_labels'] = len(args['label_list'])


# Model
fname = "finetuned_pytorch_model_32_ep5.bin" 
output_model_file = os.path.join(args['output_dir'], fname)

bert_model = model.get_model(model_path=output_model_file, 
                             bert_model=args['bert_model'], 
                             num_labels=args['num_labels'])
bert_model.to(args['device'])


# Text preprocessors
processor = data.LabelTextProcessor(args['data_dir'], labels=args['label_list'])
tokenizer = BertTokenizer.from_pretrained(args['bert_model'], do_lower_case=args['do_lower_case'])


# Predictor itself
def predict_json(query):
    inp_data = pd.DataFrame(json.loads(query))
    inp_data['text'] = inp_data['title'] + ' ' + inp_data['text']
    test_examples = processor.get_examples_from_df(inp_data)
    test_features = data.convert_examples_to_features(test_examples, args['label_list'], args['max_seq_length'], tokenizer)
    test_dataloader = data.prepare_dataloader(test_features, args['batch_size'], test=True)
    
    all_logits = None
    
    bert_model.eval()
    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(args['device']) for t in batch)
        
        with torch.no_grad():
            raw_logits = bert_model(*batch[:3])
            logits = F.softmax(raw_logits, -1)
            
        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
    
    res = pd.DataFrame(all_logits, columns=args['label_list'])
    res = pd.concat([inp_data[['id']], res], axis=1).to_json(orient='records')
    return res