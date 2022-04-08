import os 
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm, trange
import optim
import data


def fit(bert_model, train_dataloader, val_dataloader, args):
    logger = args['logger']
    num_epochs = args['num_train_epochs']
    warmup = args['warmup_proportion'] > 0
    optimizer, scheduler = optim.prepare_optimizer(bert_model, args['learning_rate'], args['warmup_proportion'], args['num_train_steps'])
    
    global_step = 0
    for i_ in trange(int(num_epochs), desc="Epoch"):
        bert_model.train()
        
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        pbar = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(pbar):
            batch = tuple(t.to(args['device']) for t in batch)
            loss = bert_model(*batch)
            
            loss.backward()
            
            pbar.set_postfix({'trailing loss': loss.item()})
            tr_loss += loss.item()
            
            scheduler.batch_step()
            if warmup:
                lr_this_step = args['learning_rate'] * optim.warmup_linear(global_step / args['num_train_steps'], 
                                                                           args['warmup_proportion'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            
            nb_tr_examples += batch[0].size(0)
            nb_tr_steps += 1
            global_step += 1
            del batch
            if args['eval_steps'] > 0 and (step + 1) % args['eval_steps'] == 0:
                logger.info("Partial evaluation after epoch %d", i_ + 1)
                evaluate(bert_model, val_dataloader, args, steps=50)
                bert_model.train()

        # Save a trained model
        logger.info("Saving model after epoch %d", i_ + 1)
        output_model_file = Path(args['output_dir'], f"epoch_{i_}.bin")
        torch.save(bert_model.state_dict(), output_model_file)
        
        logger.info("Model evaluation after epoch %d", i_ + 1)
        evaluate(bert_model, val_dataloader, args)


def evaluate(bert_model, val_dataloader, args, steps=-1):
    logger = args['logger']
    
    bert_model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in tqdm(val_dataloader):
        batch = tuple(t.to(args['device']) for t in batch)
        
        with torch.no_grad():
            tmp_eval_loss = bert_model(*batch)
            logits = bert_model(*batch[:3])

        logits = logits.detach().cpu().numpy()
        label_ids = batch[-1].to('cpu').numpy()

        eval_loss += tmp_eval_loss.mean().item()

        tmp_eval_accuracy = accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += batch[0].size(0)
        nb_eval_steps += 1
        
        if nb_eval_steps == steps:
            break

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    
    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy}

    output_eval_file = os.path.join(args['output_dir'], "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return result


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def predict(bert_model, filename, processor, tokenizer, args):
    test_examples = processor.get_test_examples(args['data_dir'], filename=filename)
    test_features = data.convert_examples_to_features(test_examples, args['label_list'], args['max_seq_length'], tokenizer)
    test_dataloader = data.prepare_dataloader(test_features, args['batch_size'], test=True)
    
    all_logits = None
    all_raw_logits = None
    all_embeddings = None
    
    bert_model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for step, batch in enumerate(tqdm(test_dataloader, desc="Prediction Iteration")):
        batch = tuple(t.to(args['device']) for t in batch)
        
        with torch.no_grad():
            raw_logits = bert_model(*batch[:3])
            logits = F.softmax(raw_logits, -1)
            embeddings = bert_model.get_embeddings(*batch[:3])
            
        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

        if all_raw_logits is None:
            all_raw_logits = raw_logits.detach().cpu().numpy()
        else:
            all_raw_logits = np.concatenate((all_raw_logits, raw_logits.detach().cpu().numpy()), axis=0)

        if all_embeddings is None:
            all_embeddings = embeddings.detach().cpu().numpy()
        else:
            all_embeddings = np.concatenate((all_embeddings, embeddings.detach().cpu().numpy()), axis=0)

        nb_eval_examples += batch[0].size(0)
        nb_eval_steps += 1
        
    return pd.DataFrame(all_logits, columns=args['label_list']), all_raw_logits, all_embeddings