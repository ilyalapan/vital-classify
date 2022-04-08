# news_classifier

# Quickstart

## Requirements

All the library versions are up-to-date with Google Colab (by April 2022). 

Additional 

## Pretrained models

Models are stored in (Google Drive)[https://drive.google.com/drive/folders/14zBJwsJvjo0eGZfnvJoBUcUKc1huv5eE?usp=sharing]. Recommend using `finetuned_pytorch_model_32_ep5.bin` or `finetuned_pytorch_model_32_ep7.bin`. Epoch-3 version could be used as a checkpoint for fine-tuning the model given num_classes fixed.

The model performs considerably well on the dataset used for training ((Kaggle dataset)[https://www.kaggle.com/rmisra/news-category-dataset])
- 89% accuracy on validation set
- 87.9% accuracy on test set

## Inference 

`bert_classification_model_inference.ipynb` presents the inference code. By default, `evals.predict()` function returns 
- Class probabilities
- Logits (softmax of logits are probabilities)
- 768-dimensional text embeddings

Example dataset is attached at `data/new_test.csv`

Predictions are performed on a sequence of 32 first words in the `text` column of a given dataset.

The notebook contains comments on further details.

## Training

`bert_classification_model_training.ipynb` presents the training code. 

Training dataset is attached at `data/cleaned.csv`. Training for 1 epoch takes 20-30 minutes on Google Colab GPU depending on training set size. 5-7 epochs is enough for achieving decent quality (85+% accuracy on validation). 

Training is performed using a sequence of 32 first words in the `text` column of a given dataset.

The notebook contains comments on further details.
