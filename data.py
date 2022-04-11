import logging
import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

        
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_test_examples(self, data_dir, data_file_name, size=-1):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError() 

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

        
class LabelTextProcessor(DataProcessor):
    def __init__(self, data_dir, labels=[]):
        self.data_dir = data_dir
        self.labels = labels
    
    def _get_examples(self, data_dir, filename, size=-1):
        if size == -1:
            data_df = pd.read_csv(os.path.join(data_dir, filename), index_col=0)
            return self._create_examples(data_df)
        else:
            data_df = pd.read_csv(os.path.join(data_dir, filename), index_col=0)
            return self._create_examples(data_df.sample(size))
    
    def get_train_examples(self, data_dir, filename='train.csv', size=-1):
        return self._get_examples(data_dir, filename, size)
        
    def get_dev_examples(self, data_dir, filename='val.csv', size=-1):
        return self._get_examples(data_dir, filename, size)
    
    def get_test_examples(self, data_dir, filename='test.csv', size=-1):
        return self._get_examples(data_dir, filename, size)

    def get_examples_from_df(self, df):
        return self._create_examples(df)

    def get_labels(self):
        return self.labels

    def _create_examples(self, df, labels_available=True):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, row) in df.iterrows():
            try:
                guid = row['id']
            except:
                guid = i

            text_a = row['text']
            try:
                labels = row['category']
            except KeyError:
                labels = 'OTHER'
#                 print("No Label Found")
            
            examples.append(
                InputExample(guid=guid, text_a=text_a, labels=labels))
        
        return examples
        

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        labels_ids = label_map[example.labels]

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=labels_ids))
    return features


def prepare_dataloader(features, bs, test=False):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    if test:
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=bs)
    return dataloader


def get_data_splits(df, train_size=.6, test_size=.2, random_seed=200):
    one_hot = pd.get_dummies(df['category'])
    df2 = df.join(one_hot)

    X = df2.sample(frac=1 - test_size, random_state=random_seed)
    test = df2.drop(X.index)

    train = X.sample(frac=(train_size / (1 - test_size)), random_state=random_seed)
    val = X.drop(train.index)
    
    return train, val, test
