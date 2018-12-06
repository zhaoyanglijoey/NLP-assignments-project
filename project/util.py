from tqdm import tqdm
import os
import torch

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id

def convert_data_to_features(data, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for ex_index, label, tweet in tqdm(data.itertuples()):
        tokens = tokenizer.tokenize(str(tweet))

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]

        tokens.insert(0, "[CLS]")
        tokens.append("[SEP]")

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        label_id = label_map[label]
        # if ex_index < 5:
        #     print("*** Example ***")
        #     print("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     print("label: %s (id = %d)" % (label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              label_id=label_id))
        ex_index += 1
    return features

def convert_to_bert_ids(seq, tokenizer, max_seq_len):
    tokens = tokenizer.tokenize(seq)
    if len(tokens) > max_seq_len - 2:
        tokens = tokens[0:(max_seq_len-2)]
    # length = len(tokens)
    tokens.insert(0, '[CLS]')
    tokens.append('[SEP]')
    ids = tokenizer.convert_tokens_to_ids(tokens)
    padded_ids = [0] * max_seq_len
    padded_ids[:len(ids)] = ids
    mask = [0] * max_seq_len
    mask[:len(ids)] = [1] * len(ids)

    # assert len(padded_ids) == max_seq_len
    # assert len(mask) == max_seq_len

    padded_ids = torch.tensor(padded_ids, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.long)

    return padded_ids, mask

def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)