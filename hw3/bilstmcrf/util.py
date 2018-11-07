import torch
import torch.nn as nn
import sys
sys.path.append('../')
import perc
from score_chunks import readTestFile, corpus_fmeasure

import os.path as osp
from tqdm import tqdm
import sys
from allennlp.commands.elmo import ElmoEmbedder


def preprocess_sentence(sentence, speech_tag_idx, elmo):
    # temporarily ignore features
    features = sentence[1]
    sentence = sentence[0]


    this_sentence = []
    this_speech_tags = []

    for word in sentence:
        word_info = word.split()
        word = word_info[0].lower()
        speech_tag = word_info[1]

        this_sentence.append(word)
        this_speech_tags.append(speech_tag)

    this_sentence = elmo.embed_sentence(this_sentence)[2]
    this_sentence = torch.from_numpy(this_sentence)

    this_speech_tags = prepare_sequence(this_speech_tags, speech_tag_idx)
    return this_sentence, this_speech_tags

def preprocess_target(sentence, tag2idx):
    sentence = sentence[0]
    target_tags = [word.split()[2] for word in sentence]
    return prepare_sequence(target_tags, tag2idx)
    # return target_tags

def build_vocab(train_data):
    word_idx = {'<UNKNOWN>': 0}
    speech_tag_idx = {'<UNKNOWN>': 0}

    for sentence in train_data:
        sentence = sentence[0]
        for word_info in sentence:
            word_info = word_info.split()
            word = word_info[0].lower()
            speech_tag = word_info[1]
            if word not in word_idx:
                word_idx[word] = len(word_idx)
            if speech_tag not in speech_tag_idx:
                speech_tag_idx[speech_tag] = len(speech_tag_idx)

    return word_idx, speech_tag_idx

def prepare_training_data(train_data, speech_tag_idx, tag2idx):
    print("initializing ELMo embedding... ", file=sys.stderr)
    if torch.cuda.is_available():
        elmo = ElmoEmbedder(cuda_device=0)
    else:
        elmo = ElmoEmbedder()
    print("loaded. ", file=sys.stderr)
    training_tuples = []
    for sentence in tqdm(train_data):
        preprocessed_sentence, preprocessed_speech_tag = preprocess_sentence(sentence, speech_tag_idx, elmo)

        preprocessed_tag = preprocess_target(sentence, tag2idx)
        training_tuples.append((preprocessed_sentence, preprocessed_speech_tag, preprocessed_tag))

    return training_tuples

def prepare_test_data(test_dataset, speech_tag_idx):
    print("initializing ELMo embedding... ", file=sys.stderr)
    if torch.cuda.is_available():
        elmo = ElmoEmbedder(cuda_device=0)
    else:
        elmo = ElmoEmbedder()
    print("loaded. ", file=sys.stderr)

    return [preprocess_sentence(sentence, speech_tag_idx, elmo) for sentence in tqdm(test_dataset)]

def build_tag_index(tag_set):
    target_tag_idx = {}
    reversed_tag_index = {}
    target_tag_idx['<start>'] = 0
    target_tag_idx['<end>'] = 1
    reversed_tag_index[0] = '<start>'
    reversed_tag_index[1] = '<end>'
    for tag in tag_set:
        target_tag_idx[tag] = len(target_tag_idx)
        reversed_tag_index[target_tag_idx[tag]] = tag

    return target_tag_idx, reversed_tag_index

def prepare_sequence(seq, index_set):
    indices = []
    # use -1 for OOV words.
    for symbol in seq:
        if symbol in index_set:
            indices.append(index_set[symbol])
        else:
            indices.append(index_set['<UNKNOWN>'])

    return torch.tensor(indices, dtype=torch.long)

def dump_model(model, word_idx, speech_tag_idx, target_tag_idx, reversed_tag_index, file):
    checkpoint = {
        'model': model,
        'word_index': word_idx,
        'speech_tag_index': speech_tag_idx,
        'tag_index': target_tag_idx,
        'reverse_tag_index': reversed_tag_index
    }
    torch.save(checkpoint, file)

def load_model(file):
    # return torch.load(file)
    return torch.load(file, map_location=lambda storage, loc: storage)

def test_model(model, data_tuples, idx2tag, device):
    predicted_tags = []
    with torch.no_grad():
        for input_seq, input_tag in data_tuples:
            input_seq = input_seq.to(device)
            input_tag = input_tag.to(device)
            tag_indices = model.decode(input_seq, input_tag)
            predicted_tags.append([idx2tag[tag_idx.cpu().item()] for tag_idx in tag_indices])

    return predicted_tags

def format_prediction(predicted_tags, test_data):
    output = ''
    for idx, _ in enumerate(predicted_tags):
        output += ("\n".join(perc.conll_format(predicted_tags[idx], test_data[idx][0])))+'\n\n\n'
    return output

def compute_score(output, ref_file):
    boundary = '-X-'
    outside = 'O'
    test, _ = readTestFile(output, boundary, outside, False, 2)
    with open(ref_file) as f:
        reference, _ = readTestFile(f.read(), boundary, outside, False, 2)
    return corpus_fmeasure(reference, test, False)

def log_sum_exp(score):
    maxscore = torch.max(score, -1)[0] # [C]
    return maxscore + torch.log(torch.sum(torch.exp(score - maxscore.unsqueeze(-1)), -1))