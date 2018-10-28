import torch
import torch.nn as nn

from tqdm import tqdm
import sys
from allennlp.commands.elmo import ElmoEmbedder

print("initializing ELMo embedding... ", file=sys.stderr)
if torch.cuda.is_available():
    elmo = ElmoEmbedder(cuda_device=0)
else:
    elmo = ElmoEmbedder()
print("loaded. ", file=sys.stderr)

def preprocess_sentence(sentence, speech_tag_idx):
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
    training_tuples = []
    for sentence in tqdm(train_data):
        preprocessed_sentence, preprocessed_speech_tag = preprocess_sentence(sentence, speech_tag_idx)

        preprocessed_tag = preprocess_target(sentence, tag2idx)
        training_tuples.append((preprocessed_sentence, preprocessed_speech_tag, preprocessed_tag))

    return training_tuples

def prepare_test_data(test_dataset, speech_tag_idx):

    return [preprocess_sentence(sentence, speech_tag_idx) for sentence in tqdm(test_dataset)]

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

def predict_seq(model, input_seq):
    with torch.no_grad():
        output = model(input_seq[0], input_seq[1])
        return output.max(1)[1]

def decode_seq(predicted_seq):
    if use_gpu:
        predicted_seq = predicted_seq.cpu()
    return [reversed_tag_index[idx] for idx in predicted_seq.numpy()]

def validate_model(model, validation_pairs):
    loss_function = nn.NLLLoss()
    error = 0.0
    with torch.no_grad():
        for input_seq, speech_tag, target_seq in validation_pairs:
            output = model(input_seq, speech_tag)
            loss = loss_function(output, target_seq)
            error += loss.item()
    return error

def extract_model_data(model_data):
    global word_idx
    global speech_tag_idx
    global target_tag_idx
    global reversed_tag_index

    # test_mode = True
    word_idx = model_data['word_index']
    speech_tag_idx = model_data['speech_tag_index']
    target_tag_idx = model_data['tag_index']
    reversed_tag_index = model_data['reverse_tag_index']

    model = BiLSTM(len(word_idx), len(speech_tag_idx), len(target_tag_idx))

    model.load_state_dict(model_data['model'])

    return model

def test_all(model_data, test_dataset, tag_set):
    model = extract_model_data(model_data)
    test_data = prepare_test_data(test_dataset)

    predicted_tag_sequences = []
    for input_seq in tqdm(test_data):
        output = predict_seq(model, input_seq)
        decoded_tags = decode_seq(output)
        predicted_tag_sequences.append(decoded_tags)

    return predicted_tag_sequences

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

def log_sum_exp(score):
    maxscore = torch.max(score, -1)[0] # [C]
    return maxscore + torch.log(torch.sum(torch.exp(score - maxscore.unsqueeze(-1)), -1))