import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import perc

from tqdm import tqdm

embedding_dimension = 8
hidden_unit_dimension = 8

use_gpu = torch.cuda.is_available()

word_idx = {}
speech_tag_idx = {}
target_tag_idx = {}
reversed_tag_index = {}

# training_sentences = []


# def prepare_training_input():
#     training_data_pairs = []
#     for sentence, target_tag in training_sentences:
#         training_data_pairs.append(
#             (prepare_sequence(sentence, word_idx), prepare_sequence(target_tag, target_tag_idx))
#         )
#     return training_data_pairs

def preprocess_sentence(sentence):
    # temporarily ignore features
    sentence = sentence[0]
    # features = sentence[1]

    this_sentence = []

    for word in sentence:
        word_info = word.split()
        word = word_info[0].lower()
        # speech_tag = word_info[1]

        this_sentence.append(word)
    return prepare_sequence(this_sentence, word_idx)

def preprocess_target(sentence):
    sentence = sentence[0]
    target_tags = [word.split()[2] for word in sentence]
    return prepare_sequence(target_tags, target_tag_idx)

def build_vocab(train_data):
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


def prepare_training_data(train_data):
    training_pairs = []
    for sentence in train_data:
        preprocessed_sentence = preprocess_sentence(sentence)
        preprocessed_tag = preprocess_target(sentence)
        training_pairs.append((preprocessed_sentence, preprocessed_tag))

    return training_pairs


def prepare_test_data(test_dataset):
    # test_sentences = [preprocess_sentence(sentence) for sentence in test_dataset]
    # return test_sentences
    return [preprocess_sentence(sentence) for sentence in test_dataset]

def build_tag_index(tag_set):
    for tag in tag_set:
        target_tag_idx[tag] = len(target_tag_idx)
        reversed_tag_index[target_tag_idx[tag]] = tag

def prepare_sequence(seq, index_set):
    indices = []
    # use -1 for OOV words.
    for symbol in seq:
        if symbol in index_set:
            indices.append(index_set[symbol])
        else:
            indices.append(0)

    if use_gpu:
        return torch.tensor(indices, dtype=torch.long).cuda()
    else:
        return torch.tensor(indices, dtype=torch.long)

def predict_seq(model, input_seq):
    with torch.no_grad():
        output = model(input_seq)
        return output.max(1)[1]

def decode_seq(predicted_seq):
    if use_gpu:
        predicted_seq = predicted_seq.cpu()
    return [reversed_tag_index[idx] for idx in predicted_seq.numpy()]


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        if use_gpu:
            self.lstm = self.lstm.cuda()

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if use_gpu:
            return (torch.zeros(2, 1, self.hidden_dim).cuda(),
                    torch.zeros(2, 1, self.hidden_dim).cuda())
        else:
            return (torch.zeros(2, 1, self.hidden_dim),
                    torch.zeros(2, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden
        )
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores



def validate_model(model, validation_pairs):
    loss_function = nn.NLLLoss()
    error = 0.0
    with torch.no_grad():
        for input_seq, target_seq in validation_pairs:
            output = model(input_seq)
            loss = loss_function(output, target_seq)
            error += loss.item()
    return error

def train(data, tag_set, num_epochs):

    model = BiLSTM(embedding_dimension, hidden_unit_dimension,
                 len(word_idx), len(tag_set))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    if use_gpu:
        model = model.cuda()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for input_seq, target_tag in tqdm(data[1:90]):

            # initialize hidden state and grads before each step.
            model.zero_grad()
            model.hidden = model.init_hidden()

            training_output = model(input_seq)

            loss = loss_function(training_output, target_tag)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        valid_loss = validate_model(model, data[81:100])
        print(f"epoch {epoch} done. Training loss = {loss}, Validation loss = {valid_loss}")

    return model


def neural_train(train_data, tag_set, num_epochs):

    build_vocab(train_data)
    build_tag_index(tag_set)

    training_pairs = prepare_training_data(train_data)

    print(tag_set)
    trained_model = train(training_pairs, tag_set, num_epochs)

    return trained_model

def extract_model_data(model_data):
    global word_idx
    global speech_tag_idx
    global target_tag_idx
    global reversed_tag_index

    word_idx = model_data['word_index']
    speech_tag_idx = model_data['speech_tag_index']
    target_tag_idx = model_data['tag_index']
    reversed_tag_index = model_data['reverse_tag_index']

    model = BiLSTM(embedding_dimension, hidden_unit_dimension,
                 len(word_idx), len(target_tag_idx))
    model.load_state_dict(model_data['model'])

    return model

def test_all(model_data, test_dataset, tag_set):
    model = extract_model_data(model_data)
    test_data = prepare_test_data(test_dataset)

    predicted_tag_sequences = []
    for input_seq in test_data:
        output = predict_seq(model, input_seq)
        decoded_tags = decode_seq(output)
        predicted_tag_sequences.append(decoded_tags)

    return predicted_tag_sequences

def dump_model(model, file):
    checkpoint = {
        'model': model.state_dict(),
        'word_index': word_idx,
        'speech_tag_index': speech_tag_idx,
        'tag_index': target_tag_idx,
        'reverse_tag_index': reversed_tag_index
    }
    torch.save(checkpoint, file)

def load_model(file):
    return torch.load(file, map_location=lambda storage, loc: storage)


# TODO: treat OOV reasonably
# TODO: add speech tag embeddings
# TODO: change word embedding part
# TODO: utilizing features
