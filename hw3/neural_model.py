import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

from tqdm import tqdm
import sys
from BiLSTM_CRF import BiLSTM_CRF
import perc
from neural_config import *
from allennlp.commands.elmo import ElmoEmbedder

use_gpu = torch.cuda.is_available()

word_idx = {'<UNKNOWN>': 0}
speech_tag_idx = {'<UNKNOWN>': 0}
target_tag_idx = {}
reversed_tag_index = {}

elmo = None
if use_elmo:
    print("initializing ELMo embedding... ", file=sys.stderr)
    if use_gpu:
        elmo = ElmoEmbedder(cuda_device=0)
    else:
        elmo = ElmoEmbedder()
    print("loaded. ", file=sys.stderr)

def preprocess_sentence(sentence):
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

    if use_elmo:
        this_sentence = elmo.embed_sentence(this_sentence)[2]
        this_sentence = torch.from_numpy(this_sentence)
        if use_gpu:
            this_sentence = this_sentence.cuda()

    else:
        this_sentence = prepare_sequence(this_sentence, word_idx)
    this_speech_tags = prepare_sequence(this_speech_tags, speech_tag_idx)
    return this_sentence, this_speech_tags

def preprocess_target(sentence):
    sentence = sentence[0]
    target_tags = [word.split()[2] for word in sentence]
    return prepare_sequence(target_tags, target_tag_idx)
    # return target_tags

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
    training_tuples = []
    # if use_elmo:
    #     print("loading pre-trained ELMo...", file=sys.stderr)
    #     global elmo
    #     elmo = ElmoEmbedder()
    #     print("ELMo loaded. Now preprocess training sentences", file=sys.stderr)
    for sentence in tqdm(train_data):
        preprocessed_sentence, preprocessed_speech_tag = preprocess_sentence(sentence)

        preprocessed_tag = preprocess_target(sentence)
        training_tuples.append((preprocessed_sentence, preprocessed_speech_tag, preprocessed_tag))

    print("training data prepared. ", file=sys.stderr)
    return training_tuples



def prepare_test_data(test_dataset):
    # if use_elmo:
    #     global elmo
    #     elmo = ElmoEmbedder()
    return [preprocess_sentence(sentence) for sentence in tqdm(test_dataset)]

def build_tag_index(tag_set):
    target_tag_idx['<start>'] = 0
    target_tag_idx['<end>'] = 1
    for tag in tag_set:
        target_tag_idx[tag] = len(target_tag_idx)
        reversed_tag_index[target_tag_idx[tag]] = tag

def prepare_sequence_batch(seq_batch, index_set):
    return [prepare_sequence(sequence, index_set) for sequence in seq_batch]

def prepare_sequence(seq, index_set):
    indices = []
    # use -1 for OOV words.
    for symbol in seq:
        if symbol in index_set:
            indices.append(index_set[symbol])
        else:
            indices.append(index_set['<UNKNOWN>'])

    if use_gpu and not test_mode:
        return torch.tensor(indices, dtype=torch.long).cuda()
    else:
        return torch.tensor(indices, dtype=torch.long)

def predict_seq(model, input_seq):
    with torch.no_grad():
        output = model(input_seq[0], input_seq[1])
        return output.max(1)[1]

def decode_seq(predicted_seq):
    if use_gpu:
        predicted_seq = predicted_seq.cpu()
    return [reversed_tag_index[idx] for idx in predicted_seq.numpy()]


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, speech_tag_size, tagset_size):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_unit_dimension

        if use_elmo:
            global word_embedding_dimension
            word_embedding_dimension = elmo_dimension

        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dimension)
        self.speech_embeddings = nn.Embedding(speech_tag_size, speech_embedding_dimension)
        self.lstm = nn.LSTM(word_embedding_dimension + speech_embedding_dimension,
                            hidden_unit_dimension,
                            num_layers=LSTM_layer,
                            bidirectional=True)
        self.speech_lstm = nn.LSTM
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        if use_gpu and not test_mode:
            self.lstm = self.lstm.cuda()

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_unit_dimension * 2, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden_first_size = 2 * LSTM_layer
        if use_gpu and not test_mode:
            return (torch.zeros(hidden_first_size, 1, self.hidden_dim).cuda(),
                    torch.zeros(hidden_first_size, 1, self.hidden_dim).cuda())
        else:
            return (torch.zeros(hidden_first_size, 1, self.hidden_dim),
                    torch.zeros(hidden_first_size, 1, self.hidden_dim))

    def forward(self, sentence, speech_tags):
        sentence_length = len(speech_tags)
        word_embeds = sentence[0:sentence_length] if use_elmo \
            else self.word_embeddings(sentence)
        speech_embeds = self.speech_embeddings(speech_tags)
        embeds = torch.cat((word_embeds, speech_embeds), 1)
        lstm_out, self.hidden = self.lstm(
            embeds.view(sentence_length, 1, -1), self.hidden
        )
        tag_space = self.hidden2tag(lstm_out.view(sentence_length, -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def validate_model(model, validation_pairs):
    loss_function = nn.NLLLoss()
    error = 0.0
    with torch.no_grad():
        for input_seq, speech_tag, target_seq in validation_pairs:
            output = model(input_seq, speech_tag)
            loss = loss_function(output, target_seq)
            error += loss.item()
    return error

def train(tuples, tag_set, num_epochs):

    print("initializing LSTM model... ", file=sys.stderr)
    model = BiLSTM(len(word_idx), len(speech_tag_idx), len(tag_set))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    if use_gpu:
        model = model.cuda()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for input_seq, input_tag, target_tag in tqdm(tuples):

            # initialize hidden state and grads before each step.
            model.zero_grad()
            model.hidden = model.init_hidden()

            training_output = model(input_seq, input_tag)

            loss = loss_function(training_output, target_tag)
            running_loss += loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()

        valid_loss = validate_model(model, tuples[101:200])
        print(f"epoch {epoch} done. Training loss = {running_loss}, Validation loss = {valid_loss}",
              file=sys.stderr)

    return model


def neural_train(train_data, tag_set, num_epochs):

    build_vocab(train_data)
    build_tag_index(tag_set)
    if prototyping_mode:
        train_data = train_data[1:32]

    print("preparing training tuples...", file=sys.stderr)
    training_tuples = prepare_training_data(train_data)

    trained_model = train(training_tuples, tag_set, num_epochs)

    return trained_model

def bilstmcrf_train(train_data, tag_set, num_epochs):

    build_vocab(train_data)
    build_tag_index(tag_set)
    if prototyping_mode:
        train_data = train_data[1:32]

    print("preparing training tuples...", file=sys.stderr)
    training_tuples = prepare_training_data(train_data)

    print("initializing BiLSTM-CRF model... ", file=sys.stderr)
    model = BiLSTM_CRF(len(word_idx), len(speech_tag_idx), len(target_tag_idx))
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    if use_gpu:
        model = model.cuda()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for input_seq, input_tag, target_tag in tqdm(training_tuples):

            # initialize hidden state and grads before each step.
            model.zero_grad()
            loss = model.NLLloss(input_seq, input_tag, target_tag)
            running_loss += loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()

        valid_loss = validate_model(model, training_tuples[101:200])
        print(f"epoch {epoch} done. Training loss = {running_loss}, Validation loss = {valid_loss}",
              file=sys.stderr)

    return model

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
    if use_gpu and not test_mode:
        model = model.cuda()
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
    # return torch.load(file)
    return torch.load(file, map_location=lambda storage, loc: storage)


# PARTLY-DONE: treat OOV reasonably
# MAYBE-DONE: add speech tag embeddings
# TODO: change word embedding part
# TODO: utilizing features
