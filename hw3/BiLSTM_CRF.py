import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

from tqdm import tqdm
import sys
import util
import perc
import bilstmcrf_config as config
from allennlp.commands.elmo import ElmoEmbedder

START_IDX = 0
END_IDX = 1

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, speech_tag_size, tagset_size):
        super(BiLSTM, self).__init__()
        self.hidden_dim = config.hidden_unit_dimension

        if config.use_elmo:
            global word_embedding_dimension
            word_embedding_dimension = config.elmo_dimension

        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dimension)
        self.speech_embeddings = nn.Embedding(speech_tag_size, config.speech_embedding_dimension)
        self.lstm = nn.LSTM(word_embedding_dimension + config.speech_embedding_dimension,
                            config.hidden_unit_dimension,
                            num_layers=config.LSTM_layer,
                            bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(config.hidden_unit_dimension * 2, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2*config.LSTM_layer, 1, self.hidden_dim),
                torch.randn(2*config.LSTM_layer, 1, self.hidden_dim))

    def forward(self, sentence, speech_tags):
        sentence_length = len(speech_tags)
        word_embeds = sentence[0:sentence_length] if self.use_elmo \
            else self.word_embeddings(sentence)
        speech_embeds = self.speech_embeddings(speech_tags)
        embeds = torch.cat((word_embeds, speech_embeds), 1)
        lstm_out, self.hidden = self.lstm(
            embeds.view(sentence_length, 1, -1), self.hidden
        )
        tag_space = self.hidden2tag(lstm_out.view(sentence_length, -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_space

class CRF(nn.Module):
    def __init__(self, tagset_size):
        self.tagset_size = tagset_size
        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size)) # [C, C]
        self.transitions[START_IDX, :] = -10000
        self.transitions[:, END_IDX] = -10000

    def forward(self, h):
        forward_var = torch.full((self.tagset_size), -10000) # [C]
        forward_var[0][START_IDX] = 0

        for emit_score in h:
            emit_score = emit_score.view(-1, 1).expand(-1, self.tagset_size) # [C, 1] => [C, C]
            forward_var = forward_var.view(1, -1).expand(self.tagset_size, -1)
            forward_score_t = forward_var + self.transitions + emit_score
            forward_var = util.log_sum_exp(forward_score_t)

        terminal_var = forward_var.view(-1) + self.transitions[END_IDX]
        forward_score = util.log_sum_exp(terminal_var)
        return forward_score

    def score(self, h, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([START_IDX], dtype=torch.long), tags])

        for i, emit_score in enumerate(h):
            score += self.transitions[tags[i+1], tags[i]] + emit_score[tags[i+1]]
        score += self.transitions[END_IDX, tags[-1]]
        return score

    def decode(self, h):
        bkptrs = []
        forward_var = torch.full((self.tagset_size), -10000)
        forward_var[0][START_IDX] = 0

        for emit_score in h:
            emit_score = emit_score.view(-1, 1).expand(-1, self.tagset_size) # [C, 1] => [C, C]
            forward_var = forward_var.view(1, -1).expand(self.tagset_size, -1)
            forward_score_t = forward_var + self.transitions + emit_score
            forward_var, prev_ind = torch.max(forward_score_t, -1)
            bkptrs.append(prev_ind)

        terminal_var = forward_var.view(-1) + self.transitions[END_IDX]
        forward_score, best_tag = torch.max(terminal_var, -1)

        best_path = [best_tag]
        for bkptr in bkptrs[::-1]:
            best_tag = bkptr[best_tag]
            best_path.append(best_tag)

        start = best_path.pop()
        assert start == START_IDX
        best_path.reverse()
        return best_path



class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, speech_tag_size, tagset_size):
        self.vocab_size = vocab_size
        self.speech_tag_size = speech_tag_size
        self.tagset_size = tagset_size

        self.bilstm = BiLSTM(vocab_size, speech_tag_size, tagset_size)
        self.crf = CRF(tagset_size)

    def NLLloss(self, sentence, speech_tags, tags):
        h = self.bilstm(sentence, speech_tags)
        forward_score = self.crf(h)
        gold_score = self.crf.score(h, tags)

        return forward_score - gold_score

    def decode(self, sentence, speech_tags):
        h = self.bilstm(sentence, speech_tags)
        return self.crf.decode(h)