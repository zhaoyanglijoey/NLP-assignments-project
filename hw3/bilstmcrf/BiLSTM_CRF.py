import torch
import torch.nn as nn

from bilstmcrf import util

START_IDX = 0
END_IDX = 1
ELMO_DIM = 1024

class BiLSTM_Encoder_decoder(nn.Module):
    def __init__(self, speech_tag_size, tagset_size, device,
                 layer, hidden_dim, pos_dim):
        super(BiLSTM_Encoder_decoder, self).__init__()
        self.device = device

        word_embedding_dimension = ELMO_DIM
        self.hidden_dim = hidden_dim
        self.layer = layer
        self.speech_embeddings = nn.Embedding(speech_tag_size, pos_dim)
        self.encoder = nn.LSTM(word_embedding_dimension + pos_dim,
                               self.hidden_dim,
                               num_layers=self.layer,
                               bidirectional=True)
        self.decoder = nn.LSTM(word_embedding_dimension + pos_dim,
                               self.hidden_dim,
                               num_layers=self.layer,
                               bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

        self.encode2init_h = nn.Linear(self.hidden_dim * 2 * layer, self.hidden_dim * 2 * layer)
        self.encode2init_c = nn.Linear(self.hidden_dim * 2 * layer, self.hidden_dim * 2 * layer)

    def init_hidden(self):
        return (torch.zeros(2*self.layer, 1, self.hidden_dim).to(self.device),
                torch.zeros(2*self.layer, 1, self.hidden_dim).to(self.device))

    def forward(self, sentence, speech_tags):
        enc_hidden = self.init_hidden()
        sentence_length = len(sentence)
        word_embeds = sentence
        speech_embeds = self.speech_embeddings(speech_tags)
        embeds = torch.cat((word_embeds, speech_embeds), 1)

        _, (enc_hidden, enc_cell) = self.encoder(embeds.view(sentence_length, 1, -1), enc_hidden)

        enc_hidden = enc_hidden.view(1, -1)
        enc_cell = enc_cell.view(1, -1)
        init_h = self.encode2init_h(enc_hidden).view(2*self.layer, 1, self.hidden_dim)
        init_c = self.encode2init_c(enc_cell).view(2*self.layer, 1, self.hidden_dim)

        lstm_out, _ = self.decoder(embeds.view(sentence_length, 1, -1), (init_h, init_c))
        lstm_feats = self.hidden2tag(lstm_out.view(sentence_length, -1))

        return lstm_feats

class BiLSTM_Enc_Dec_CRF(nn.Module):
    def __init__(self, speech_tag_size, tagset_size, device,
                 layer, hidden_dim, pos_dim):
        super(BiLSTM_Enc_Dec_CRF, self).__init__()
        self.enc_dec = BiLSTM_Encoder_decoder(speech_tag_size, tagset_size, device,
                                              layer, hidden_dim, pos_dim)
        self.crf = CRF(tagset_size, device)

    def NLLloss(self, sentence, speech_tags, tags):
        lstm_feats = self.enc_dec(sentence, speech_tags)
        forward_score = self.crf(lstm_feats)
        gold_score = self.crf.score(lstm_feats, tags)

        return forward_score - gold_score

    def decode(self, sentence, speech_tags):
        lstm_feats = self.enc_dec(sentence, speech_tags)
        return self.crf.decode(lstm_feats)

class BiLSTM_CRF(nn.Module):
    def __init__(self, speech_tag_size, tagset_size, device,
                 layer, hidden_dim, pos_dim):
        super(BiLSTM_CRF, self).__init__()
        self.speech_tag_size = speech_tag_size
        self.tagset_size = tagset_size

        self.bilstm = BiLSTM(speech_tag_size, tagset_size, device,
                             layer, hidden_dim, pos_dim)
        self.crf = CRF(tagset_size, device)

    def NLLloss(self, sentence, speech_tags, tags):
        h = self.bilstm(sentence, speech_tags)
        forward_score = self.crf(h)
        gold_score = self.crf.score(h, tags)

        return forward_score - gold_score

    def decode(self, sentence, speech_tags):
        h = self.bilstm(sentence, speech_tags)
        return self.crf.decode(h)

class BiLSTM(nn.Module):
    def __init__(self, speech_tag_size, tagset_size, device,
                 layer, hidden_dim, pos_dim):
        super(BiLSTM, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.layer = layer

        word_embedding_dimension = ELMO_DIM

        self.speech_embeddings = nn.Embedding(speech_tag_size, pos_dim)
        self.lstm = nn.LSTM(word_embedding_dimension + pos_dim,
                            hidden_dim,
                            num_layers=layer,
                            bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(2*self.layer, 1, self.hidden_dim).to(self.device),
                torch.zeros(2*self.layer, 1, self.hidden_dim).to(self.device))

    def forward(self, sentence, speech_tags):
        self.hidden = self.init_hidden()
        sentence_length = len(speech_tags)
        word_embeds = sentence[:sentence_length]
        speech_embeds = self.speech_embeddings(speech_tags)
        embeds = torch.cat((word_embeds, speech_embeds), 1)
        lstm_out, self.hidden = self.lstm(
            embeds.view(sentence_length, 1, -1), self.hidden
        )
        tag_space = self.hidden2tag(lstm_out.view(sentence_length, -1))
        # tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_space

class CRF(nn.Module):
    def __init__(self, tagset_size, device):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.device = device
        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size)) # [C, C]

        self.transitions.data[START_IDX, :] = -10000
        self.transitions.data[:, END_IDX] = -10000

    def forward(self, h):
        forward_var = torch.full((1, self.tagset_size), -10000).to(self.device) # [C]
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
        score = torch.zeros(1).to(self.device)
        tags = torch.cat([torch.tensor([START_IDX], dtype=torch.long).to(self.device), tags])

        for i, emit_score in enumerate(h):
            score += self.transitions[tags[i+1], tags[i]] + emit_score[tags[i+1]]
        score += self.transitions[END_IDX, tags[-1]]
        return score

    def decode(self, h):
        bkptrs = []
        forward_var = torch.full((1, self.tagset_size), -10000).to(self.device)
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


