import nlm
import re


class NlmScorer:
    def __init__(self, model, cuda):
        self.model = model
        self.cuda = cuda

    def _get_bitstring_spans(self, bitstring):
        """get a list of spans that are contiguous and have 'o' in
        the string position. ignore '.' positions"""
        return {i.span()[0]: i.span()[1] for i in re.finditer('o', bitstring)}

    def score_bitstring(self, sequence, bitstring):
        """a bitstring is a string where 'o' represents an item to
        be scored and '.' represents an item to be ignored while
        scoring the sequence. the sequence string and bitstring
        must be of the same length and the sequence cannot contain
        punctuation or spaces"""
        spans = self._get_bitstring_spans(bitstring)

        # we use the tab character \t to represent the positions
        # to skip when scoring the sequence
        seq_by_bits = ''.join([sequence[i] if i in spans else '\t' for i in range(len(sequence))])
        seqs = seq_by_bits.split()
        lm_logprob = sum(list(map(lambda seq: nlm.score_sequence(seq, self.model, self.cuda), seqs)))
        return lm_logprob

    def score_seq(self, seq):
        return nlm.score_sequence(seq, self.model, self.cuda)

    def score_partial_seq(self, seq):
        return nlm.score_partial_seq(seq, self.model, self.cuda)

if __name__ == '__main__':
    cuda = False
    model = nlm.load_model("data/mlstm_ns.pt", cuda=cuda)
    scorer = NlmScorer(model, cuda)
    print(scorer.score_bitstring('thisisatest', 'oo...oo.ooo'))
    print(scorer.score_bitstring('thisisateas', 'oo...oo.ooo'))
    print(scorer.score_bitstring('zkxxuqxzpuq', 'oo...oo.ooo'))
