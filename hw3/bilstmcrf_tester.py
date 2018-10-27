import os
import perc
import optparse
import sys
import bilstmcrf_config as config
from util import *
from BiLSTM_CRF import BiLSTM_CRF

def test_model(model, data_tuples, idx2tag, cuda):
    predicted_tags = []
    with torch.no_grad():
        for input_seq, input_tag in data_tuples:
            if cuda:
                input_seq = input_seq.cuda()
                input_tag = input_tag.cuda()
            tag_indices = model.decode(input_seq, input_tag).cpu()
            predicted_tags.append([idx2tag[tag_idx] for tag_idx in tag_indices])

    return predicted_tags

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--inputfile", dest="inputfile", default=os.path.join("data", "dev.txt"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "dev.feats"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join("data", "default.model"), help="weights for all features stored on disk")
    (opts, _) = optparser.parse_args()

    cuda = torch.cuda.is_available()
    tagset = perc.read_tagset(opts.tagsetfile)
    print("reading data ...", file=sys.stderr)

    test_data = perc.read_labeled_data(opts.inputfile, opts.featfile, verbose=False)
    print("done.", file=sys.stderr)
    if config.prototyping_mode:
        test_data = test_data[0:32]

    model_data = load_model(opts.modelfile)

    word_idx = model_data['word_index']
    speech_tag_idx = model_data['speech_tag_index']
    tag2idx = model_data['tag_index']
    idx2tag = model_data['reverse_tag_index']

    model = BiLSTM_CRF(len(word_idx), len(speech_tag_idx), len(tag2idx))
    model.load_state_dict(model_data['model'])
    if cuda:
        model.cuda()
    test_tuples = prepare_test_data(test_data, speech_tag_idx)

    predicted_tags = test_model(model, test_tuples, idx2tag, cuda)

    for idx, _ in enumerate(predicted_tags):
        print("\n".join(perc.conll_format(predicted_tags[idx], test_data[idx][0])))
        print()
