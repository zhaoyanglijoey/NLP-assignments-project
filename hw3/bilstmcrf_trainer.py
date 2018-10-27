import perc
import sys, optparse, os
import neural_model
from collections import defaultdict

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs", dest="numepochs", default=int(10), help="number of epochs of training; in each epoch we iterate over over all the training examples")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join("data", "default.model"), help="weights for all features stored on disk")
    (opts, _) = optparser.parse_args()

    # each element in the feat_vec dictionary is:
    # key=feature_id value=weight

    tagset = perc.read_tagset(opts.tagsetfile)
    print("reading data ...", file=sys.stderr)
    train_data = perc.read_labeled_data(opts.trainfile, opts.featfile, verbose=False)
    print("done.", file=sys.stderr)
    trained_model = neural_model.bilstmcrf_train(train_data, tagset, int(opts.numepochs))
    neural_model.dump_model(trained_model, opts.modelfile)
