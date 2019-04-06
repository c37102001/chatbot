import argparse
import logging
import os
import pdb
import sys
import traceback
import pickle
import json
from embedding import Embedding
from preprocessor import Preprocessor


def main(args):

    test_json_path = args.test_json_path
    embedding_pkl_path = "../data/embedding.pkl"   # TODO

    preprocessor = Preprocessor(None)
    with open(embedding_pkl_path, 'rb') as f:
        preprocessor.embedding = pickle.load(f)

    # test
    logging.info('Processing test from {}'.format(test_json_path))
    test = preprocessor.get_dataset(
        test_json_path, args.n_workers,
        {'n_positive': -1, 'n_negative': -1, 'shuffle': False}
    )
    test_pkl_path = "../data/test.pkl"    # TODO
    logging.info('Saving test to {}'.format(test_pkl_path))
    with open(test_pkl_path, 'wb') as f:
        pickle.dump(test, f)


def _parse_args():
    parser = argparse.ArgumentParser(description="Preprocess and generate preprocessed pickle.")
    parser.add_argument('test_json_path', type=str, help='[input] Path to the directory that .')
    parser.add_argument('--n_workers', type=int, default=4)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
        level=logging.INFO, datefmt='%m-%d %H:%M:%S'
    )
    args = _parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
