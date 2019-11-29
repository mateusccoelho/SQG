import os
import json
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

import learning.treelstm.Constants as Constants
from learning.treelstm.tree import Tree
from learning.treelstm.vocab import Vocab

from transformers import BertTokenizer

class QGDataset(data.Dataset):
    def __init__(self, path, vocab, bert_tok, num_classes):
        super(QGDataset, self).__init__()
        self.vocab = vocab
        self.bert_tok = bert_tok
        self.num_classes = num_classes

        # Converte os tokens para indices do vocabularios
        self.lsentences = self.read_sentences(os.path.join(path, 'a.txt'), bert=True)
        self.rsentences = self.read_sentences(os.path.join(path, 'b.toks'))
        self.rtrees = self.read_trees(os.path.join(path, 'b.parents'))

        # cria tensor de labels
        self.labels = self.read_labels(os.path.join(path, 'sim.txt'))

        self.size = len(self.lsentences)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        lsent = deepcopy(self.lsentences[index])
        rtree = deepcopy(self.rtrees[index])
        rsent = deepcopy(self.rsentences[index])
        label = deepcopy(self.labels[index])
        return (lsent, rtree, rsent, label)

    def read_sentences(self, filename, bert=False):
        with open(filename, 'r') as f:
            sentences = [self.read_sentence(line, bert) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line, bert):
        if(bert):
            indices = self.bert_tok.encode(line, add_special_tokens=True)
        else:
            indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.LongTensor(indices)

    def read_trees(self, filename):
        with open(filename, 'r') as f:
            trees = [self.read_tree(line) for line in tqdm(f.readlines())]
        return trees

    def read_tree(self, line):
        parents = list(map(int, line.split()))
        trees = dict()
        root = None
        for i in range(1, len(parents) + 1):
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx - 1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    def read_labels(self, filename):
        with open(filename, 'r') as f:
            labels = list(map(lambda x: float(x), f.readlines()))
            labels = torch.Tensor(labels)
        return labels
