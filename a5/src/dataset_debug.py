#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 27/8/2022 20:13
# @Author  : Yu Li
# @Site    : 
# @File    : dataset_debug.py
# -*- coding: utf-8
import random
import torch
from torch.utils.data import Dataset
import argparse
import numpy as np

"""
The input-output pairs (x, y) of the NameDataset are of the following form:

  x: Where was Khatchig Mouradian born?⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  x: Where was Jacob Henry Studer born?⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

Using the PAD_CHAR characters in y before the ⁇[place] keeps the trainer from
optimizing the model to predict the question, "Where was...".

Note that the NameDataset should take the pretraining_dataset defined in run.py
as an input. This is to allow the vocab specification of the NameDataset to be
the same as that of the pretraining dataset.

You don't need to implement anything in NameDataset.
"""


class NameDataset(Dataset):
    def __init__(self, pretraining_dataset, data):
        self.MASK_CHAR = u"\u2047"  # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1"  # the empty square character, for pad
        self.itos = pretraining_dataset.itos
        self.stoi = pretraining_dataset.stoi
        self.block_size = pretraining_dataset.block_size
        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))

    def __len__(self):
        # returns the length of the dataset
        return len(self.data) - 1

    def __getitem__(self, idx):
        inp, oup = self.data[idx].split('#')
        x = inp + self.MASK_CHAR + oup + self.MASK_CHAR
        x = x + self.PAD_CHAR * (self.block_size - len(x))
        y = self.PAD_CHAR * (len(inp) - 1) + x[len(inp):]

        x = x[:-1]
        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
        return x, y


"""
[part e]

Write a class that yields examples of a simplified span corruption objective.
Do not change the signature of the __init__ or __getitem__ functions.

Make sure to implement the full spec for full credit -- we list below the
criteria that must be satisfied for a full implementation.

--------------
Vocabulary Specification

Your vocabulary is to be accessible via two dictionaries:
  self.stoi: a dictionary from characters in the vocabulary to indices of type
      int
  self.itos: a dictionary from indices of type int to characters in the
      vocabulary

Your vocabulary must have the following form: 

  Identifier 0 must be assigned to the unicode element u"\u25A1".
      This is the empty_square_character.
      Further, let self.PAD_CHAR = u"\u25A1"
  Identifier 1 must be assigned to the unicode element u"\u2047".
      This is the doublequestionmark character, which we'll use
      as a sentinel to represent that text is missing from the input
      Further, let self.MASK_CHAR = u"\u2047"
  Identifiers 2, ..., len(self.itos)-1 should be the sorted list of characters
      that appear in the data argument.

--------------
Masking Specification

The __getitem__ function takes an index and returns a data point (x, y) where
x and y are Long tensors of length self.block_size. x encodes the input
sequence, and y encodes the output sequence.

0. ✅Use the idx argument of __getitem__ to retrieve the element of self.data
at the given index. We'll call the resulting data entry a document.

1. ✅Randomly truncate the document to a length no less than 4 characters,
and no more than int(self.block_size*7/8) characters.

- IMPORTANT: You are free to decide how to perform this random truncation, but
make sure that the length is picked _randomly_ (every possible length from 4
to int(self.block_size*7/8) has a chance of being picked) for full credit.

2. ✅Now, break the (truncated) document into three substrings:

    [prefix] [masked_content] [suffix]

  In other words, choose three strings prefix, masked_content and suffix
    such that prefix + masked_content + suffix = [the original document].
  The length of [masked_content] should be random, and 1/4 the length of the
    truncated document on average.

- IMPORTANT: You are free to decide how to perform this operation, but
make sure that the length is picked _randomly_ (has a chance of being more or
less than 1/4 the length of the truncated document) for full credit.

3. Rearrange these substrings into the following form:

    [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads]

  This resulting string, denoted masked_string, serves as the output example.
  Here MASK_CHAR is the masking character defined in Vocabulary Specification,
    and [pads] is a string of repeated PAD_CHAR characters chosen so that the
    entire string is of length self.block_size.
  Intuitively, the [masked_content], a string, is removed from the document and
    replaced with MASK_CHAR (the masking character defined in Vocabulary
    Specification). After the suffix of the string, the MASK_CHAR is seen again,
    followed by the content that was removed, and the padding characters.

4. We now use masked_string to construct the input and output example pair. To
do so, simply take the input string to be masked_string[:-1], and the output
string to be masked_string[1:]. In other words, for each character, the goal is
to predict the next character in the masked string.

5. Making use of the vocabulary that you defined, encode the resulting input
and output strings as Long tensors and return the resulting data point.

----------------
Here are some examples of input-output pairs (x, y):

  x: Khatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: hatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

  x: Jaco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: aco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

  x: John Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: ohn Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□


"""


class CharCorruptionDataset1(Dataset):
    def __init__(self, data, block_size):
        self.MASK_CHAR = u"\u2047"  # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1"  # the empty square character, for pad

        chars = list(sorted(list(set(data))))
        assert self.MASK_CHAR not in chars
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.PAD_CHAR)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data.split('\n')

    def __len__(self):
        # returns the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # TODO [part e]: see spec above
        document = self.data[idx]
        # 1. randomly truncate to [4, 7/8 * block_size]
        doc_len = len(document)
        truncate_len = random.randint(4, int(self.block_size * 7 / 8))
        truncate_len = min(doc_len, truncate_len)
        truncated_doc = document[:truncate_len]
        # 2. break to [prefix] [masked_content] [suffix]
        masked_len = random.randint(int(1 / 8 * truncate_len), int(3 / 8 * truncate_len))
        assert truncate_len >= 4, (doc_len, truncate_len, masked_len, document, idx)
        prefix_len = random.randint(1, truncate_len - masked_len - 1)

        prefix = truncated_doc[:prefix_len]
        masked_content = truncated_doc[prefix_len:prefix_len + masked_len]
        suffix = truncated_doc[prefix_len + masked_len:]

        # 3. rearrange to masked_string: [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads]
        masked_string = prefix + self.MASK_CHAR + suffix + self.MASK_CHAR + masked_content + self.PAD_CHAR * (
                self.block_size - truncate_len - 2)
        assert len(masked_string) == self.block_size
        print(len(masked_string))
        # 4. input = masked_string[:-1], output = masked_string[1:]
        x = masked_string[:-1]
        y = masked_string[1:]

        # 5. encode to Long tensors
        x = torch.LongTensor([self.stoi[c] for c in x])
        y = torch.LongTensor([self.stoi[c] for c in y])
        return x, y

        inp, oup = self.data[idx].split('\t')
        x = inp + self.MASK_CHAR + oup + self.MASK_CHAR
        x = x + self.PAD_CHAR * (self.block_size - len(x))
        y = self.PAD_CHAR * (len(inp) - 1) + x[len(inp):]

        x = x[:-1]
        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
        print(len(x), len(y))
        return x, y

class CharCorruptionDataset2(Dataset):
    def __init__(self, data, block_size):
        self.MASK_CHAR = u"\u2047"  # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1"  # the empty square character, for pad

        chars = list(sorted(list(set(data))))
        assert self.MASK_CHAR not in chars
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.PAD_CHAR)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data.split('\n')

    def __len__(self):
        # returns the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # TODO [part e]: see spec above
        # corruption_dataset[idx]通过这个方法返回元素
        # 0. use the idx argument of __getitem__ to retrieve the element of self.data at the given index
        # this data entry is called document
        entry = self.data[idx]
        # 1. Randomly truncate the document to a length no less than 4 characters,
        # and no more than int(self.block_size*7/8) characters.
        truncate_index = random.randint( 4, int(self.block_size * 7/8) ) # 截断之后的长度
        entry = entry[:truncate_index]
        # 2. 将截断之后的document随机分成
        #     [prefix][masked_content][suffix]
        # 三个部分，其中[masked_content]要占1/4
        entry_len = len(entry)
        masked_content_len = int(np.random.normal(entry_len * 1/4))
        masked_content_start = random.randint(0, entry_len - masked_content_len)
        masked_content = entry[masked_content_len: masked_content_start+masked_content_len]
        prefix = entry[:masked_content_start]
        suffix = entry[masked_content_start+masked_content_len:]
        # 3. 将masked_string重新排列成 [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads]的样子
        # [pads]是用很多个pad_char将masked_string补全成block_size的长度
        masked_string = prefix + self.MASK_CHAR + suffix + self.MASK_CHAR + masked_content
        assert len(masked_string) == self.block_size
        num_pads = self.block_size - len(masked_string)
        masked_string += self.PAD_CHAR * num_pads
        print(len(masked_string))
        # 4. We now use masked_string to construct the input and output example pair. To
        # do so, simply take the input string to be masked_string[:-1], and the output
        # string to be masked_string[1:]. In other words, for each character, the goal is
        # to predict the next character in the masked string.
        input = masked_string[:-1]
        output = masked_string[1:]
        # input和output刚好错开了，我们的目标是用前一个character预测后一个character
        x = torch.tensor([self.stoi[c] for c in input], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in output], dtype=torch.long)
        print(len(x), len(y))
        return x, y


"""
Code under here is strictly for your debugging purposes; feel free to modify
as desired.
"""
if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('dataset_type', help="Type of dataset to sample from."
                                           "Options: namedata, charcorruption.",
                      choices=["namedata", "charcorruption"])
    args = argp.parse_args()

    if args.dataset_type == 'namedata':
        # Even if it hasn't been implemented, we use it to define the vocab
        corruption_dataset = CharCorruptionDataset(open('wiki.txt').read(), 128)
        # Make the name dataset
        name_dataset = NameDataset(corruption_dataset,
                                   open('birth_places_train.tsv').read())
        for _, example in zip(range(4), name_dataset):
            x, y = example
            print('x:', ''.join([name_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([name_dataset.itos[int(c)] for c in y]))
        pass
    elif args.dataset_type == 'charcorruption':
        corruption_dataset = CharCorruptionDataset1(open('wiki.txt').read(), 128)
        for _, example in zip(range(4), corruption_dataset):
            x, y = example
        corruption_dataset = CharCorruptionDataset2(open('wiki.txt').read(), 128)
        for _, example in zip(range(4), corruption_dataset):
            x, y = example
    else:
        raise ValueError("Unknown dataset type in command line args: {}"
                         .format(args.dataset_type))

