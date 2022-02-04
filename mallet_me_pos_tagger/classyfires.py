import os
import re
from collections import Counter


def vect_from_raw(sentence_list):
    sentence_clean = [re.sub('\n', '', x) for x in sentence_list]
    sentence_clean = [x for x in sentence_clean if x]
    sentence_clean = [re.sub(',', 'comma', x) for x in sentence_clean]
    sentence_split = [re.split(r"\s+", x) for x in sentence_clean]
    sentence_pairs = [[re.split(r"\/", sl) for sl in l] for l in sentence_split]
    return sentence_pairs


class WordVector:
    def __init__(self, word, tag, sentence_num, word_num, is_rare, prevTag, prev2Tags, prevW, prev2W, nextW, next2W):
        self.curW = word
        self.tag = tag
        self.sentence_num = sentence_num
        self.word_num = word_num
        self.is_rare = is_rare
        self.pref = []
        self.suf = []
        self.containsNum = 0
        self.containsUppercase = 0
        self.containshyphen = 0
        self.prevT = prevTag
        self.prevTwoTags = prev2Tags
        self.prevW = prevW
        self.prev2W = prev2W
        self.nextW = nextW
        self.next2W = next2W
        self.init_vect = []
        self.kept_vect = []
        self.set_rare_features()
        self.create_init_vect()

    def set_rare_features(self):
        if self.is_rare:
            if re.search(r'-', self.curW):
                self.containshyphen = 1
            if re.search(r'\d', self.curW):
                self.containsNum = 1
            if re.search(r'[A-Z]', self.curW):
                self.containsUppercase = 1

            for n in range(1, 5):
                try:
                    prefix = self.curW[:n]
                    self.pref.append(prefix)
                except:
                    pass

            for n in range(len(self.curW) - 1, len(self.curW) - 5, -1):
                try:
                    suffix = self.curW[n:]
                    self.suf.append(suffix)
                except:
                    pass

    def create_init_vect(self):
        its_rare = self.is_rare
        vect_dict = self.__dict__.copy()
        helper_list = ['is_rare', 'tag', 'sentence_num', 'word_num', 'init_vect', 'kept_vect']
        contains_list = ['containsNum', 'containshyphen', 'containsUppercase']

        for helper in helper_list:
            del (vect_dict[helper])

        for key, value in vect_dict.items():
            if key == 'pref' or key == 'suf':
                if its_rare:
                    if value:
                        for item in value:
                            feature = str(key) + '=' + str(item)
                            self.init_vect.append(feature)

            elif key in contains_list:
                if its_rare:
                    if value >= 1:
                        feature = str(key)
                        self.init_vect.append(feature)
            else:
                feature = str(key) + '=' + str(value)
                self.init_vect.append(feature)

    def create_final_vect(self, kept_features):
        self.kept_vect = [feat for feat in self.init_vect if feat in kept_features]


class Corpus:
    def __init__(self, train_raw, test_raw, rare_threshold, feat_threshold):
        self.train_raw = train_raw
        self.test_raw = test_raw
        self.rare_threshold = rare_threshold
        self.feat_threshold = feat_threshold
        self.train_voc = {}
        self.init_feats = {}
        self.kept_feats = {}
        self.train_vect = []
        self.test_vect = []
        self.process_train()
        self.process_test()

    def generate_word_vects(self, word_pairs):
        vector_list = []
        for sentence_idx, line in enumerate(word_pairs):
            sent_num = sentence_idx + 1
            for word_idx, pair in enumerate(line):
                curW_idx = word_idx
                prevW_idx = word_idx - 1
                prev2W_idx = word_idx - 2
                nextW_idx = word_idx + 1
                next2W_idx = word_idx + 2
                # print(pair)
                curW = pair[0]
                curT = pair[1]

                if curW in self.train_voc:
                    curW_freq = self.train_voc[curW]
                else:
                    curW_freq = 0

                if curW_freq < self.rare_threshold:
                    curW_is_rare = True
                else:
                    curW_is_rare = False

                if curW_idx == 0:
                    prevW = 'BOS'
                    prev2W = 'BOS'
                    prevT = 'BOS'
                    prevTwoTags = 'BOS+' + str(prevT)
                elif curW_idx == 1:
                    prevW = line[prevW_idx][0]
                    prev2W = 'BOS'
                    prevT = line[prevW_idx][1]
                    prevTwoTags = 'BOS+' + str(prevT)
                else:
                    prevW = line[prevW_idx][0]
                    prev2W = line[prev2W_idx][0]
                    prevT = line[prevW_idx][1]
                    prevTwoTags = line[prev2W_idx][1] + '+' + str(prevT)

                if curW_idx == len(line) - 1:
                    nextW = 'EOS'
                    next2W = 'EOS'
                elif curW_idx == len(line) - 2:
                    nextW = line[nextW_idx][0]
                    next2W = 'EOS'
                else:
                    nextW = line[nextW_idx][0]
                    next2W = line[next2W_idx][0]

                word_vector = WordVector(curW, curT, sent_num, curW_idx, curW_is_rare
                                         , prevT, prevTwoTags, prevW, prev2W, nextW, next2W
                                         )
                vector_list.append(word_vector)
        return vector_list

    def process_train(self):
        train_pairs = vect_from_raw(self.train_raw)

        vocab = []
        [[vocab.append(sl[0]) for sl in l] for l in train_pairs]

        self.train_voc = dict(Counter(vocab).most_common())
        self.train_vect = self.generate_word_vects(train_pairs)

        init_feat_list = []
        for vect in self.train_vect:
            for feature in vect.init_vect:
                init_feat_list.append(feature)

        self.init_feats = dict(Counter(init_feat_list).most_common())

        self.kept_feats = {k: v for (k, v) in self.init_feats.items() if v >= self.feat_threshold or 'curW' in k}

        for vect in self.train_vect:
            vect.create_final_vect(self.kept_feats)

    def process_test(self):
        test_pairs = vect_from_raw(self.test_raw)
        self.test_vect = self.generate_word_vects(test_pairs)
        for vect in self.test_vect:
            vect.create_final_vect(self.kept_feats)

    def save_feats(self, output_directory):
        voc_path = output_directory + '/train_voc'
        init_feats_path = output_directory + '/init_feats'
        kept_feats_path = output_directory + '/kept_feats'
        feature_summary_path = output_directory + '/feature_summary'

        with open(voc_path, 'w', encoding='utf8') as f:
            for feature, freq in self.train_voc.items():
                f.write('%s %s\n' % (feature, freq))

        with open(init_feats_path, 'w', encoding='utf8') as f:
            for feature, freq in self.init_feats.items():
                f.write('%s %s\n' % (feature, freq))

        with open(kept_feats_path, 'w', encoding='utf8') as f:
            for feature, freq in self.kept_feats.items():
                f.write('%s %s\n' % (feature, freq))

        len_init = len(self.init_feats)
        len_kept = len(self.kept_feats)
        with open(feature_summary_path, 'w', encoding='utf8') as f:
            f.write('Num of Features=%s\n' % (len_init))
            f.write('Num of Kept Features=%s\n' % (len_kept))

    def save_to_mallet(self, output_directory, train=True):
        output_lines = []
        if train:
            vectors = self.train_vect
            output_path = output_directory + '/final_train.vectors.txt'
        else:
            vectors = self.test_vect
            output_path = output_directory + '/final_test.vectors.txt'
        for vect in vectors:
            sentence_num = vect.sentence_num
            word_num = vect.word_num
            word = vect.curW
            tag = vect.tag
            features = vect.kept_vect
            features = [str(x) + ' 1' for x in features]
            if vect.is_rare:
                contains_set = {'containsNum', 'containshyphen', 'containsUppercase'}
                for i in contains_set:
                    str_i = str(i) + ' 1'
                    if str_i not in features:
                        str_i_zero = str(i) + ' 0'
                        features.append(str_i_zero)
            features_str = ' '.join(features)
            output_line = str(sentence_num) + '-' + str(word_num) + '-' + str(word) + ' ' + str(tag) + ' ' + str(
                features_str)
            output_lines.append(output_line)

        with open(output_path, 'w', encoding='utf8') as f:
            f.writelines("%s\n" % line for line in output_lines)
