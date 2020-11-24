# Copyright (c) 2020, XLXW In 403 AI Department Of Zhejiang Univeristy.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Import Libs
import numpy as np
from tqdm import tqdm

# Feature
class feat_utils(object):
    @staticmethod
    def tokenize(sentence):
        punc = ['.', '?', '!', "'", '''"''', ',', '%', ':']
        for p in punc:
            sentence = sentence.replace(p, ' ' + p)
        tokens = sentence.split()
        return tokens


# Pattern Data Structure
class P(object):
    def __init__(self, token, next_max_idx, exclude_fore = None, exclude_behind = None, reverse = False):
        super(P, self).__init__()
        self.token = token
        self.next = next_max_idx
        if exclude_fore is not None:
            self.exclude_fore = exclude_fore
        else:
            self.exclude_fore = []
        if exclude_behind is not None:
            self.exclude_behind = exclude_behind
        else:
            self.exclude_behind = []
        self.reverse = reverse


# Pattern Matcher
class Feature_Generator(object):
    def __init__(self):
        super(Feature_Generator, self).__init__()
        self.Patterns =[
            # 0-RULE1   : 'if....then' Exclude 'even/what/as if ... then'
            [P(['if'], -1, exclude_fore=['even', 'what', 'as']), P(['then'], 0 )],
            # 1-RULE2-1 : 'if ... had/hadn’t ...'
            #             Exclude even/what/as if ... had/hadn’t/had not ...
            #             OR any sentences where a/the/to/an immediately follows had/had not
            [P(['if'], -1, exclude_fore=['even', 'what', 'as']), P(['had', "hadn't"], 0, exclude_behind=['a', 'the', 'to', 'an'])],
            # 2-RULE2-2 : 'if ... had not ...'
            #             Exclude even/what/as if ... had/hadn’t/had not ...
            #             OR any sentences where a/the/to/an immediately follows had/had not
            [P(['if'], -1, exclude_fore=['even', 'what', 'as']), P('had', 1), P('not', 0, exclude_behind=['a', 'the', 'to', 'an'])],
            # 3-RULE3-1 : 'could/may/might/should/would/haven’t/wouldn’t/couldn’t/shouldn’t ...'
            #             Exclude a/an/the/to immediately follows have/not have/haven’t
            [P(["could", "may", "might", "should", "would", "haven’t", "wouldn’t", "couldn’t", "shouldn’t"], 0, exclude_behind=['a', 'the', 'to', 'an'])],
            # 4-RULE3-2 : 'ought to have ...'
            #             Exclude a/an/the/to immediately follows have/not have/haven’t
            [P(["ought"], 1), P(["to"], 1) ,P(["have"], 0, exclude_behind=['a', 'the', 'to', 'an'])],
            # 5-RULE3-3 : 'not have ...'
            #             Exclude a/an/the/to immediately follows have/not have/haven’t
            [P(["not"], 1), P(["have"], 0, exclude_behind=['a', 'the', 'to', 'an'])],
            # 6-RULE4   : 'what if ...'
            [P(['what'], 1), P('if', 0)],
            # 7-RULE5   : 'even if'
            [P(['even'], 1), P('if', 0)],
            # 8-RULE6-1 : 'if I/there/he/she were/weren’t.....'
            #             Exclude even/what/as I/there/you/he/she/you were/weren’t ...
            #             OR Sentences where “to” follows were/weren’t
            [P(['if'], 1, exclude_fore=['even', 'what', 'as']), P(['i', 'there', 'he', 'she', 'you'], 1), P(['were', "weren't"], 0, exclude_behind=['to'])],
            # 9-RULE6-2 : 'if I/there/he/she were not....'
            #             Exclude even/what/as I/there/you/he/she/you were/weren’t not ...
            #             OR Sentences where “to” follows were not
            [P(['if'], 1, exclude_fore=['even', 'what', 'as']), P(['i', 'there', 'he', 'she', 'you'], 1), P(['were'], 1), P(["not"], 0, exclude_behind=['to'])],
            # 10-RULE6-3 : 'if ... were/weren’t ...'
            #             Exclude even/what/as if ... were/weren’t ...
            [P(['if'], -1, exclude_fore=['even', 'what', 'as']), P(['were', "weren't"], 0)],
            # 11-RULE6-4 : 'if ... were not to ...'
            #             Exclude even/what/as if ... were not to ...
            [P(['if'], -1, exclude_fore=['even', 'what', 'as']), P(['were'], 1), P(['not'], 1), P(['to'], 0)],
            # 12-RULE7-1 : 'wish ... could/may/should/wouldn’t/couldn’t/shouldn’t have/haven’t ...
            #              OR wish I’d/we’d/you’d/he’d/she’d/they’d/there’d have/haven’t ...'
            #              Exclude Sentences that fit either of the patterns but have “to” immediately follow “wish” or
            #              a/the/to/an immediately follow have/haven’t
            [P(['wish'], -1, exclude_behind=['to']), P(
                ['could', 'may', 'should', "wouldn’t", "couldn’t", "shouldn’t", "I’d", "we’d", "you’d", "he’d", "she’d", "they’d", "there’d"], 1),
             P(['have', "haven't"], 0, exclude_behind=['a', 'the', 'an', 'to'])],
            # 13-RULE7-2 : 'wish ... could/may/should/wouldn’t/couldn’t/shouldn’t not have...
            #              OR wish I’d/we’d/you’d/he’d/she’d/they’d/there’d not have ...'
            #              Exclude Sentences that fit either of the patterns but have “to” immediately follow “wish” or
            #              a/the/to/an immediately follow not have
            [P(['wish'], -1, exclude_behind=['to']), P(
                ['could', 'may', 'should', "wouldn’t", "couldn’t", "shouldn’t", "I’d", "we’d", "you’d", "he’d", "she’d", "they’d", "there’d"], 1),
             P(['not'], 1), P(['have'], 0, exclude_behind=['a', 'the', 'an', 'to'])],
            # 14-RULE8-1 : 'wish ... were/weren’t/had/hadn’t ...
            #              Exclude Sentences that fit the pattern but have “to” immediately follow “wish” or a/the/to/an
            #              follow were/weren’t/had/hadn’t
            [P(['wish'], -1, exclude_behind=['to']), P(['were', "weren’t", "had", "hadn’t",], 0, exclude_behind=['a', 'the', 'an', 'to'])],
            # 15-RULE8-2 : 'wish ... had not ...
            #              Exclude Sentences that fit the pattern but have “to” immediately follow “wish” or a/the/to/an
            #              follow had not
            [P(['wish'], -1, exclude_behind=['to']), P(['had'], 1), P(['not'], 0, exclude_behind=['a', 'the', 'an', 'to'])],
            # 16-RULE9   : 'wish ...
            #              Exclude Sentences where “to” immediately follows “wish”
            [P(['wish'], 0, exclude_behind=['to'])],
            # 17-RULE10-1: 'but for ... could/might/would/should/wouldn’t/couldn’t/shouldn’t have/haven’t ..'
            #              Exclude Sentences that follow the pattern but where “now” follows “but for” and a/the/to/an
            #              follows have/haven’t
            [P(['but'], 1,), P(['for'], -1, exclude_behind=['now']), P(['could', 'might', 'should', 'would', "wouldn’t", "couldn’t", "shouldn’t"], 1),
             P(['have', "haven't"], 0, exclude_behind=['a', 'the', 'an', 'to'])],
            # 18-RULE10-1: 'but for ... could/might/would/should/wouldn’t/couldn’t/shouldn’t have not ..'
            #              Exclude Sentences that follow the pattern but where “now” follows “but for” and a/the/to/an
            #              follows have not
            [P(['but'], 1, ), P(['for'], -1, exclude_behind=['now']), P(['could', 'might', 'should', 'would', "wouldn’t", "couldn’t", "shouldn’t"], 1),
             P(['have'], 1), P(['not'], 0, exclude_behind=['a', 'the', 'an', 'to'])],
            # 19-RULE11  : 'if only ...'
            #              Exclude even/what/as if only ... OR if only for ...
            [P(['if'], 1, exclude_fore=['even', 'what', 'as']), P(['only'], 0, exclude_behind=['for'])],
            # ---------------------------------------------------------------------------------------------------------
            # 20-RULE12-1: 'had/were ...'
            #              Exclude had/were ... ?
            [P(['had', 'were'], -1), P(['?'], 0, reverse=True)], # * Special Case
            # 21-RULE12-2: 'had/were ...'
            #              Exclude had/were ... ?
            [P(['had', 'were'], 0)],  # * Special Case
            # ---------------------------------------------------------------------------------------------------------
            # 22-RULE14-1: 'I’d/we’d/you’d/he’d/she’d/they’d/there’d/ would/could/should/might/wouldn’t/couldn’t/shouldn’t have ... without ...'
            #              Exclude Sentence cannot end with a question mark or exclamation mark
            [P(["i’d", "we’d", "you’d", "he’d", "she’d", "they’d", "there’d", "would", "could", "should", "might", "wouldn’t", "couldn’t", "shouldn’t"], 1),
             P(['have'], -1), P(['without'],-1), P(['?', '!'], 0 ,reverse=True)],
            # 23-RULE14-2: 'I’d/we’d/you’d/he’d/she’d/they’d/there’d/ would/could/should/might/wouldn’t/couldn’t/shouldn’t have ... without ...'
            #              Exclude Sentence cannot end with a question mark or exclamation mark
            [P(["i’d", "we’d", "you’d", "he’d", "she’d", "they’d", "there’d", "would", "could", "should", "might","wouldn’t", "couldn’t", "shouldn’t"], 1),
             P(['have'], -1), P(['without'], 0)],
            # 24-RULE14-3: 'without ... I’d/we’d/you’d/he’d/she’d/they’d/there’d/would/could/should/might/wouldn’t/couldn’t/shouldn’t have ...
            #              Exclude Sentence cannot end with a question mark or exclamation mark
            [P(['without'], -1), P(["i’d", "we’d", "you’d", "he’d", "she’d", "they’d", "there’d", "would", "could", "should", "might", "wouldn’t", "couldn’t", "shouldn’t"], 1),
             P(['have'], -1), P(['?', '!'], 0, reverse=True)],
            # 25-RULE14-4: 'without ... I’d/we’d/you’d/he’d/she’d/they’d/there’d/would/could/should/might/wouldn’t/couldn’t/shouldn’t not have ...
            #              Exclude Sentence cannot end with a question mark or exclamation mark
            [P(['without'], -1), P(["i’d", "we’d", "you’d", "he’d", "she’d", "they’d", "there’d", "would", "could", "should", "might", "wouldn’t", "couldn’t", "shouldn’t"], 1),
             P(['have'], 0)],
            # 26-RULE14-5: 'without ... would not/could not/should not have ...
            #              Exclude Sentence cannot end with a question mark or exclamation mark
            [P(['without'], -1), P(["would", "could", "should"], 1), P(['not'], 1),  P(['have'], -1), P(['?', '!'], 0, reverse=True)],
            # 27-RULE14-6: 'without ... would not/could not/should not have ...
            #              Exclude Sentence cannot end with a question mark or exclamation mark
            [P(['without'], -1), P(["would", "could", "should"], 1), P(['not'], 1), P(['have'], 0)]
        ]
        self.inner = {
            0  : [6, 7],   # Rule 1    - Rule4 & Rule 5
            1  : [2],      # Rule 2-1  - Rule2-2
            3  : [4, 5],   # Rule 3-1  - Rule 3-2 & Rule 3-3
            8  : [9, 10],  # Rule 6-1  - Rule 6-2
            10 : [11],     # Rule 6-3  - Rule 6-4
            12 : [13],     # Rule 7-1  - Rule 7-2
            15 : [16],     # Rule 8-1  - Rule 8-2
            17 : [18]      # Rule 10-1 - Rule 10-2
        }
        self.include = {
            10 : [8],            # Rule 6-3 include Rule 6-1,
            16 : [15, 14, 13],   # WISH CASE
            #----------------------------------------------------------------------------------------------------------
            21 : [20],           # Special Case
            #----------------------------------------------------------------------------------------------------------
            23 : [22],           # Rule 14
            25 : [24],
            27 : [26]
        }
        self.inner_lock = [True] * len(self.Patterns)
        self.include_lock = [True] * len(self.Patterns)

    # Main Function Entity
    def __call__(self, sentences):
        labels = np.zeros(len(sentences)).astype(np.int32)
        for idx in tqdm(range(len(sentences)), desc='Processsing'):
            labels[idx] = self._match_sentence_pattern(sentences[idx])
        return  labels

    # Reset Bool Array Of Inner Lock & Include Lock
    def _reset_lock(self):
        self.inner_lock = [True] * len(self.Patterns)
        self.include_lock = [True] * len(self.Patterns)

    # Judge Exclude Pattern
    def _judge_pattern(self, tokens, pattern, start):
        left, right = start-1 if start >= 0 else 0, start+1 if start < len(tokens)-1 else len(tokens)-1
        if tokens[left] in pattern.exclude_fore or tokens[right] in pattern.exclude_behind:
            return False
        else:
            return True

    # Judge Reverse
    def _judge_reverse(self, patid):
        flag = False
        for pat in self.Patterns[patid]:
            if pat.reverse:
                flag = True
                break
        return flag

    # Get Sub Tokens Of Sentence
    def _get_subtokens(self, tokens, pattern, start):
        range_left = pattern.next if pattern.next != -1 and start + pattern.next < len(tokens) else len(tokens) - 1
        tp_search = tokens[start:start+range_left + 1]
        return tp_search

    # Judge Pattern Includes
    def _judge_include(self, patid):
        flag = True
        if patid in self.include.keys():
            for idx in self.include[patid]:
                if self.include_lock[idx] is not True:
                    flag = False
                    break
        return  flag

    # Match The Pattern
    def _match_sentence_pattern(self, tokens):
        labels, pidx = [], 0
        self._reset_lock()
        for pat in range(len(self.Patterns)):
            if self.inner_lock[pat] is not True:
                labels.append(0)
                continue
            st_idx = [i for i, x in enumerate(tokens) if x in self.Patterns[pat][pidx].token]
            if len(st_idx) == 0 :
                labels.append(0)
                continue
            flag = 0
            for sidx in st_idx:
                if not self._judge_pattern(tokens, self.Patterns[pat][0], sidx):
                    self.include_lock[pat] = False
                    if pat in self.inner.keys():
                        for lock_idx in self.inner[pat]:
                            self.inner_lock[lock_idx] = False
                    continue
                if self._recur_match(tokens, self.Patterns[pat], pat, 0, sidx):
                   flag += 1
            if flag > 0:
                if pat in self.include.keys() and self._judge_include(pat) is not True:
                    labels.append(0)
                else:
                    if self._judge_reverse(pat):
                        labels.append(0)
                        self.include_lock[pat] = False
                    else:
                        labels.append(1)
            else:
                labels.append(0)
        return 1 in labels

    # Judge Internal Pattern
    def _recur_match(self, tokens, pattern, patternid, pat_idx, st_idx):
        if pat_idx == len(pattern)-1:
            return True
        tp_search = self._get_subtokens(tokens, pattern[pat_idx], st_idx)
        st_idx = [i+st_idx for i, x in enumerate(tp_search) if x in pattern[pat_idx+1].token]
        if len(st_idx) == 0:
            return False
        for st in st_idx:
            if not self._judge_pattern(tokens, pattern[pat_idx+1], st):
                self.include_lock[patternid] = False
                if patternid in self.inner.keys():
                    for lock_idx in self.inner[patternid]:
                        self.inner_lock[lock_idx] = False
                return False
            return self._recur_match(tokens, pattern, patternid, pat_idx+1, st)

if __name__ == '__main__':
    test_data = [[]]
    fgen = Feature_Generator()
    label = fgen(test_data)
    print(label)