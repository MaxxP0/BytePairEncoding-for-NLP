import collections
from collections import Counter
import pickle
import re
from pathlib import Path



class BPEtokenizer():
    def __init__(self):
        self.trained_tokenizer = None
        self.vocab = {}
        self.tokens = {'<unk>':0,'<pad>':1,'<sos>':2,'<eos>':3,'\n':4}


    def __len__(self):
        return len(self.tokens)

    
    def load(cls,save_path):
        bpe = BPEtokenizer()
        bpe.tokens = pickle.load(Path(save_path).open('rb'))
        return bpe


    def build_vocab(self,corpus):
        """Build vocab from text corpus"""

        # Separate each char in word by space and add mark end of token
        corpus = (re.split('(\W)',corpus))
        corpus = [x for x in corpus if x]

        new_corpus = []
        for word in corpus:
            if word != ' ':
                new_corpus.append(word)
            else:
                if len(new_corpus) > 0:
                    new_corpus[-1]+=(' ')
        
        tokens = [" ".join(word.replace(' ', '</w>')) for word in new_corpus]
        
        # Count frequency of tokens in corpus
        self.vocab = Counter(tokens)
        return self.vocab


    def get_pairs(self):
        """Get counts of pairs of consecutive symbols"""

        pairs = collections.defaultdict(int)
        for word, frequency in self.vocab.items():
            symbols = word.split()

            # Counting up occurrences of pairs
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += frequency

        return pairs


    def merge_vocab(self,pair):
        """Merge all occurrences of the most frequent pair"""
        
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in self.vocab:
            # replace most frequent pair in all vocabulary
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = self.vocab[word]

        self.vocab = v_out
        return self.vocab


    def get_token_pairs(self):
        tokens = {}
        for word in self.vocab:
            word = word.split()
            for token in word:
                if token not in tokens:
                    tokens[token] = 0
                tokens[token] +=1
            
        return tokens


    def to_token_list(self):
        tokens = self.get_token_pairs()
        tokens = {k: v for k, v in sorted(tokens.items(), key=lambda item: item[1],reverse=True)}

        new_d = {}
        for k in sorted(tokens, key=len, reverse=True):
            new_d[k] = tokens[k]

        tokens = new_d

        lengh = len(self.tokens)
        for i,token in enumerate(tokens):
            i += lengh
            self.tokens[token] = i


    def train(self,corpus,iterations,vocab_size,save_path,merges_per_iteration=20):
        """
        parameters:
        -----------
        corpus    : str
            str   : corpus on which the BPE should train

        iterations: int
            int   : maximum number of iterations allowed
        
        vocab_size: int
            int   : maximum lengh of the vocabulary allowed
                    if vocab_size < letters in corpus
                        vocab_size = letters in corpus
                    ensuring the whole corpus can be encoded

        save_path : str
            str   : name of the trained vocabulary

        merges_per_iteration: int
            int   : number of maximum allowed merges per iteration
                    for faster training                  

        ___________
        return    :
            saves a trained vocabulary which is
            sorted by size and frequency of tokens
        """

        self.build_vocab(corpus)
        for i in range(iterations):
            print(len(self.get_token_pairs()))

            token_pair_lengh = len(self.get_token_pairs())

            if not vocab_size >= (token_pair_lengh+len(self.tokens)+1):
                break
            
            if (token_pair_lengh+merges_per_iteration+10) > vocab_size:
                merges_per_iteration = 1

            token_pairs = self.get_pairs()
            token_pairs = sorted(token_pairs, key=token_pairs.get, reverse=True)[:merges_per_iteration]

            if not token_pairs:
                break

            for token_pair in token_pairs:
                self.vocab = self.merge_vocab(token_pair)
        
        self.to_token_list()

        pickle.dump(self.tokens,Path(save_path).open('wb'))


    def encode(self,string,
                    max_lengh=None,
                    hard_stop=False,
                    sos_token=False,
                    eos_token=False,
                    long_tensor_format=None):

        """
        parameters:
        -----------
        max_lengh : int
            int   : the sequence lengh will be size(n,max_lengh)
            None  : a sequence of any kind will be encoded without change of dimensionality

        hard_stop : bool
            True  : the sequence will be split into one sequence of lengh == max_lengh
            False : a sequence of lengh>max_lengh will be split into a batch of
                sequences of lengh == max_lengh and will be padded accordingly
                
        sos_token : bool
            True  : a START of Sentence token will be the first token of the sequence

        eos_token : bool
            True  : a END of Sentence token will be the last token of the sequence

        long_tensor_format : TensorObject
            None  : encoded sequence will be returned as python list of lists
            torch.LongTensor : the sequence will be returned as a torch Tensor

        ___________
        return    : list        
        """
        encoded_word = []

        if sos_token == True:
            encoded_word.append(self.tokens['<sos>'])

        eos_lengh_offset = 0
        eos_token_added = False
        if eos_token==True:
            eos_lengh_offset = 1

        split = re.split('(\W)',string)
        split = [x for x in split if x]

        #Number of spaces is preserved
        #here the spaces are added to the
        #last word      'hello',' ',' '
        #becomes        'hello  '
        new_split = []
        for word in split:
            if word != ' ':
                new_split.append(word)
            else:
                if len(new_split) > 0:
                    new_split[-1]+=(' ')
        split = new_split


        #encoding works by decreasing the
        #word lengh from right to left
        #and checking if the word/subword
        #is in the vocab if True the
        #coresponding token is appended
        #this process is repeated for the
        #remainder of the word, this does
        #NOT guarantee that the optimal
        #subwords are found
        for word in split:
            tokenized_lengh = 0
            word0 = word.replace(' ','</w>')

            while tokenized_lengh <= len(word0):
                word = word0[tokenized_lengh:]

                if len(word) > 0:
                    for index in range(len(word)):
                        index = len(word) - index
                        
                        if word[:index] in self.tokens:
                            encoded_word.append(self.tokens[word[:index]])
                            tokenized_lengh += len(word[:index])                            
                            break

                        if (len(word[:index]) == 1) & (word[:index] not in self.tokens):
                            encoded_word.append(self.tokens['<unk>'])
                            tokenized_lengh += 1
                            break
                    
                    #This case if actual seq lengh is HIGHER than specified max lengh
                    #and hard_stop == True, this is placed here for encoding speed
                    if (hard_stop==True) & (max_lengh!=None):
                        if (len(encoded_word)+eos_lengh_offset) == max_lengh:

                            if eos_token==True:
                                encoded_word.append(self.tokens['<eos>'])
                                eos_token_added = True

                            if long_tensor_format != None:
                                encoded_word = long_tensor_format(encoded_word)
                                
                            return [encoded_word]
                else:
                    break
        
        #This case if actual seq lengh is LOWER than specified max lengh and needs to be padded
        if (hard_stop==True) & (max_lengh!=None):
            if eos_token==True:
                encoded_word.append(self.tokens['<eos>'])
                eos_token_added = True

            while len(encoded_word) < max_lengh:
                encoded_word.append(self.tokens['<pad>'])
        
        
        #This case if seq needs to be split up and padded to match seq lengh
        if (hard_stop==False) & (max_lengh!=None):
            if eos_token==True:
                encoded_word.append(self.tokens['<eos>'])
                eos_token_added = True

            encoded_word = [encoded_word[x:x+max_lengh] for x in range(0, len(encoded_word), max_lengh)]
        

            while len(encoded_word[-1]) < max_lengh:
                encoded_word[-1].append(self.tokens['<pad>'])

        #The following two statements if max lengh is not defined and a sequence of n size can be encoded
        else:
            encoded_word = [encoded_word]

        if (eos_token == True) & (eos_token_added==False):
            encoded_word[-1].append(self.tokens['<eos>'])

        if long_tensor_format != None:
            encoded_word = long_tensor_format(encoded_word)

        return encoded_word


    def decode(self,tokens):
        #decodes encodings and replaces
        #white space tokens with white
        #space

        string = ''
        for i in tokens:
            string += list(self.tokens)[i]

        string = string.replace('</w>',' ')
        return string
