# BytePairEncoding-for-NLP
A simple python BPE implementation

How to Train, Encode, Decode
```python
text = """
Byte pair encoding[1][2] or digram coding[3] is a simple
form of data compression in which the most common pair of consecutive
bytes of data is replaced with a byte that does not occur within that data."""

from BPE import BPEtokenizer
bpe = BPEtokenizer()
bpe.train(text,1000,75,'vocab')

token_list = bpe.encode('Hello World this is the BPE tokenizer',4,False,True,True)
print(token_list)

for token in token_list:
   print(bpe.decode(token))
```

How to Load
```python

bpe = bpe.load('wiki1k.vocab')
token_list = bpe.encode('Hello World this is the BPE tokenizer')

for token in token_list:
   print(bpe.decode(token))
```
