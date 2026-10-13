"""Align the tokens of two prompts word by word, so attention maps can move between them.

A word can split into a different number of tokens in each prompt:
    "lemon cake."     -> [CLS] lemon            cake . [SEP] [PAD] ...
    "pistachio cake." -> [CLS] pi ##sta ##chio  cake . [SEP] [PAD] ...
so token i of one prompt is not token i of the other.
"""
import torch


def word_token_positions(prompt, tokenizer):
    """Token positions of each space-separated word, and the position of [SEP].

    Position 0 is [CLS]. BERT splits on spaces before it splits words into
    pieces, so tokenizing word by word gives the same tokens as the whole prompt.
    """
    positions, start = [], 1
    for word in prompt.split():
        count = len(tokenizer.tokenize(word))
        positions.append(list(range(start, start + count)))
        start += count
    return positions, start


def token_mapper(source, target, tokenizer, length=77):
    """[length, length] matrix M such that ``target_map = source_map @ M``.

    M[s, t] is the share of source token s's attention that goes to target token t.
    Both prompts must have the same number of words. Words with equal token
    counts map token to token; otherwise each source token of the word spreads
    its attention evenly over the target word's tokens. Every row sums to 1,
    so each pixel's attention over the tokens still sums to 1.

    Padding keeps its position instead of shifting with the words: about 90% of
    the attention lands on padding, and a padding token's text feature depends
    on its absolute position (shifting it moved that attention onto unlike
    features and ruined the layout).
    """
    source_words, source_end = word_token_positions(source, tokenizer)
    target_words, target_end = word_token_positions(target, tokenizer)
    if len(source_words) != len(target_words):
        raise ValueError(f"{source!r} and {target!r} must have the same number of words.")
    if max(source_end, target_end) >= length - 1:
        raise ValueError(f"{source!r} or {target!r} is longer than {length - 3} tokens.")

    mapper = torch.zeros(length, length)
    mapper[0, 0] = 1  # [CLS]
    for source_ids, target_ids in zip(source_words, target_words):
        if len(source_ids) == len(target_ids):
            mapper[source_ids, target_ids] = 1  # one to one
        else:
            mapper[source_ids[0]:source_ids[-1] + 1, target_ids[0]:target_ids[-1] + 1] = 1 / len(target_ids)
    mapper[source_end, target_end] = 1  # [SEP]
    # Padding stays in place. Source padding where the target still has words
    # goes to the target's first padding position, so no attention is lost.
    for s in range(source_end + 1, length):
        mapper[s, max(s, target_end + 1)] = 1
    return mapper
