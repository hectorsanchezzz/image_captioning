from typing import List

import numpy as np
from numpy.typing import ArrayLike


class GenerationVocab:
    """Word vocabulary with <START>, <STOP>, <PAD>, and <UNK> for the baseline GRU."""

    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    PAD_TAG = "<PAD>"
    UNKNOWN_TAG = "<UNK>"

    AUX_TAGS = [START_TAG, STOP_TAG, PAD_TAG, UNKNOWN_TAG]

    def __init__(self, VOCAB: List) -> None:
        self.tokens = self.AUX_TAGS + VOCAB

        self.token2index = {tok: ii for ii, tok in enumerate(self.tokens)}
        self.index2token = {ii: tok for tok, ii in self.token2index.items()}

    def __len__(self) -> int:
        return len(self.token2index)

    def tokenise(self, line: List[str]) -> List[int]:
        return [
            self.token2index[tok]
            if tok in self.token2index
            else self.token2index[self.UNKNOWN_TAG]
            for tok in line
        ]

    def pad(self, tokenised: List[int], size: int) -> ArrayLike:
        padded = np.full((size,), self.token2index[self.PAD_TAG])
        max_index = min(len(tokenised), size - 2)
        padded[1 : max_index + 1] = tokenised[:max_index]
        padded[0] = self.token2index[self.START_TAG]
        padded[max_index + 1] = self.token2index[self.STOP_TAG]
        return padded

    def prepare(self, line: List[str], size: int) -> ArrayLike:
        return self.pad(self.tokenise(line), size)

    def unpad(self, padded: ArrayLike) -> List[int]:
        output: List[int] = []
        for tok in padded:
            if tok not in {
                self.token2index[self.START_TAG],
                self.token2index[self.PAD_TAG],
            }:
                if tok == self.token2index[self.STOP_TAG]:
                    return output
                output.append(tok)

        return output

    def detokenise(self, tokenised: List[int]) -> List[str]:
        return [self.index2token.get(ind, self.UNKNOWN_TAG) for ind in tokenised]

    def unprepare(self, padded: ArrayLike) -> List[str]:
        return self.detokenise(self.unpad(padded))
