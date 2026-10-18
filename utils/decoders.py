from typing import Any, Dict, List, Type

import numpy as np

from .vocab import GenerationVocab


class GreedyTextDecoder:
    """Greedy CTC decoder (collapse blanks and repeated labels)."""

    def __init__(
        self,
        blank_index: int,
        tokenizer: Type[GenerationVocab],
        confidences: bool = False,
    ) -> None:
        self._confidences = confidences
        self.tokenizer = tokenizer
        self._blank_index = blank_index

    def __call__(self, ctc_output, *args) -> List[Dict[str, Any]]:
        """Decode CTC logits of shape (batch, time, classes)."""
        indices = ctc_output.argmax(axis=-1)
        output = []

        for sample, mat in zip(indices, ctc_output):
            previous = self._blank_index
            decoded = []
            confs = []
            for ind, element in enumerate(sample):
                if element == self._blank_index:
                    previous = self._blank_index
                    continue
                if element == previous:
                    continue
                decoded.append(element)
                previous = element
                confs.append(mat[ind])

            decoded = np.array(decoded)
            confs = np.array(confs)
            if self._confidences:
                output.append({"text": decoded, "text_conf": confs})
            else:
                output.append({"text": decoded, "text_conf": None})

        return output
