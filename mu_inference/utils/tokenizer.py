"""
Mu Tokenizer
============

Tokenizer loading and management.

Wraps HuggingFace tokenizers with consistent interface.
"""

import os
import logging
from typing import Optional, List, Union

logger = logging.getLogger(__name__)


class MuTokenizer:
    """
    Wrapper around HuggingFace tokenizer.

    Provides consistent interface for the Mu engine.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

        # Ensure padding token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return len(self._tokenizer)

    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        return self._tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        """Beginning of sequence token ID."""
        return self._tokenizer.bos_token_id

    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
        return self._tokenizer.pad_token_id

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs,
    ):
        """
        Encode text to token IDs.

        Args:
            text: Input text or list of texts
            add_special_tokens: Add BOS/EOS tokens
            return_tensors: "pt" for PyTorch tensors

        Returns:
            Token IDs (list or tensor)
        """
        return self._tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors,
            **kwargs,
        )

    def decode(
        self,
        token_ids: Union[List[int], "torch.Tensor"],
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Skip special tokens in output

        Returns:
            Decoded text
        """
        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )

    def batch_encode(
        self,
        texts: List[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
        **kwargs,
    ):
        """
        Encode a batch of texts.

        Args:
            texts: List of input texts
            padding: Pad to same length
            truncation: Truncate to max_length
            max_length: Maximum sequence length
            return_tensors: "pt" for PyTorch

        Returns:
            BatchEncoding with input_ids and attention_mask
        """
        return self._tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs,
        )

    def batch_decode(
        self,
        token_ids_batch: Union[List[List[int]], "torch.Tensor"],
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List[str]:
        """
        Decode a batch of token ID sequences.

        Args:
            token_ids_batch: Batch of token ID sequences
            skip_special_tokens: Skip special tokens

        Returns:
            List of decoded texts
        """
        return self._tokenizer.batch_decode(
            token_ids_batch,
            skip_special_tokens=skip_special_tokens,
            **kwargs,
        )

    def __call__(self, *args, **kwargs):
        """Forward call to underlying tokenizer."""
        return self._tokenizer(*args, **kwargs)


def load_tokenizer(
    model_path: str,
    trust_remote_code: bool = True,
) -> MuTokenizer:
    """
    Load tokenizer from model path.

    Args:
        model_path: Local path or HuggingFace model ID
        trust_remote_code: Trust remote code for custom tokenizers

    Returns:
        MuTokenizer instance
    """
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer from {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        logger.warning(f"Failed to load tokenizer: {e}")
        logger.info("Trying with use_fast=False...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )

    logger.info(f"Tokenizer loaded: vocab_size={len(tokenizer)}")

    return MuTokenizer(tokenizer)
