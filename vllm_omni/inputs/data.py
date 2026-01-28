from typing import Any

try:
    from typing import NotRequired
except ImportError:
    # Python < 3.11: use typing_extensions
    from typing_extensions import NotRequired

from typing import TypeAlias

import torch
from vllm.inputs.data import (
    EmbedsInputs,
    EmbedsPrompt,
    TextPrompt,
    TokenInputs,
    TokensPrompt,
)


class OmniTextPrompt(TextPrompt):
    """Text prompt with optional embeddings and additional information."""

    prompt_embeds: NotRequired[torch.Tensor]
    additional_information: NotRequired[dict[str, Any]]


class OmniTokensPrompt(TokensPrompt):
    """Tokens prompt with optional embeddings and additional information.

    Extends TokensPrompt to support prompt embeddings and additional
    information payloads for direct transfer between pipeline stages.

    Attributes:
        prompt_embeds: Optional tensor containing prompt embeddings
        additional_information: Optional dictionary containing additional
            information (tensors or lists) to pass along with the prompt
    """

    prompt_embeds: NotRequired[torch.Tensor]
    """The embeddings of the prompt."""

    # New: optional additional information dictionary
    # Values may be torch.Tensor or list
    additional_information: NotRequired[dict[str, Any]]


class OmniTokenInputs(TokenInputs):
    """Token inputs with optional embeddings and additional information.

    Extends TokenInputs to support prompt embeddings and additional
    information payloads for direct transfer between pipeline stages.

    Attributes:
        prompt_embeds: Optional tensor containing prompt embeddings
            aligned with token IDs
        additional_information: Optional dictionary containing additional
            information (tensors or lists) to pass along with the inputs
    """

    # New: optional prompt embeddings aligned with token ids
    prompt_embeds: NotRequired[torch.Tensor]

    # New: optional additional information dictionary
    # Values may be torch.Tensor or list
    additional_information: NotRequired[dict[str, Any]]


class OmniEmbedsPrompt(EmbedsPrompt):
    """Embeddings prompt with optional additional information."""

    additional_information: NotRequired[dict[str, Any]]


class OmniEmbedsInputs(EmbedsInputs):
    """Embeddings inputs with optional additional information."""

    additional_information: NotRequired[dict[str, Any]]


OmniSingletonPrompt: TypeAlias = str | OmniTextPrompt | OmniTokensPrompt | OmniEmbedsPrompt


def token_inputs_omni(
    prompt_token_ids: list[int],
    prompt: str | None = None,
    cache_salt: str | None = None,
    prompt_embeds: torch.Tensor | None = None,
    additional_information: dict[str, Any] | None = None,
) -> OmniTokenInputs:
    """Construct token inputs with optional embeddings and metadata.

    Creates an OmniTokenInputs object with token IDs and optional
    embeddings and additional information for pipeline stage transfer.

    Args:
        prompt_token_ids: List of token IDs for the prompt
        prompt: Optional prompt string
        cache_salt: Optional cache salt for prefix caching
        prompt_embeds: Optional tensor containing prompt embeddings
        additional_information: Optional dictionary containing additional
            information (tensors or lists)

    Returns:
        OmniTokenInputs instance with the provided data
    """
    inputs = OmniTokenInputs(type="token", prompt_token_ids=prompt_token_ids)

    if prompt is not None:
        inputs["prompt"] = prompt
    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt
    if prompt_embeds is not None:
        inputs["prompt_embeds"] = prompt_embeds
    if additional_information is not None:
        inputs["additional_information"] = additional_information

    return inputs


def embeds_inputs_omni(
    prompt_embeds: torch.Tensor,
    cache_salt: str | None = None,
    additional_information: dict[str, Any] | None = None,
) -> OmniEmbedsInputs:
    """Construct embeddings inputs with optional metadata."""
    inputs: OmniEmbedsInputs = OmniEmbedsInputs(type="embeds", prompt_embeds=prompt_embeds)
    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt
    if additional_information is not None:
        inputs["additional_information"] = additional_information
    return inputs
