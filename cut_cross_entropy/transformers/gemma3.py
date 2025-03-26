# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from types import MethodType
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
from transformers.cache_utils import Cache, HybridCache, StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gemma3.modeling_gemma3 import (
    _CONFIG_FOR_DOC,
    GEMMA3_INPUTS_DOCSTRING,
    logger,
)
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

from .utils import PatchOptions, TransformersModelT, apply_lce

_PATCH_OPTS: PatchOptions | None = None


@add_start_docstrings_to_model_forward(GEMMA3_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def cce_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[HybridCache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
    **loss_kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        num_logits_to_keep (`int`, *optional*):
            Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, GemmaForCausalLM

    >>> model = GemmaForCausalLM.from_pretrained("google/gemma-2-9b")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

    >>> prompt = "What is your favorite condiment?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "What is your favorite condiment?"
    ```"""

    if self.training and self.config._attn_implementation != "eager":
        logger.warning_once(
            "It is strongly recommended to train Gemma3 models with the `eager` attention implementation "
            f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
        )
    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **loss_kwargs,
    )

    hidden_states = outputs[0]
    loss = None
    logits = None

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        loss = apply_lce(hidden_states, self.lm_head.weight, labels, _PATCH_OPTS, **loss_kwargs)
    else:
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
        # In Gemma3 the `final_logit_softcapping` defaults to None, this will not activate (however, I am leaving it in as a softcapping value can be optionally applied in theory)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def multimodal_cce_forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **lm_kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

    Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        >>> model = Gemma3ForConditionalGeneration.from_pretrained("google/Gemma3-test-224px-hf")
        >>> processor = AutoProcessor.from_pretrained("google/Gemma3-test-224px-hf")

        >>> prompt = "answer en Where is the cow standing?"
        >>> url = "https://huggingface.co/gv-hf/Gemma3-test-224px-hf/resolve/main/cow_beach_1.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "answer en Where is the cow standing?\nbeach"
        ```"""

    if self.training and self.config._attn_implementation != "eager":
        logger.warning_once(
            "It is strongly recommended to train Gemma3 models with the `eager` attention implementation "
            f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
        )
    if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    is_training = token_type_ids is not None and labels is not None

        # Replace image id woth PAD if the image token if OOV, to avoid index-errors
    if input_ids is not None and self.config.image_token_index >= self.vocab_size:
        special_image_mask = input_ids == self.config.image_token_index
        llm_input_ids = input_ids.clone()
        llm_input_ids[special_image_mask] = 0
    else:
        llm_input_ids = input_ids

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(llm_input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # Merge text and images
    if pixel_values is not None:
        image_features = self.get_image_features(pixel_values)

        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_index, dtype=torch.long, device=inputs_embeds.device)
            )
        else:
            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

        if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
            image_tokens_in_text = (special_image_mask).sum(dim=1).sum(dim=0)[0]
            raise ValueError(
                f"Number of images does not match number of special image tokens in the input text. "
                f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                "tokens from image embeddings."
            )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    # mask out pad-token-ids in labels for BC
    if labels is not None and self.pad_token_id in labels:
        logger.warning_once(
            "`labels` contains `pad_token_id` which will be masked with `config.ignore_index`. "
            "You have to mask out `pad_token_id` when preparing `labels`, this behavior will be removed in v.4.46.",
        )
        labels = torch.where(input_ids == self.pad_token_id, self.config.ignore_index, labels)

    causal_mask = self._update_causal_mask(
        attention_mask, token_type_ids, past_key_values, cache_position, inputs_embeds, is_training
    )
    outputs = self.language_model(
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        logits_to_keep=logits_to_keep,
        **lm_kwargs,
    )

    hidden_states = outputs[0]
    loss = None
    logits = None

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        loss = apply_lce(hidden_states, self.language_model.lm_head.weight, labels, _PATCH_OPTS, **loss_kwargs)
    else:
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = self.language_model.lm_head(hidden_states[:, -logits_to_keep, :])
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Gemma3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


def patch_gemma3(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    from transformers.models.gemma3 import modeling_gemma3

    _PATCH_OPTS = patch_options

    if isinstance(maybe_model, transformers.PreTrainedModel):
        assert isinstance(
            maybe_model, modeling_gemma3.Gemma3ForCausalLM
        ), f"Expected a Gemma3ForCausalLM model. Got {type(maybe_model)}."
        maybe_model.forward = MethodType(cce_forward, maybe_model)
        return maybe_model
    else:
        modeling_gemma3.Gemma3ForCausalLM.forward = cce_forward
        modeling_gemma3.Gemma3ForConditionalGeneration.forward = multimodal_cce_forward
