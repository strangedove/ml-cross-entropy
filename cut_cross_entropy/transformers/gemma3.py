# --- Existing imports and _PATCH_OPTS setup ---
from types import MethodType
from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
import transformers
from transformers.cache_utils import HybridCache, Cache, StaticCache # Added Cache, StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gemma3.modeling_gemma3 import (
    _CONFIG_FOR_DOC,
    GEMMA3_INPUTS_DOCSTRING,
    logger,
    Gemma3CausalLMOutputWithPast, # Ensure this is imported if needed for return type hint
    is_torchdynamo_compiling, # Added import
)
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
# Assuming apply_lce and PatchOptions are correctly defined elsewhere
from .utils import PatchOptions, TransformersModelT, apply_lce

_PATCH_OPTS: PatchOptions | None = None

# --- cce_forward (looks ok for Gemma3ForCausalLM) remains the same ---
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
    # num_logits_to_keep is deprecated, use logits_to_keep if needed by inner model
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **loss_kwargs, # Renamed from lm_kwargs for consistency? Or keep lm_kwargs if specific?
                  # Assuming these are loss-related args for apply_lce
) -> Union[Tuple, CausalLMOutputWithPast]:
    # ... (cce_forward implementation - seems okay for its target) ...
    # NOTE: The original cce_forward uses num_logits_to_keep in the lm_head call.
    # If logits_to_keep is meant for apply_lce, it should be used there.
    # If it's meant for the standard path, the lm_head call needs it.
    # Let's assume it's for the standard path as in the original Gemma3ForCausalLM.forward

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
        # Pass logits_to_keep down if the inner model uses it (Gemma3TextModel doesn't directly, but passes it to layers)
        # The key difference is Gemma3ForCausalLM applies it *before* lm_head
        **loss_kwargs, # Pass remaining kwargs down
    )

    hidden_states = outputs[0]
    loss = None
    logits = None # Initialize logits

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        assert labels is not None
        # Assuming apply_lce takes hidden_states, lm_head weights, labels, options, and maybe kwargs
        loss = apply_lce(hidden_states, self.lm_head.weight, labels, _PATCH_OPTS, **loss_kwargs)
        # Logits remain None when using LCE, as LCE calculates loss directly
    else:
        # Calculate logits using the logic from the original Gemma3ForCausalLM.forward
        # Only compute necessary logits if logits_to_keep is specified
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # Apply softcapping if configured (from original Gemma3ForCausalLM.forward)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        # Calculate standard loss if labels are provided
        if labels is not None:
            # Use the standard loss function defined in the model (copied from original)
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

    if not return_dict:
        # Ensure output structure matches original, including handling None loss/logits
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    # Ensure the output class matches (CausalLMOutputWithPast for Gemma3ForCausalLM)
    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


# --- REVISED multimodal_cce_forward ---
@add_start_docstrings_to_model_forward(GEMMA3_INPUTS_DOCSTRING) # Keep relevant docstrings
# Use the specific output type for the multimodal model
@replace_return_docstrings(output_type=Gemma3CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def multimodal_cce_forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: Optional[torch.FloatTensor] = None, # Keep pixel_values
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
    token_type_ids: Optional[torch.LongTensor] = None, # Keep token_type_ids
    cache_position: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    # Pass logits_to_keep down to the language_model call
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **lm_kwargs, # Keep lm_kwargs for passing down, potentially used by apply_lce too
) -> Union[Tuple, Gemma3CausalLMOutputWithPast]:
    r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens *within the language model if it supports it*. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).
                **Note**: In this multimodal context, `logits_to_keep` is passed *down* to the underlying language model.

    Returns:
        Gemma3CausalLMOutputWithPast (or tuple)

        Example: (Copied from original, should still work)
        ```python
        # ... (original example) ...
        ```"""

    # --- Start: Setup code copied directly from original Gemma3ForConditionalGeneration.forward ---
    if self.training and self.config.text_config._attn_implementation != "eager": # Check text_config for attn_impl
        logger.warning_once(
            "It is strongly recommended to train Gemma3 models with the `eager` attention implementation "
            f"instead of `{self.config.text_config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
        )
    if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    # Use text_config for output_hidden_states and use_return_dict? Original used self.config. Check config structure. Assuming self.config is the top-level multimodal config.
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    is_training = token_type_ids is not None and labels is not None

    # Replace image id woth PAD if the image token if OOV, to avoid index-errors
    # Use self.config (multimodal) for image_token_index
    if input_ids is not None and self.config.image_token_index >= self.vocab_size:
        special_image_mask = input_ids == self.config.image_token_index
        llm_input_ids = input_ids.clone()
        llm_input_ids[special_image_mask] = 0 # Assuming 0 is PAD or a safe index
    else:
        llm_input_ids = input_ids

    if inputs_embeds is None:
        # Use the multimodal model's embedding getter
        inputs_embeds = self.get_input_embeddings()(llm_input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # Store original image_features for the output object later
    image_features = None
    # Merge text and images
    if pixel_values is not None:
        image_features = self.get_image_features(pixel_values) # shape (num_images, image_seq_len, embed_dim)

        # Create mask based on image token index
        if input_ids is None: # Handle case where only inputs_embeds are provided
             # This comparison might be fragile if embeds are modified. Relying on input_ids is safer.
             # Need a robust way to identify image token positions in embeds if input_ids aren't available.
             # Assuming input_ids are generally available when pixel_values are used.
             raise NotImplementedError("Identifying image tokens solely from inputs_embeds is not robustly implemented.")
             # Simplified placeholder logic if needed, but relies on exact embed value:
             # img_token_embed = self.get_input_embeddings()(
             #    torch.tensor(self.config.image_token_index, dtype=torch.long, device=inputs_embeds.device)
             # )
             # special_image_mask = torch.all(inputs_embeds == img_token_embed, dim=-1) # Check all embed dims match
        else:
            special_image_mask = (input_ids == self.config.image_token_index)

        # Expand mask to embedding dimension
        special_image_mask_expanded = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

        # Flatten image features to match scatter locations: (num_images * image_seq_len, embed_dim)
        num_images, image_seq_len, embed_dim = image_features.shape
        flat_image_features = image_features.view(-1, embed_dim)

        # Check if the number of image tokens in input_ids matches the flattened image features
        num_image_tokens_in_ids = special_image_mask.sum()
        if not is_torchdynamo_compiling() and num_image_tokens_in_ids != flat_image_features.shape[0]:
             # Corrected error message logic from original
             raise ValueError(
                 f"Number of image tokens ({self.config.image_token_index}) in input_ids ({num_image_tokens_in_ids}) "
                 f"does not match the number of tokens expected from image features ({flat_image_features.shape[0]}). "
                 f"Each image should correspond to {image_seq_len} tokens in the input."
             )

        # Use masked_scatter: needs mask of same shape as input tensor
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask_expanded, flat_image_features.to(inputs_embeds.device, inputs_embeds.dtype))


    # mask out pad-token-ids in labels for BC (Copied from original)
    # Use self.config (multimodal) for pad_token_id and ignore_index
    if labels is not None and hasattr(self.config, 'pad_token_id') and self.config.pad_token_id is not None and self.config.pad_token_id in labels:
        # Ensure ignore_index is available in the config
        ignore_index = getattr(self.config, 'ignore_index', -100) # Defaulting to -100 if not present
        logger.warning_once(
            f"`labels` contains `pad_token_id` ({self.config.pad_token_id}) which will be masked with `ignore_index` ({ignore_index}). "
            "You have to mask out `pad_token_id` when preparing `labels`, this behavior will be removed in v.4.46.",
        )
        # Need input_ids for this comparison, ensure it's available or handle embeds case
        if input_ids is not None:
             labels = torch.where(input_ids == self.config.pad_token_id, ignore_index, labels)
        else:
             logger.warning("Cannot mask pad_token_id in labels based on input_ids as input_ids were not provided.")


    # Update causal mask (using the multimodal model's _update_causal_mask)
    causal_mask = self._update_causal_mask(
        attention_mask, token_type_ids, past_key_values, cache_position, inputs_embeds, is_training
    )
    # --- End: Setup code copied directly from original Gemma3ForConditionalGeneration.forward ---

    # Call the language model
    # Pass logits_to_keep down
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
        logits_to_keep=logits_to_keep, # Pass down
        **lm_kwargs,
    )

    # ---> Start: Patched Loss/Logits Section <---
    hidden_states = outputs[0] # Or outputs.last_hidden_state if return_dict=True
    loss = None
    logits = None # Initialize logits

    if _PATCH_OPTS is not None and _PATCH_OPTS.use_lce(labels, self.training):
        # Use LCE Loss
        assert labels is not None
        # apply_lce needs lm_head weights
        loss = apply_lce(hidden_states, self.language_model.lm_head.weight, labels, _PATCH_OPTS, **lm_kwargs)
        # Logits remain None
    else:
        # Use Standard Loss (or just calculate logits if no labels)
        # Calculate logits from the returned hidden_states.
        # NO slicing here - logits_to_keep was handled by the inner model call if applicable.
        logits = self.language_model.lm_head(hidden_states)
        # No softcapping in original Gemma3ForConditionalGeneration, so omitting here for consistency

        if labels is not None:
            # Calculate loss using the exact logic from original Gemma3ForConditionalGeneration.forward
            # Upcast logits to float for loss calculation stability
            logits_for_loss = logits.float()

            # Shift logits and labels for next token prediction
            shift_logits = logits_for_loss[..., :-1, :]
            shift_labels = labels[..., 1:].clone() # Clone to avoid modifying original labels

            # Use attention_mask (passed to forward) to mask loss
            loss_attention_mask = None
            if attention_mask is not None:
                # The attention mask provided to forward might be 2D or 4D.
                # We need a 1D mask corresponding to the sequence positions to keep for loss calculation.
                # If it's 2D (batch, seq_len), shift it.
                # If it's 4D, we need to infer the relevant mask. Usually, the 2D mask is more direct here.
                # Let's assume the original 2D attention_mask is appropriate.
                if attention_mask.dim() == 2:
                    # We need the mask for the *shifted* positions
                    loss_attention_mask = attention_mask[..., 1:].contiguous()
                elif attention_mask.dim() == 4:
                    # Difficult to reliably get the correct mask for loss from 4D.
                    # Fallback or require 2D mask for loss.
                    # Let's assume the user provides a 2D mask if loss is needed.
                     logger.warning_once("Cannot reliably determine loss mask from 4D attention_mask. Assuming no masking needed or relying on labels being -100.")
                     # If you *know* the structure, you might extract it, e.g., attention_mask[:, 0, 0, 1:]
                     # For now, we proceed without explicit mask derived from 4D attn mask, relying on label padding (-100)

            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            shift_labels = shift_labels.view(-1)

            # Apply the mask to the labels *before* calculating loss
            if loss_attention_mask is not None and loss_attention_mask.dim() == 2:
                # Flatten the mask and apply it to labels
                 active_loss = loss_attention_mask.view(-1) == 1
                 active_labels = torch.where(
                     active_loss, shift_labels, torch.tensor(loss_fct.ignore_index).type_as(shift_labels)
                 )
                 loss = loss_fct(shift_logits, active_labels)
            else:
                 # If no mask, calculate loss on all tokens (respecting ignore_index=-100 in labels)
                 loss = loss_fct(shift_logits, shift_labels)

    # ---> End: Patched Loss/Logits Section <---

    if not return_dict:
        # Ensure the output tuple structure matches original, potentially adding None for loss
        # The original conditional generation output tuple structure was (logits,) + outputs[1:]
        # If loss is calculated, it should be the first element.
        output = (logits,) + outputs[1:] # outputs[1:] contains past_key_values etc.
        return (loss,) + output if loss is not None else output

    # Use the specific multimodal output class
    return Gemma3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values, # Assumes outputs is ModelOutput when return_dict=True
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features, # Add image_hidden_states
        )

# --- patch_gemma3 function ---
def patch_gemma3(
    maybe_model: TransformersModelT | str | transformers.PretrainedConfig,
    patch_options: PatchOptions,
) -> TransformersModelT | None:
    global _PATCH_OPTS
    from transformers.models.gemma3 import modeling_gemma3 # Ensure full module is imported

    _PATCH_OPTS = patch_options

    if isinstance(maybe_model, transformers.PreTrainedModel):
        if isinstance(maybe_model, modeling_gemma3.Gemma3ForCausalLM):
             print("Patching Gemma3ForCausalLM forward method.")
             maybe_model.forward = MethodType(cce_forward, maybe_model)
        elif isinstance(maybe_model, modeling_gemma3.Gemma3ForConditionalGeneration):
             print("Patching Gemma3ForConditionalGeneration forward method.")
             maybe_model.forward = MethodType(multimodal_cce_forward, maybe_model)
        else:
             print(f"Model type {type(maybe_model)} not recognized for patching. Skipping.")
             # Or raise an error:
             # raise TypeError(f"Expected Gemma3ForCausalLM or Gemma3ForConditionalGeneration. Got {type(maybe_model)}.")
        return maybe_model
    else:
        # Apply patch globally if model instance isn't provided
        print("Patching Gemma3ForCausalLM and Gemma3ForConditionalGeneration classes globally.")
        modeling_gemma3.Gemma3ForCausalLM.forward = cce_forward
        modeling_gemma3.Gemma3ForConditionalGeneration.forward = multimodal_cce_forward
        # Return None as no instance was modified/returned
        return None
