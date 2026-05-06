import copy
import sys
import types
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from transformers.generation.utils import GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput, GenerationMixin

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from image_perturbation import (
    ImagePerturbationConfig,
    add_diffusion_noise_tensor,
    apply_image_perturbation,
    normalize_perturbation_pair,
)


@dataclass
class VCDConfig(ImagePerturbationConfig):
    alpha: float = 1.0
    beta: float = 0.1
    view_pair: str = "clean-noise"
    start_step: int = 0
    max_steps: Optional[int] = None
    min_keep: int = 0


_ORIGINAL_SAMPLE = GenerationMixin._sample
_PATCH_INSTALLED = False
_VCD_MODEL_KWARG_KEYS = {
    "images_cd",
    "pixel_values_cd",
    "pixel_values_videos_cd",
    "image_grid_thw_cd",
    "video_grid_thw_cd",
    "second_per_grid_ts_cd",
    "cd_alpha",
    "cd_beta",
    "vcd_start_step",
    "vcd_max_steps",
    "vcd_min_keep",
}


def resolve_view_pair(pair: str) -> Tuple[str, str]:
    normalized = pair.strip().lower().replace(">", "-").replace(":", "-")
    parts = [part.strip() for part in normalized.split("-") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid VCD view pair: {pair}")
    return normalize_perturbation_pair((parts[0], parts[1]))


def should_use_vcd(args) -> bool:
    return bool(getattr(args, "use_vcd_decoding", False))


def add_vcd_args(parser):
    parser.add_argument("--use-vcd-decoding", action="store_true", help="Enable inference-time VCD decoding.")
    parser.add_argument("--vcd-alpha", type=float, default=1.0, help="Contrastive strength alpha.")
    parser.add_argument(
        "--vcd-beta",
        type=float,
        default=0.1,
        help="Adaptive plausibility constraint threshold beta in [0, 1].",
    )
    parser.add_argument(
        "--vcd-gamma",
        type=float,
        default=0.1,
        help="Legacy image-space Gaussian schedule strength for non-clean-noise views.",
    )
    parser.add_argument(
        "--vcd-noise-steps",
        type=int,
        default=500,
        help="Diffusion step t used by clean-noise VCD.",
    )
    parser.add_argument(
        "--vcd-view-pair",
        type=str,
        default="clean-noise",
        help="Good-bad image pair, e.g. clean-noise or fgmask-clean.",
    )
    parser.add_argument(
        "--vcd-max-steps",
        type=int,
        default=None,
        help="Only apply VCD to the first K generated tokens, then fall back to normal decoding.",
    )
    parser.add_argument(
        "--vcd-start-step",
        type=int,
        default=0,
        help="Delay VCD until K generated tokens have already been produced.",
    )
    parser.add_argument(
        "--vcd-min-keep",
        type=int,
        default=0,
        help="Always keep at least K clean-branch candidates after the beta plausibility filter.",
    )
    parser.add_argument("--vcd-noise-std", type=float, default=25.0)
    parser.add_argument("--vcd-mask-ratio", type=float, default=0.25)
    parser.add_argument("--vcd-mask-min-ratio", type=float, default=None)
    parser.add_argument("--vcd-mask-max-ratio", type=float, default=None)
    parser.add_argument("--vcd-mask-count", type=int, default=1)
    parser.add_argument("--vcd-blur-radius", type=float, default=2.0)
    parser.add_argument("--vcd-fg-mask-keep-ratio", type=float, default=0.35)
    parser.add_argument("--vcd-fg-mask-center-bias", type=float, default=0.15)


def config_from_args(args) -> VCDConfig:
    return VCDConfig(
        alpha=float(getattr(args, "vcd_alpha", 1.0)),
        beta=float(getattr(args, "vcd_beta", 0.1)),
        gamma=float(getattr(args, "vcd_gamma", 0.1)),
        noise_steps=int(getattr(args, "vcd_noise_steps", 500)),
        view_pair=getattr(args, "vcd_view_pair", "clean-noise"),
        start_step=int(getattr(args, "vcd_start_step", 0)),
        max_steps=getattr(args, "vcd_max_steps", None),
        min_keep=int(getattr(args, "vcd_min_keep", 0)),
        noise_std=float(getattr(args, "vcd_noise_std", 25.0)),
        mask_ratio=float(getattr(args, "vcd_mask_ratio", 0.25)),
        mask_min_ratio=getattr(args, "vcd_mask_min_ratio", None),
        mask_max_ratio=getattr(args, "vcd_mask_max_ratio", None),
        mask_count=int(getattr(args, "vcd_mask_count", 1)),
        blur_radius=float(getattr(args, "vcd_blur_radius", 2.0)),
        fg_mask_keep_ratio=float(getattr(args, "vcd_fg_mask_keep_ratio", 0.35)),
        fg_mask_center_bias=float(getattr(args, "vcd_fg_mask_center_bias", 0.15)),
    )


def move_to_device(batch: dict):
    if torch.cuda.is_available():
        return {k: v.to("cuda") if hasattr(v, "to") else v for k, v in batch.items()}
    return {k: v.to("cpu") if hasattr(v, "to") else v for k, v in batch.items()}


def clone_batch(batch: dict) -> dict:
    cloned = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def _clone_optional_tensor(value):
    if torch.is_tensor(value):
        return value.clone()
    return value


def _get_rope_deltas_state(model):
    rope_owner = getattr(model, "model", None)
    if rope_owner is None or not hasattr(rope_owner, "rope_deltas"):
        return None
    return _clone_optional_tensor(rope_owner.rope_deltas)


def _set_rope_deltas_state(model, rope_deltas):
    rope_owner = getattr(model, "model", None)
    if rope_owner is not None and hasattr(rope_owner, "rope_deltas"):
        rope_owner.rope_deltas = _clone_optional_tensor(rope_deltas)


def build_generate_kwargs(processor, args) -> dict:
    do_sample = args.temperature is not None and args.temperature > 0
    tokenizer = getattr(processor, "tokenizer", processor)

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": 1,
        "length_penalty": 1.0,
        "num_return_sequences": 1,
        "use_cache": True,
        "do_sample": do_sample,
    }

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        generate_kwargs["eos_token_id"] = eos_token_id

    if do_sample:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p
        if args.top_k is not None:
            generate_kwargs["top_k"] = args.top_k

    return generate_kwargs


def _uses_contrastive_kwargs(model_kwargs: dict) -> bool:
    return (
        model_kwargs.get("images_cd") is not None
        or model_kwargs.get("pixel_values_cd") is not None
        or model_kwargs.get("pixel_values_videos_cd") is not None
    )


def _apply_plausibility_constraint(
    clean_logits: torch.Tensor,
    combined_scores: torch.Tensor,
    cd_beta: float,
    min_keep: int = 0,
) -> torch.Tensor:
    cutoff = torch.log(torch.tensor(cd_beta, device=clean_logits.device, dtype=clean_logits.dtype))
    cutoff = cutoff + clean_logits.max(dim=-1, keepdim=True).values
    keep_mask = clean_logits >= cutoff

    # Qwen2.5-VL can produce extremely sharp first-token logits on open-ended
    # prompts. In that case, the paper-style beta filter may leave only a
    # single candidate (often "The"), which makes long-form generations
    # collapse immediately into repetitive prefixes. We keep a small top-k
    # fallback from the clean branch to prevent this pathological one-token
    # candidate set while preserving the original plausibility rule whenever it
    # is already permissive enough.
    min_keep = int(min_keep or 0)
    if min_keep > 0:
        k = min(min_keep, clean_logits.shape[-1])
        topk_idx = torch.topk(clean_logits, k=k, dim=-1).indices
        keep_mask.scatter_(1, topk_idx, True)

    return combined_scores.masked_fill(~keep_mask, -float("inf"))


def _prepare_inputs_for_generation_cd_qwen25vl(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    **kwargs,
):
    cd_kwargs = dict(kwargs)
    pixel_values_cd = cd_kwargs.pop("pixel_values_cd", None)
    pixel_values_videos_cd = cd_kwargs.pop("pixel_values_videos_cd", None)
    image_grid_thw_cd = cd_kwargs.pop("image_grid_thw_cd", None)
    video_grid_thw_cd = cd_kwargs.pop("video_grid_thw_cd", None)
    second_per_grid_ts_cd = cd_kwargs.pop("second_per_grid_ts_cd", None)
    cd_kwargs.pop("pixel_values", None)
    cd_kwargs.pop("pixel_values_videos", None)
    cd_kwargs.pop("image_grid_thw", None)
    cd_kwargs.pop("video_grid_thw", None)
    cd_kwargs.pop("second_per_grid_ts", None)
    cd_kwargs.pop("images_cd", None)
    cd_kwargs.pop("cd_alpha", None)
    cd_kwargs.pop("cd_beta", None)

    return self.prepare_inputs_for_generation(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        position_ids=position_ids,
        use_cache=use_cache,
        pixel_values=pixel_values_cd,
        pixel_values_videos=pixel_values_videos_cd,
        image_grid_thw=image_grid_thw_cd,
        video_grid_thw=video_grid_thw_cd,
        second_per_grid_ts=second_per_grid_ts_cd,
        **cd_kwargs,
    )


def _validate_model_kwargs_with_vcd(self, model_kwargs):
    filtered = dict(model_kwargs)
    for key in _VCD_MODEL_KWARG_KEYS:
        filtered.pop(key, None)
    return self._vcd_original_validate_model_kwargs(filtered)


def _extract_cd_static_kwargs(model_kwargs: dict) -> dict:
    return {key: model_kwargs.get(key) for key in _VCD_MODEL_KWARG_KEYS if key in model_kwargs}


def _bootstrap_cd_model_kwargs(self, input_ids: torch.LongTensor, clean_model_kwargs: dict, cd_static_kwargs: dict):
    bootstrap_kwargs = {}
    for key, value in clean_model_kwargs.items():
        if key in {"past_key_values", "cache_position"}:
            continue
        bootstrap_kwargs[key] = value

    bootstrap_kwargs.update(cd_static_kwargs)
    return self._get_initial_cache_position(input_ids.shape[1], input_ids.device, bootstrap_kwargs)


def _sample_with_vcd(
    self,
    input_ids: torch.LongTensor,
    logits_processor,
    stopping_criteria,
    generation_config,
    synced_gpus: bool = False,
    streamer=None,
    **model_kwargs,
):
    if not _uses_contrastive_kwargs(model_kwargs):
        return _ORIGINAL_SAMPLE(
            self,
            input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    if generation_config.prefill_chunk_size is not None:
        raise NotImplementedError("VCD decoding does not support prefill_chunk_size in this patch.")

    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    batch_size, cur_len = input_ids.shape[:2]
    start_len = cur_len
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
    model_kwargs_cd = copy.copy(model_kwargs)
    cd_static_kwargs = _extract_cd_static_kwargs(model_kwargs)
    clean_rope_deltas = None
    cd_rope_deltas = None
    cd_branch_bootstrapped = False
    vcd_max_steps = model_kwargs.get("vcd_max_steps")
    vcd_start_step = int(model_kwargs.get("vcd_start_step") or 0)
    vcd_min_keep = int(model_kwargs.get("vcd_min_keep") or 0)

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        if synced_gpus:
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0, device=input_ids.device)
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            if this_peer_finished_flag.item() == 0.0:
                break

        # Qwen2.5-VL caches `rope_deltas` on the shared model object during
        # prefill. Clean and contrastive branches must keep separate caches, or
        # the CD prefill overwrites the clean branch state and later decoding
        # steps collapse into repetitive junk such as "TheThe".
        generated_steps = cur_len - start_len
        end_step = None if vcd_max_steps is None else (vcd_start_step + int(vcd_max_steps))
        use_cd_this_step = _uses_contrastive_kwargs(model_kwargs) and generated_steps >= vcd_start_step and (
            end_step is None or generated_steps < end_step
        )
        cd_bootstrap_source_kwargs = copy.copy(model_kwargs)

        _set_rope_deltas_state(self, clean_rope_deltas)
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        clean_rope_deltas = _get_rope_deltas_state(self)
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        outputs_cd = None
        if use_cd_this_step:
            if not cd_branch_bootstrapped:
                # When VCD is delayed (vcd_start_step > 0), the contrastive branch
                # has not been decoding in lockstep with the clean branch. Rebuild
                # its generation kwargs from the current full prefix so the first CD
                # step sees a valid attention mask / cache state instead of stale
                # tensors from the initial prompt length.
                model_kwargs_cd = _bootstrap_cd_model_kwargs(
                    self,
                    input_ids,
                    cd_bootstrap_source_kwargs,
                    cd_static_kwargs,
                )
                cd_rope_deltas = None
                cd_branch_bootstrapped = True
            _set_rope_deltas_state(self, cd_rope_deltas)
            model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, **model_kwargs_cd)
            outputs_cd = self(
                **model_inputs_cd,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            cd_rope_deltas = _get_rope_deltas_state(self)
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd,
                model_kwargs_cd,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            next_token_logits_cd = outputs_cd.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            cd_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 1.0
            cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1
            next_token_scores = (1.0 + cd_alpha) * next_token_logits - cd_alpha * next_token_logits_cd
            next_token_scores = _apply_plausibility_constraint(
                clean_logits=next_token_logits,
                combined_scores=next_token_scores,
                cd_beta=cd_beta,
                min_keep=vcd_min_keep,
            )
            next_token_scores = logits_processor(input_ids, next_token_scores)
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)

        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        del outputs
        if outputs_cd is not None:
            del outputs_cd

        _set_rope_deltas_state(self, clean_rope_deltas)

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        return GenerateDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            logits=raw_logits,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            past_key_values=model_kwargs.get("past_key_values"),
        )
    return input_ids


def evolve_vcd_sampling(model=None):
    global _PATCH_INSTALLED
    if not _PATCH_INSTALLED:
        GenerationMixin._sample = _sample_with_vcd
        GenerationMixin.sample = _sample_with_vcd
        _PATCH_INSTALLED = True

    if model is not None and not hasattr(model, "prepare_inputs_for_generation_cd"):
        model.prepare_inputs_for_generation_cd = types.MethodType(_prepare_inputs_for_generation_cd_qwen25vl, model)
    if model is not None and not hasattr(model, "_vcd_original_validate_model_kwargs"):
        model._vcd_original_validate_model_kwargs = model._validate_model_kwargs
        model._validate_model_kwargs = types.MethodType(_validate_model_kwargs_with_vcd, model)


def prepare_vcd_inputs(processor, texts: Sequence[str], images: Sequence[Image.Image], args):
    if not images:
        raise ValueError("VCD decoding requires image inputs.")

    cfg = config_from_args(args)
    good_tag, bad_tag = resolve_view_pair(cfg.view_pair)

    # We keep the clean branch untouched and only build a second distorted branch,
    # which mirrors the official VCD setup.
    good_images = [apply_image_perturbation(img, good_tag, cfg) for img in images]
    good_inputs = processor(text=list(texts), images=good_images, return_tensors="pt", padding=True)

    use_tensor_noise = good_tag == "clean" and bad_tag == "noise" and "pixel_values" in good_inputs
    if use_tensor_noise:
        bad_inputs = clone_batch(good_inputs)
    else:
        bad_images = [apply_image_perturbation(img, bad_tag, cfg) for img in images]
        bad_inputs = processor(text=list(texts), images=bad_images, return_tensors="pt", padding=True)

    return cfg, good_inputs, bad_inputs, use_tensor_noise


def vcd_generate(
    model,
    processor,
    texts: Sequence[str],
    images: Sequence[Image.Image],
    args,
) -> List[str]:
    evolve_vcd_sampling(model)

    cfg, good_inputs, bad_inputs, use_tensor_noise = prepare_vcd_inputs(processor, texts, images, args)

    input_token_len = good_inputs["input_ids"].shape[1]
    good_inputs = move_to_device(good_inputs)
    bad_inputs = move_to_device(bad_inputs)

    if use_tensor_noise and "pixel_values" in bad_inputs:
        bad_inputs["pixel_values"] = add_diffusion_noise_tensor(bad_inputs["pixel_values"], cfg.noise_steps)

    generate_kwargs = build_generate_kwargs(processor, args)
    generate_kwargs.update(
        {
            "pixel_values_cd": bad_inputs.get("pixel_values"),
            "pixel_values_videos_cd": bad_inputs.get("pixel_values_videos"),
            "image_grid_thw_cd": bad_inputs.get("image_grid_thw"),
            "video_grid_thw_cd": bad_inputs.get("video_grid_thw"),
            "second_per_grid_ts_cd": bad_inputs.get("second_per_grid_ts"),
            "cd_alpha": cfg.alpha,
            "cd_beta": cfg.beta,
            "vcd_start_step": cfg.start_step,
            "vcd_max_steps": cfg.max_steps,
            "vcd_min_keep": cfg.min_keep,
        }
    )

    with torch.inference_mode():
        output_ids = model.generate(**good_inputs, **generate_kwargs)

    outputs = processor.batch_decode(
        output_ids[:, input_token_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return [o.strip() for o in outputs]
