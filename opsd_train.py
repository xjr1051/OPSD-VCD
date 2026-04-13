import os
import wandb

from datasets import load_dataset
from transformers import AutoProcessor, AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig
from opsd_trainer import OPSDTrainer
from dataclasses import dataclass, field

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with Thinking Machines loss option."""

    use_tinker_loss: bool = field(
        default=False,
        metadata={
            "help": "Use Thinking Machines style on-policy reverse KL loss instead of GKD's full-vocab JSD loss. "
            "This is much more memory efficient (O(1) vs O(vocab_size) per token)."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy (step 0) as a fixed teacher. Only works with use_peft=True. "
            "The teacher will use the base model without LoRA adapters, while the student updates."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the generated text so far. "
            "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    reason_first: bool = field(
        default=False,
        metadata={
            "help": "Let the teacher model first rationalize (generate rationalization explictly) about the given reasoning first then act as teacher."
        },
    )
    top_k_loss: int = field(
        default=0,
        metadata={
            "help": "Restrict the JSD loss to only the top-k tokens of the teacher distribution. Both student and "
            "teacher distributions are renormalized over these k tokens before computing JSD. "
            "Set to 0 (default) to use the full vocabulary."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={
            "help": "Clip the JSD loss for each token to a maximum value. This can improve stability by preventing "
            "extremely high-loss stylistic tokens from dominating the training signal. Set to 0 for no clipping."
        },
    )

    use_ema_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use an exponential moving average (EMA) of student weights as the teacher. "
            "The EMA teacher is a smoothly-lagged version of the student, avoiding the teacher "
            "collapsing to the current policy (dynamic) or staying frozen (fixed_teacher). "
            "Mutually exclusive with fixed_teacher."
        },
    )
    ema_decay: float = field(
        default=0.999,
        metadata={
            "help": "EMA decay factor. Higher values make the teacher change more slowly. "
            "Typical range: 0.99–0.9999. Only used when use_ema_teacher=True."
        },
    )
    use_vcd_opsd: bool = field(
        default=False,
        metadata={
            "help": "Enable visual OPSD training branch. This follows OPSD on-policy training: trajectory is sampled from student condition, teacher provides privileged/stronger visual supervision."
        },
    )
    vcd_alpha: float = field(
        default=1.0,
        metadata={
            "help": "Contrastive strength alpha for teacher logits mixing: z_vcd = (1+alpha)*z_good - alpha*z_bad."
        },
    )
    good_view_field: str = field(
        default="problem_good_view",
        metadata={
            "help": "Legacy dataset field for teacher/factual view. Used as a fallback when pair-based view fields are missing."
        },
    )
    bad_view_field: str = field(
        default="problem_bad_view",
        metadata={
            "help": "Legacy dataset field for student/perturbed view. Used as a fallback when pair-based view fields are missing."
        },
    )
    view_pairs: str = field(
        default="clean-noise,mask-clean",
        metadata={
            "help": "Teacher-student view pairs for VLM distillation, e.g. 'clean-noise,mask-clean'. Format: teacher-student."
        },
    )
    view_field_prefix: str = field(
        default="problem_",
        metadata={
            "help": "Prefix used to resolve pair tags to dataset fields. Example: prefix='problem_' and pair clean-noise map to problem_clean/problem_noise."
        },
    )
    pair_sampling_strategy: str = field(
        default="random",
        metadata={
            "help": "How to choose a pair when multiple teacher-student pairs are configured and available. Options: random, first, round_robin."
        },
    )
    dataset_name: str = field(
        default="siyanzhao/Openthoughts_math_30k_opsd",
        metadata={
            "help": "Dataset name or local path to training data. Replace this with your VLM view-pair dataset."
        },
    )
    dataset_config_name: str = field(
        default=None,
        metadata={
            "help": "Optional dataset config name passed to load_dataset."
        },
    )
    train_split: str = field(
        default="train",
        metadata={
            "help": "Dataset split used for training."
        },
    )
    problem_field: str = field(
        default="problem",
        metadata={
            "help": "Field name used as the base prompt/question."
        },
    )
    solution_field: str = field(
        default="solution",
        metadata={
            "help": "Field name used as reference rationale/answer for teacher prompting."
        },
    )
    use_image_perturbation_pairs: bool = field(
        default=False,
        metadata={
            "help": "Build teacher-student view pairs from online image perturbations in the collator. Requires a multimodal processor/model."
        },
    )
    image_field: str = field(
        default="image",
        metadata={
            "help": "Image column name used to build perturbation-based view pairs."
        },
    )
    image_token: str = field(
        default="<image>",
        metadata={
            "help": "Image placeholder token inserted into textual prompts for multimodal chat templates."
        },
    )
    noise_std: float = field(
        default=25.0,
        metadata={
            "help": "Gaussian noise std for the noise view transform."
        },
    )
    mask_ratio: float = field(
        default=0.25,
        metadata={
            "help": "Rectangle area ratio for the mask view transform."
        },
    )
    blur_radius: float = field(
        default=2.0,
        metadata={
            "help": "Gaussian blur radius for the blur view transform."
        },
    )
    use_multimodal_processor: bool = field(
        default=False,
        metadata={
            "help": "Load AutoProcessor instead of AutoTokenizer. Required for image perturbation pair training."
        },
    )
    use_privileged_visual_teacher: bool = field(
        default=False,
        metadata={
            "help": "Enable privileged visual teacher p_T(.|x,v,z*): teacher prompt includes privileged visual evidence field."
        },
    )
    use_single_visual_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use pure visual single-teacher mode in VCD-OPSD: teacher uses only the good/factual view branch, student uses the weak view branch."
        },
    )
    privileged_visual_field: str = field(
        default="privileged_visual_evidence",
        metadata={
            "help": "Dataset field name for privileged visual evidence z* used only by teacher."
        },
    )


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    is_distributed_launch = world_size > 1 or local_rank >= 0
    if is_distributed_launch and getattr(training_args, "gradient_checkpointing", False):
        print(
            "[launch] Disabling gradient_checkpointing in distributed mode "
            "to avoid DDP reentrant-backward conflicts."
        )
        training_args.gradient_checkpointing = False

    ################
    # WandB Run Name & Output Directory
    ################
    # Format learning rate (e.g., 2e-4 -> "2e-4" or 0.0002 -> "2e-4")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = world_size

    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    # Use custom run_config if provided, otherwise generate automatic name
    if script_args.run_config:
        full_wandb_run_config = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        # Append run_config to output_dir if it doesn't already end with it
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path

            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        # Extract model name from path (e.g., "Qwen3-1.7B" from "/home/siyanzhao/models/Qwen3-1.7B")
        model_name = model_args.model_name_or_path.split("/")[-1]

        # Create concise run name
        full_wandb_run_config = (
            f"opsd_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
        )

        # Add fixed_teacher to wandb name if enabled
        if script_args.fixed_teacher:
            full_wandb_run_config += "_fixteach"

    # Print configuration info
    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"{'='*80}\n")

    ################
    # WandB Initialization
    ################
    # Validate fixed_teacher argument
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError(
            "fixed_teacher=True requires use_peft=True. As the fixed teacher is implemented by disabling LoRA adapters."
        )

    if script_args.use_vcd_opsd and script_args.reason_first:
        raise ValueError("use_vcd_opsd=True and reason_first=True are mutually exclusive in this visual OPSD setup.")

    if script_args.use_image_perturbation_pairs and not script_args.use_multimodal_processor:
        raise ValueError(
            "use_image_perturbation_pairs=True requires use_multimodal_processor=True."
        )

    if script_args.use_single_visual_teacher and not script_args.use_vcd_opsd:
        raise ValueError("use_single_visual_teacher=True requires use_vcd_opsd=True.")

    if script_args.use_single_visual_teacher and script_args.use_privileged_visual_teacher:
        raise ValueError(
            "use_single_visual_teacher=True and use_privileged_visual_teacher=True are mutually exclusive."
        )

    # Only initialize wandb on main process (LOCAL_RANK 0 or not set)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=full_wandb_run_config,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "lmbda": training_args.lmbda,
                "max_length": training_args.max_length,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "use_tinker_loss": script_args.use_tinker_loss,
                "fixed_teacher": script_args.fixed_teacher,
                "top_k_loss": script_args.top_k_loss if script_args.top_k_loss > 0 else None,
                "use_ema_teacher": script_args.use_ema_teacher,
                "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
                "use_vcd_opsd": script_args.use_vcd_opsd,
                "vcd_alpha": script_args.vcd_alpha if script_args.use_vcd_opsd else None,
                "good_view_field": script_args.good_view_field if script_args.use_vcd_opsd else None,
                "bad_view_field": script_args.bad_view_field if script_args.use_vcd_opsd else None,
                "view_pairs": script_args.view_pairs if script_args.use_vcd_opsd else None,
                "view_field_prefix": script_args.view_field_prefix if script_args.use_vcd_opsd else None,
                "pair_sampling_strategy": script_args.pair_sampling_strategy if script_args.use_vcd_opsd else None,
                "dataset_name": script_args.dataset_name,
                "dataset_config_name": script_args.dataset_config_name,
                "train_split": script_args.train_split,
                "problem_field": script_args.problem_field,
                "solution_field": script_args.solution_field,
                "use_image_perturbation_pairs": script_args.use_image_perturbation_pairs,
                "image_field": script_args.image_field if script_args.use_image_perturbation_pairs else None,
                "image_token": script_args.image_token if script_args.use_image_perturbation_pairs else None,
                "noise_std": script_args.noise_std if script_args.use_image_perturbation_pairs else None,
                "mask_ratio": script_args.mask_ratio if script_args.use_image_perturbation_pairs else None,
                "blur_radius": script_args.blur_radius if script_args.use_image_perturbation_pairs else None,
                "use_multimodal_processor": script_args.use_multimodal_processor,
                "use_privileged_visual_teacher": script_args.use_privileged_visual_teacher,
                "use_single_visual_teacher": script_args.use_single_visual_teacher,
                "privileged_visual_field": script_args.privileged_visual_field if script_args.use_privileged_visual_teacher else None,
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    # Determine dtype - handle both old torch_dtype and new dtype attributes
    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    print(f"\n{'='*80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    print(
        f"[launch] WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}, "
        f"quantization_config={'set' if quantization_config is not None else 'none'}"
    )

    if is_distributed_launch and quantization_config is not None:
        # In distributed mode, the k-bit loading path may route through a
        # device_map='auto' branch, which is incompatible with DDP/Accelerate.
        # For small/medium models this full-precision load is typically stable.
        print(
            "[launch] Detected distributed run with k-bit quantization config; "
            "disabling quantization to avoid device_map=auto incompatibility."
        )
        quantization_config = None

    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    if is_distributed_launch:
        # TRL's create_model_from_path defaults to device_map='auto' when this
        # key is absent. In DDP that can trigger Accelerate's device-map guard.
        # We set it explicitly to None so each DDP worker keeps a single-device
        # placement path and avoids pre-sharded multi-device model states.
        model_kwargs["device_map"] = None

    training_args.model_init_kwargs = model_kwargs

    # No separate teacher model needed - we use the same model with privileged info

    # `AutoProcessor` is required for multimodal models because it bundles
    # tokenizer + image/video preprocessing in one object.
    # For text-only runs we keep using `AutoTokenizer` to avoid introducing
    # unnecessary processor-side behavior differences.
    if script_args.use_multimodal_processor:
        processing_class = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            use_fast=False,
        )
        # Some processors expose `pad_token_*` on an inner tokenizer only.
        # We normalize padding behavior here so generation/collation code can
        # reliably use left-padding and a valid pad token id.
        if hasattr(processing_class, "tokenizer"):
            processing_class.tokenizer.padding_side = "left"
            if processing_class.tokenizer.pad_token is None:
                processing_class.tokenizer.pad_token = processing_class.tokenizer.eos_token
    else:
        processing_class = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            padding_side="left",
        )
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token

    ################
    # Dataset
    ################
    # Load the math dataset with ground truth solutions
    ################
    # Training
    ################
    # Add presence_penalty to training_args so it can be accessed in the trainer
    training_args.presence_penalty = script_args.presence_penalty

    # Keep dataset loading fully configurable so the same training script can
    # be reused across different VCD/OPSD datasets without code changes.
    if script_args.dataset_config_name:
        print(
            f"[stage] loading dataset name={script_args.dataset_name}, config={script_args.dataset_config_name}, split={script_args.train_split}"
        )
        dataset = load_dataset(script_args.dataset_name, script_args.dataset_config_name)
    else:
        print(
            f"[stage] loading dataset name={script_args.dataset_name}, split={script_args.train_split}"
        )
        dataset = load_dataset(script_args.dataset_name)
    train_dataset = dataset[script_args.train_split]
    print(f"[stage] dataset loaded: split={script_args.train_split}, size={len(train_dataset)}")

    # Pass all view/pair and multimodal options to the trainer so batching,
    # generation and loss computation share one consistent configuration source.
    print("[stage] initializing OPSDTrainer")
    trainer = OPSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=processing_class,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        use_vcd_opsd=script_args.use_vcd_opsd,
        vcd_alpha=script_args.vcd_alpha,
        good_view_field=script_args.good_view_field,
        bad_view_field=script_args.bad_view_field,
        view_pairs=script_args.view_pairs,
        view_field_prefix=script_args.view_field_prefix,
        pair_sampling_strategy=script_args.pair_sampling_strategy,
        problem_field=script_args.problem_field,
        solution_field=script_args.solution_field,
        use_image_perturbation_pairs=script_args.use_image_perturbation_pairs,
        image_field=script_args.image_field,
        image_token=script_args.image_token,
        noise_std=script_args.noise_std,
        mask_ratio=script_args.mask_ratio,
        blur_radius=script_args.blur_radius,
        use_privileged_visual_teacher=script_args.use_privileged_visual_teacher,
        use_single_visual_teacher=script_args.use_single_visual_teacher,
        privileged_visual_field=script_args.privileged_visual_field,
    )
    print("[stage] OPSDTrainer initialized")

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    print("[stage] entering trainer.train()")
    trainer.train()

    trainer.save_model(training_args.output_dir)
