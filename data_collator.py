import torch
import random
import numpy as np
import os
from PIL import Image
from PIL import ImageFile
from io import BytesIO
from urllib.request import urlopen
from image_perturbation import (
    ImagePerturbationConfig,
    add_diffusion_noise_tensor,
    apply_image_perturbation,
    normalize_perturbation_pair,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SelfDistillationDataCollator:
    """
    Data collator for self-distillation that creates both student and teacher inputs.

    Student: sees only the problem (with chat template)
    Teacher: sees problem + solution + transition prompt (with chat template)

    To enable batch-level operations (like original GKD), we pad prompts to the same length
    within each batch, and track the actual (unpadded) prompt lengths for loss masking.

    In visual OPSD mode, we keep OPSD's on-policy setup:
    - student prompt uses bad/weak view
    - teacher builds two branches (good and bad view) for contrastive supervision
    """

    def __init__(
        self,
        tokenizer,
        max_length=2048,
        reason_first=True,
        enable_vcd_opsd=False,
        good_view_field="problem_good_view",
        bad_view_field="problem_bad_view",
        view_pairs="clean-noise,mask-clean",
        view_field_prefix="problem_",
        pair_sampling_strategy="random",
        problem_field="problem",
        solution_field="solution",
        use_image_perturbation_pairs=False,
        image_field="image",
        image_token="<image>",
        noise_std=25.0,
        noise_steps=500,
        use_tensor_diffusion_noise=False,
        mask_ratio=0.25,
        mask_min_ratio=None,
        mask_max_ratio=None,
        mask_count=1,
        blur_radius=2.0,
        fg_mask_keep_ratio=0.35,
        fg_mask_center_bias=0.15,
        object_bbox_field="",
        target_object_field="",
        object_bbox_label_field="",
        use_privileged_visual_teacher=False,
        use_single_visual_teacher=False,
        privileged_visual_field="privileged_visual_evidence",
        max_image_side=768,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reason_first = reason_first
        self.enable_vcd_opsd = enable_vcd_opsd
        self.good_view_field = good_view_field
        self.bad_view_field = bad_view_field
        self.view_pairs = self._parse_view_pairs(view_pairs)
        self.view_field_prefix = view_field_prefix
        self.pair_sampling_strategy = pair_sampling_strategy
        self.problem_field = problem_field
        self.solution_field = solution_field
        self.use_image_perturbation_pairs = use_image_perturbation_pairs
        self.image_field = image_field
        self.image_token = image_token
        self.noise_std = noise_std
        self.noise_steps = noise_steps
        self.use_tensor_diffusion_noise = use_tensor_diffusion_noise
        self.mask_ratio = mask_ratio
        self.mask_min_ratio = mask_min_ratio
        self.mask_max_ratio = mask_max_ratio
        self.mask_count = mask_count
        self.blur_radius = blur_radius
        self.fg_mask_keep_ratio = fg_mask_keep_ratio
        self.fg_mask_center_bias = fg_mask_center_bias
        self.object_bbox_field = object_bbox_field
        self.target_object_field = target_object_field
        self.object_bbox_label_field = object_bbox_label_field
        self.use_privileged_visual_teacher = use_privileged_visual_teacher
        self.use_single_visual_teacher = use_single_visual_teacher
        self.privileged_visual_field = privileged_visual_field
        env_max_image_side = os.environ.get("OPSD_MAX_IMAGE_SIDE")
        if env_max_image_side is not None:
            try:
                self.max_image_side = int(env_max_image_side)
            except ValueError:
                self.max_image_side = max_image_side
        else:
            self.max_image_side = max_image_side

        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if self.pad_token_id is None and hasattr(self.tokenizer, "tokenizer"):
            self.pad_token_id = getattr(self.tokenizer.tokenizer, "pad_token_id", None)
        if self.pad_token_id is None:
            raise ValueError("A valid pad_token_id is required for batching in SelfDistillationDataCollator.")

        if self.use_image_perturbation_pairs and not (
            hasattr(self.tokenizer, "image_processor") or hasattr(self.tokenizer, "feature_extractor")
        ):
            raise ValueError(
                "use_image_perturbation_pairs=True requires a multimodal processor with image support, "
                "but a text-only tokenizer was provided."
            )

        if self.use_image_perturbation_pairs:
            # Keep one fixed perturbation pair per run.
            self.view_pairs = [self._normalize_perturbation_pair(pair) for pair in self.view_pairs]
            if len(self.view_pairs) != 1:
                raise ValueError(
                    "use_image_perturbation_pairs=True now requires exactly one view pair "
                    "(for example: clean-noise or clean-mask)."
                )

        if self.reason_first and self.enable_vcd_opsd:
            raise ValueError(
                "reason_first and enable_vcd_opsd cannot be enabled together in this baseline skeleton."
            )

        if self.use_single_visual_teacher and not self.enable_vcd_opsd:
            raise ValueError("use_single_visual_teacher=True requires enable_vcd_opsd=True.")

        if self.use_single_visual_teacher and self.use_privileged_visual_teacher:
            raise ValueError(
                "use_single_visual_teacher=True and use_privileged_visual_teacher=True are mutually exclusive."
            )

        self.perturbation_cfg = ImagePerturbationConfig(
            noise_std=self.noise_std,
            noise_steps=self.noise_steps,
            mask_ratio=self.mask_ratio,
            mask_min_ratio=self.mask_min_ratio,
            mask_max_ratio=self.mask_max_ratio,
            mask_count=self.mask_count,
            blur_radius=self.blur_radius,
            fg_mask_keep_ratio=self.fg_mask_keep_ratio,
            fg_mask_center_bias=self.fg_mask_center_bias,
            object_bbox_field=self.object_bbox_field,
            target_object_field=self.target_object_field,
            object_bbox_label_field=self.object_bbox_label_field,
        )

        # Prompt for reasoning about the solution before teaching
        self.reason_first_prompt = (
            "\n\nThe reference reasoning above arrives at the correct answer. "
            "Please analyze this solution and explain the key reasoning steps and problem-solving strategies employed. "
            "Do NOT use <think> tags. Do NOT derive your own solution. "
            "Simply analyze and explain the reference solution provided above.\n"
        )
        # Prompt for transitioning to teaching mode after reasoning
        self.transition_prompt = (
            "\n\nAfter reading the reference solution above, make sure you truly understand "
            "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
            "own words and independent reasoning, derive the same final answer to the problem above. "
            "Think step by step, explore different approaches, and don't be afraid to backtrack "
            "or reconsider if something doesn't work out:\n"
        )

        # Set padding side explicitly for consistency. For AutoProcessor-based
        # multimodal runs, padding_side typically lives on the inner tokenizer.
        padding_owner = self.tokenizer
        if not hasattr(padding_owner, "padding_side") and hasattr(self.tokenizer, "tokenizer"):
            padding_owner = self.tokenizer.tokenizer

        if hasattr(padding_owner, "padding_side"):
            print(f"[DataCollator] Original padding_side: {padding_owner.padding_side}")
            # Decoder-only generation is more efficient and numerically safer with left padding.
            padding_owner.padding_side = "left"
            print(f"[DataCollator] Set padding_side to: {padding_owner.padding_side}")
        else:
            print("[DataCollator] padding_side not available on processor/tokenizer; keep backend default.")
        print(f"[DataCollator] Reason first mode: {self.reason_first}")
        print(f"[DataCollator] VCD-OPSD mode: {self.enable_vcd_opsd}")
        if self.enable_vcd_opsd:
            print(
                f"[DataCollator] VCD view fields: good={self.good_view_field}, bad={self.bad_view_field}"
            )
            print(f"[DataCollator] View pair config: {self.view_pairs}")
            print(f"[DataCollator] View field prefix: {self.view_field_prefix}")
            print(f"[DataCollator] Pair sampling strategy: {self.pair_sampling_strategy}")
            print(f"[DataCollator] Single visual teacher: {self.use_single_visual_teacher}")
        print(f"[DataCollator] Privileged visual teacher: {self.use_privileged_visual_teacher}")
        if self.use_privileged_visual_teacher:
            print(f"[DataCollator] Privileged field: {self.privileged_visual_field}")
        print(f"[DataCollator] Image perturbation pairs: {self.use_image_perturbation_pairs}")
        if self.use_image_perturbation_pairs:
            print(f"[DataCollator] Image field: {self.image_field}")
            print(
                "[DataCollator] Perturb params: "
                f"noise_std={self.noise_std}, noise_steps={self.noise_steps}, "
                f"use_tensor_diffusion_noise={self.use_tensor_diffusion_noise}, "
                f"mask_ratio={self.mask_ratio}, "
                f"mask_min_ratio={self.mask_min_ratio if self.mask_min_ratio is not None else self.mask_ratio}, "
                f"mask_max_ratio={self.mask_max_ratio if self.mask_max_ratio is not None else self.mask_ratio}, "
                f"mask_count={self.mask_count}, blur_radius={self.blur_radius}"
            )
            print(
                "[DataCollator] Foreground mask params: "
                f"keep_ratio={self.fg_mask_keep_ratio}, center_bias={self.fg_mask_center_bias}, "
                f"bbox_field={self.object_bbox_field or 'none'}, "
                f"target_field={self.target_object_field or 'none'}, "
                f"bbox_label_field={self.object_bbox_label_field or 'none'}"
            )

    @staticmethod
    def _parse_view_pairs(view_pairs):
        # Normalize user-provided pair specs into a robust internal format.
        # Supported examples: "clean-noise,mask-clean", "clean>noise", "clean:noise",
        # or explicit iterable pairs like [("clean", "noise")].
        pairs = []
        if view_pairs is None:
            return pairs

        if isinstance(view_pairs, str):
            candidates = [item.strip() for item in view_pairs.split(",") if item.strip()]
        else:
            candidates = list(view_pairs)

        for candidate in candidates:
            if isinstance(candidate, (list, tuple)) and len(candidate) == 2:
                teacher_tag, student_tag = str(candidate[0]).strip(), str(candidate[1]).strip()
            else:
                normalized = str(candidate).strip().replace(">", "-").replace(":", "-")
                parts = [part.strip() for part in normalized.split("-") if part.strip()]
                if len(parts) != 2:
                    continue
                teacher_tag, student_tag = parts[0], parts[1]

            if teacher_tag and student_tag:
                pairs.append((teacher_tag, student_tag))

        return pairs

    @staticmethod
    def _normalize_perturbation_pair(pair):
        return normalize_perturbation_pair(pair)

    def _build_view_field_name(self, view_tag):
        return f"{self.view_field_prefix}{view_tag}"

    def _pair_available(self, feature, pair):
        teacher_tag, student_tag = pair
        teacher_field = self._build_view_field_name(teacher_tag)
        student_field = self._build_view_field_name(student_tag)
        return teacher_field in feature and student_field in feature

    def _select_pair(self, feature, example_idx):
        if not self.view_pairs:
            return None

        if self.use_image_perturbation_pairs:
            # In image perturbation mode, pairs are transform tags and do not require
            # text view columns to exist in the dataset.
            if self.pair_sampling_strategy == "first":
                return self.view_pairs[0]
            if self.pair_sampling_strategy == "round_robin":
                return self.view_pairs[example_idx % len(self.view_pairs)]
            return random.choice(self.view_pairs)

        # For dataset-defined textual view columns, only keep pairs for which
        # both sides exist in the current sample to avoid runtime KeyError.
        available_pairs = [pair for pair in self.view_pairs if self._pair_available(feature, pair)]
        if not available_pairs:
            return None

        if self.pair_sampling_strategy == "first":
            return available_pairs[0]
        if self.pair_sampling_strategy == "round_robin":
            return available_pairs[example_idx % len(available_pairs)]
        return random.choice(available_pairs)

    def _to_pil_image(self, image_obj):
        if image_obj is None:
            # Some datasets include samples with missing image content.
            # Use a neutral placeholder so training can continue robustly.
            image = Image.new("RGB", (224, 224), color=(255, 255, 255))
            return image
        if isinstance(image_obj, Image.Image):
            image = image_obj.convert("RGB")
        elif isinstance(image_obj, str):
            source = image_obj.strip()
            try:
                if source.startswith("http://") or source.startswith("https://"):
                    with urlopen(source, timeout=30) as resp:
                        data = resp.read()
                    image = Image.open(BytesIO(data)).convert("RGB")
                else:
                    image = Image.open(source).convert("RGB")
            except Exception:
                # Broken/truncated images should not crash training workers.
                image = Image.new("RGB", (224, 224), color=(255, 255, 255))
        elif isinstance(image_obj, np.ndarray):
            if image_obj.dtype != np.uint8:
                image_obj = np.clip(image_obj, 0, 255).astype(np.uint8)
            image = Image.fromarray(image_obj).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image_obj)}")

        if isinstance(self.max_image_side, int) and self.max_image_side > 0:
            w, h = image.size
            longest = max(w, h)
            if longest > self.max_image_side:
                scale = self.max_image_side / float(longest)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return image

    def _apply_perturbation(self, image_obj, view_tag, feature=None):
        image = self._to_pil_image(image_obj)
        try:
            return apply_image_perturbation(image, view_tag, self.perturbation_cfg, feature=feature or {})
        except ValueError:
            return image

    def _should_use_tensor_diffusion_noise(self, view_tag):
        return self.use_tensor_diffusion_noise and str(view_tag).strip().lower() == "noise"

    def _apply_tensor_diffusion_noise_to_encoded(self, encoded, example_indices):
        if not example_indices or "pixel_values" not in encoded:
            return encoded
        pixel_values = encoded["pixel_values"].clone()
        index_tensor = torch.tensor(example_indices, dtype=torch.long, device=pixel_values.device)
        pixel_values[index_tensor] = add_diffusion_noise_tensor(pixel_values[index_tensor], self.noise_steps)
        encoded["pixel_values"] = pixel_values
        return encoded

    @staticmethod
    def _extract_multimodal_fields(encoded):
        multimodal = {}
        for key in (
            "pixel_values",
            "image_grid_thw",
            "pixel_values_videos",
            "video_grid_thw",
        ):
            if key in encoded:
                multimodal[key] = encoded[key]
        return multimodal

    def _tokenize_with_optional_images(self, prompts, max_prompt_len, images=None):
        # Centralize tokenizer/processor invocation so text-only and multimodal
        # branches share exactly the same padding/truncation contract.
        kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": max_prompt_len,
            "return_tensors": "pt",
        }
        if images is not None:
            encoded = self.tokenizer(text=prompts, images=images, **kwargs)
        else:
            encoded = self._tokenize_text_only(prompts, **kwargs)
        return encoded

    def _tokenize_text_only(self, prompts, **kwargs):
        # For processor-based multimodal tokenizers, pass text as a named
        # argument; positional input may be interpreted as images.
        if hasattr(self.tokenizer, "image_processor") or hasattr(self.tokenizer, "feature_extractor"):
            return self.tokenizer(text=prompts, **kwargs)
        return self.tokenizer(prompts, **kwargs)

    def __call__(self, features):

        batch_size = len(features)

        # Build all prompt variants first, then tokenize in batch. This keeps
        # chat templating consistent with evaluation and avoids per-example
        # tokenizer calls later in the trainer step.
        student_prompts = []
        teacher_prompts = []
        teacher_reasoning_prompts = []  # NEW: for reason_first mode
        teacher_good_prompts = []
        teacher_bad_prompts = []
        student_images = []
        teacher_images = []
        teacher_good_images = []
        teacher_bad_images = []
        student_tensor_noise_indices = []
        teacher_bad_tensor_noise_indices = []
        teacher_good_tensor_noise_indices = []

        for idx, feature in enumerate(features):
            # Extract problem and solution from dataset
            # Handle different possible column names
            if self.problem_field not in feature:
                raise KeyError(f"Missing required field: {self.problem_field}")
            if self.solution_field not in feature:
                raise KeyError(f"Missing required field: {self.solution_field}")

            problem = feature[self.problem_field]
            solution = feature[self.solution_field]
            has_image_input = self.image_field in feature and (
                hasattr(self.tokenizer, "image_processor") or hasattr(self.tokenizer, "feature_extractor")
            )

            privileged_visual = None
            if self.use_privileged_visual_teacher:
                if self.privileged_visual_field not in feature:
                    raise KeyError(f"Missing required privileged field: {self.privileged_visual_field}")
                privileged_visual = feature[self.privileged_visual_field]

            if self.enable_vcd_opsd:
                selected_pair = self._select_pair(feature, idx)
                if selected_pair is not None:
                    teacher_view_tag, student_view_tag = selected_pair
                    if self.use_image_perturbation_pairs:
                        if self.image_field not in feature:
                            raise KeyError(f"Missing required image field: {self.image_field}")

                        raw_image = feature[self.image_field]
                        if self._should_use_tensor_diffusion_noise(teacher_view_tag) or self._should_use_tensor_diffusion_noise(student_view_tag):
                            base_image = self._to_pil_image(raw_image)
                            teacher_img = base_image.copy()
                            student_img = base_image.copy()
                        else:
                            teacher_img = self._apply_perturbation(raw_image, teacher_view_tag, feature=feature)
                            student_img = self._apply_perturbation(raw_image, student_view_tag, feature=feature)

                        # In perturbation mode, we isolate the variable to image quality.
                        # Textual condition remains identical by default.
                        problem_good_view = problem
                        problem_bad_view = problem

                        teacher_good_images.append(teacher_img)
                        if self._should_use_tensor_diffusion_noise(teacher_view_tag):
                            teacher_good_tensor_noise_indices.append(len(teacher_good_images) - 1)
                        if not self.use_single_visual_teacher:
                            teacher_bad_images.append(student_img)
                            if self._should_use_tensor_diffusion_noise(student_view_tag):
                                teacher_bad_tensor_noise_indices.append(len(teacher_bad_images) - 1)
                        student_images.append(student_img)
                        if self._should_use_tensor_diffusion_noise(student_view_tag):
                            student_tensor_noise_indices.append(len(student_images) - 1)
                        teacher_images.append(teacher_img)
                    else:
                        teacher_view_field = self._build_view_field_name(teacher_view_tag)
                        student_view_field = self._build_view_field_name(student_view_tag)
                        problem_good_view = feature.get(teacher_view_field, problem)
                        problem_bad_view = feature.get(student_view_field, problem)
                else:
                    # Backward-compatible fallback for datasets that only provide legacy names.
                    problem_good_view = feature.get(self.good_view_field, problem)
                    problem_bad_view = feature.get(self.bad_view_field, problem)

                # Student trajectory source (OPSD): student-side view from selected teacher-student pair.
                student_problem = problem_bad_view
            else:
                problem_good_view = problem
                problem_bad_view = problem
                student_problem = problem

            if not self.use_image_perturbation_pairs and self.image_field in feature:
                # For multimodal processors without online perturbation, pass the same raw image
                # to student and teacher branches.
                base_image = self._to_pil_image(feature[self.image_field])
                student_images.append(base_image)
                teacher_images.append(base_image)

            if self.use_image_perturbation_pairs or has_image_input:
                student_user_message = (
                    f"{self.image_token}\nProblem: {student_problem}\n\n"
                    "Please reason step by step, and put your final answer within \\boxed{}."
                )
            else:
                student_user_message = f"Problem: {student_problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."

            # Student prompt: just the problem with instruction (matching evaluation format)
            student_messages = [{"role": "user", "content": student_user_message}]

            # Apply chat template for student (matching evaluation)
            student_prompt = self.tokenizer.apply_chat_template(
                student_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            student_prompts.append(student_prompt)

            if self.reason_first:
                # Reasoning prompt: ask teacher to analyze the solution
                reasoning_user_message = (
                    f"Problem: {problem}\n\n"
                    f"Here is a correct reasoning to this problem:"
                    f"=== Reference Reasoning Start ===\n"
                    f"{solution}\n"
                    f"=== Reference Reasoning End ===\n\n"
                    f"{self.reason_first_prompt}"
                )
                reasoning_messages = [{"role": "user", "content": reasoning_user_message}]
                reasoning_prompt = self.tokenizer.apply_chat_template(
                    reasoning_messages, tokenize=False, add_generation_prompt=True
                )
                teacher_reasoning_prompts.append(reasoning_prompt)

                # Teacher prompt will be constructed during training after reasoning
                # For now, create placeholder (will be replaced in training_step)
                teacher_prompts.append("")  # Placeholder
            else:
                if self.enable_vcd_opsd and not self.use_privileged_visual_teacher:
                    if self.use_image_perturbation_pairs:
                        teacher_good_user_message = (
                            f"{self.image_token}\nProblem: {problem_good_view}\n\n"
                            f"Here is a reference solution to this problem:\n"
                            f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                            f"{self.transition_prompt}\n"
                            f"Please reason step by step, and put your final answer within \\boxed{{}}."
                        )
                        if not self.use_single_visual_teacher:
                            teacher_bad_user_message = (
                                f"{self.image_token}\nProblem: {problem_bad_view}\n\n"
                                f"Here is a reference solution to this problem:\n"
                                f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                                f"{self.transition_prompt}\n"
                                f"Please reason step by step, and put your final answer within \\boxed{{}}."
                            )
                    else:
                        if has_image_input:
                            teacher_good_user_message = (
                                f"{self.image_token}\nProblem: {problem_good_view}\n\n"
                                f"Here is a reference solution to this problem:\n"
                                f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                                f"{self.transition_prompt}\n"
                                f"Please reason step by step, and put your final answer within \\boxed{{}}."
                            )
                            if not self.use_single_visual_teacher:
                                teacher_bad_user_message = (
                                    f"{self.image_token}\nProblem: {problem_bad_view}\n\n"
                                    f"Here is a reference solution to this problem:\n"
                                    f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                                    f"{self.transition_prompt}\n"
                                    f"Please reason step by step, and put your final answer within \\boxed{{}}."
                                )
                        else:
                            teacher_good_user_message = (
                                f"Problem: {problem_good_view}\n\n"
                                f"Here is a reference solution to this problem:\n"
                                f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                                f"{self.transition_prompt}\n"
                                f"Please reason step by step, and put your final answer within \\boxed{{}}."
                            )
                            if not self.use_single_visual_teacher:
                                teacher_bad_user_message = (
                                    f"Problem: {problem_bad_view}\n\n"
                                    f"Here is a reference solution to this problem:\n"
                                    f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                                    f"{self.transition_prompt}\n"
                                    f"Please reason step by step, and put your final answer within \\boxed{{}}."
                                )
                    teacher_good_messages = [{"role": "user", "content": teacher_good_user_message}]

                    teacher_good_prompt = self.tokenizer.apply_chat_template(
                        teacher_good_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    )
                    teacher_good_prompts.append(teacher_good_prompt)
                    if self.use_single_visual_teacher:
                        teacher_prompts.append(teacher_good_prompt)
                    else:
                        teacher_bad_messages = [{"role": "user", "content": teacher_bad_user_message}]
                        teacher_bad_prompt = self.tokenizer.apply_chat_template(
                            teacher_bad_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=True,
                        )
                        teacher_bad_prompts.append(teacher_bad_prompt)
                else:
                    if self.use_privileged_visual_teacher:
                        if self.use_image_perturbation_pairs or has_image_input:
                            teacher_user_message = (
                                f"{self.image_token}\nProblem: {problem}\n\n"
                                f"Privileged grounded visual evidence (teacher-only):\n"
                                f"=== Privileged Visual Evidence Begin ===\n{privileged_visual}\n=== Privileged Visual Evidence End ===\n\n"
                                f"Here is a reference solution to this problem:\n"
                                f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                                f"{self.transition_prompt}\n"
                                f"Please reason step by step, and put your final answer within \\boxed{{}}."
                            )
                        else:
                            teacher_user_message = (
                                f"Problem: {problem}\n\n"
                                f"Privileged grounded visual evidence (teacher-only):\n"
                                f"=== Privileged Visual Evidence Begin ===\n{privileged_visual}\n=== Privileged Visual Evidence End ===\n\n"
                                f"Here is a reference solution to this problem:\n"
                                f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                                f"{self.transition_prompt}\n"
                                f"Please reason step by step, and put your final answer within \\boxed{{}}."
                            )
                    else:
                        # Original teacher prompt (unchanged)
                        if has_image_input:
                            teacher_user_message = (
                                f"{self.image_token}\nProblem: {problem}\n\n"
                                f"Here is a reference solution to this problem:\n"
                                f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                                f"{self.transition_prompt}\n"
                                f"Please reason step by step, and put your final answer within \\boxed{{}}."
                            )
                        else:
                            teacher_user_message = (
                                f"Problem: {problem}\n\n"
                                f"Here is a reference solution to this problem:\n"
                                f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                                f"{self.transition_prompt}\n"
                                f"Please reason step by step, and put your final answer within \\boxed{{}}."
                            )
                    teacher_messages = [{"role": "user", "content": teacher_user_message}]

                    # Apply chat template for teacher
                    teacher_prompt = self.tokenizer.apply_chat_template(
                        teacher_messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                    )
                    teacher_prompts.append(teacher_prompt)

        # First pass (no padding): obtain true prompt lengths per sample.
        # We need these lengths for precise label masking later.
        has_student_images = len(student_images) == len(student_prompts) and len(student_images) > 0
        if has_student_images:
            # Measure true prompt lengths first, then pad only to this batch max.
            # This avoids padding every multimodal batch to global `self.max_length`
            # which can unnecessarily blow up activation memory.
            student_encoded_no_pad = self.tokenizer(
                text=student_prompts,
                images=student_images,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]
            max_student_prompt_len = max(student_prompt_lengths)
            student_encoded = self._tokenize_with_optional_images(
                student_prompts,
                max_student_prompt_len,
                images=student_images,
            )
            student_encoded = self._apply_tensor_diffusion_noise_to_encoded(
                student_encoded,
                student_tensor_noise_indices,
            )
        else:
            student_encoded_no_pad = self._tokenize_text_only(
                student_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]

            # Dynamic per-batch max length keeps padding minimal and improves throughput.
            max_student_prompt_len = max(student_prompt_lengths)

            # Second pass (with padding): create fixed-shape tensors for batching.
            student_encoded = self._tokenize_with_optional_images(
                student_prompts,
                max_student_prompt_len,
                images=None,
            )

        result = {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,  # Single value for batch!
            # Keep individual lengths for proper masking
            "student_prompt_lengths_per_example": torch.tensor(student_prompt_lengths),
        }
        for key, value in self._extract_multimodal_fields(student_encoded).items():
            result[f"student_prompt_{key}"] = value

        if self.reason_first:
            # Tokenize reasoning prompts
            reasoning_encoded_no_pad = self._tokenize_text_only(
                teacher_reasoning_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            reasoning_prompt_lengths = [len(ids) for ids in reasoning_encoded_no_pad["input_ids"]]
            max_reasoning_prompt_len = max(reasoning_prompt_lengths)

            reasoning_encoded = self._tokenize_text_only(
                teacher_reasoning_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_reasoning_prompt_len,
                return_tensors="pt",
            )

            # Tokenize transition prompt (this will be appended after reasoning)
            # Don't use chat template here - just the raw text
            transition_text = f"\n{self.transition_prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            transition_encoded = self._tokenize_text_only(
                [transition_text] * batch_size,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_reasoning_prompts": reasoning_encoded["input_ids"],
                    "teacher_reasoning_attention_mask": reasoning_encoded["attention_mask"],
                    "teacher_reasoning_prompt_length": max_reasoning_prompt_len,
                    "teacher_transition_tokens": transition_encoded["input_ids"],
                }
            )
        else:
            if self.enable_vcd_opsd and not self.use_privileged_visual_teacher:
                if self.use_single_visual_teacher:
                    has_teacher_images = len(teacher_images) == len(teacher_prompts) and len(teacher_images) > 0
                    if has_teacher_images:
                        teacher_encoded_no_pad = self.tokenizer(
                            text=teacher_prompts,
                            images=teacher_images,
                            padding=False,
                            truncation=True,
                            max_length=self.max_length,
                        )
                    else:
                        teacher_encoded_no_pad = self._tokenize_text_only(
                            teacher_prompts,
                            padding=False,
                            truncation=True,
                            max_length=self.max_length,
                        )

                    teacher_prompt_lengths = [len(ids) for ids in teacher_encoded_no_pad["input_ids"]]
                    max_teacher_prompt_len = max(teacher_prompt_lengths)

                    teacher_encoded = self._tokenize_with_optional_images(
                        teacher_prompts,
                        max_teacher_prompt_len,
                        images=teacher_images if has_teacher_images else None,
                    )

                    result.update(
                        {
                            "teacher_prompts": teacher_encoded["input_ids"],
                            "teacher_prompt_attention_mask": teacher_encoded["attention_mask"],
                            "teacher_prompt_length": max_teacher_prompt_len,
                            "teacher_prompt_lengths_per_example": torch.tensor(teacher_prompt_lengths),
                        }
                    )
                    for key, value in self._extract_multimodal_fields(teacher_encoded).items():
                        result[f"teacher_prompt_{key}"] = value
                else:
                    if self.use_image_perturbation_pairs:
                        teacher_good_encoded_no_pad = self.tokenizer(
                            text=teacher_good_prompts,
                            images=teacher_good_images,
                            padding=False,
                            truncation=True,
                            max_length=self.max_length,
                        )
                    else:
                        teacher_good_encoded_no_pad = self._tokenize_text_only(
                            teacher_good_prompts,
                            padding=False,
                            truncation=True,
                            max_length=self.max_length,
                        )
                    teacher_good_prompt_lengths = [
                        len(ids) for ids in teacher_good_encoded_no_pad["input_ids"]
                    ]
                    max_teacher_good_prompt_len = max(teacher_good_prompt_lengths)

                    teacher_good_encoded = self._tokenize_with_optional_images(
                        teacher_good_prompts,
                        max_teacher_good_prompt_len,
                        images=teacher_good_images if self.use_image_perturbation_pairs else None,
                    )
                    teacher_good_encoded = self._apply_tensor_diffusion_noise_to_encoded(
                        teacher_good_encoded,
                        teacher_good_tensor_noise_indices,
                    )

                    if self.use_image_perturbation_pairs:
                        teacher_bad_encoded_no_pad = self.tokenizer(
                            text=teacher_bad_prompts,
                            images=teacher_bad_images,
                            padding=False,
                            truncation=True,
                            max_length=self.max_length,
                        )
                    else:
                        teacher_bad_encoded_no_pad = self._tokenize_text_only(
                            teacher_bad_prompts,
                            padding=False,
                            truncation=True,
                            max_length=self.max_length,
                        )
                    teacher_bad_prompt_lengths = [
                        len(ids) for ids in teacher_bad_encoded_no_pad["input_ids"]
                    ]
                    max_teacher_bad_prompt_len = max(teacher_bad_prompt_lengths)

                    teacher_bad_encoded = self._tokenize_with_optional_images(
                        teacher_bad_prompts,
                        max_teacher_bad_prompt_len,
                        images=teacher_bad_images if self.use_image_perturbation_pairs else None,
                    )
                    teacher_bad_encoded = self._apply_tensor_diffusion_noise_to_encoded(
                        teacher_bad_encoded,
                        teacher_bad_tensor_noise_indices,
                    )

                    # Preserve baseline key names (`teacher_prompts`, `teacher_prompt_*`) so
                    # existing logging/debug code keeps working without refactor.
                    result.update(
                        {
                            "teacher_prompts": teacher_good_encoded["input_ids"],
                            "teacher_prompt_attention_mask": teacher_good_encoded["attention_mask"],
                            "teacher_prompt_length": max_teacher_good_prompt_len,
                            "teacher_prompt_lengths_per_example": torch.tensor(
                                teacher_good_prompt_lengths
                            ),
                            "teacher_good_prompts": teacher_good_encoded["input_ids"],
                            "teacher_good_prompt_attention_mask": teacher_good_encoded["attention_mask"],
                            "teacher_good_prompt_length": max_teacher_good_prompt_len,
                            "teacher_good_prompt_lengths_per_example": torch.tensor(
                                teacher_good_prompt_lengths
                            ),
                            "teacher_bad_prompts": teacher_bad_encoded["input_ids"],
                            "teacher_bad_prompt_attention_mask": teacher_bad_encoded["attention_mask"],
                            "teacher_bad_prompt_length": max_teacher_bad_prompt_len,
                            "teacher_bad_prompt_lengths_per_example": torch.tensor(
                                teacher_bad_prompt_lengths
                            ),
                        }
                    )
                    for key, value in self._extract_multimodal_fields(teacher_good_encoded).items():
                        result[f"teacher_prompt_{key}"] = value
                        result[f"teacher_good_prompt_{key}"] = value
                    for key, value in self._extract_multimodal_fields(teacher_bad_encoded).items():
                        result[f"teacher_bad_prompt_{key}"] = value
            else:
                # Normal mode: tokenize teacher prompts
                has_teacher_images = len(teacher_images) == len(teacher_prompts) and len(teacher_images) > 0
                if has_teacher_images:
                    teacher_encoded_no_pad = self.tokenizer(
                        text=teacher_prompts,
                        images=teacher_images,
                        padding=False,
                        truncation=True,
                        max_length=self.max_length,
                    )
                else:
                    teacher_encoded_no_pad = self._tokenize_text_only(
                        teacher_prompts,
                        padding=False,
                        truncation=True,
                        max_length=self.max_length,
                    )
                teacher_prompt_lengths = [len(ids) for ids in teacher_encoded_no_pad["input_ids"]]
                max_teacher_prompt_len = max(teacher_prompt_lengths)

                teacher_encoded = self._tokenize_with_optional_images(
                    teacher_prompts,
                    max_teacher_prompt_len,
                    images=teacher_images if has_teacher_images else None,
                )

                result.update(
                    {
                        "teacher_prompts": teacher_encoded["input_ids"],
                        "teacher_prompt_attention_mask": teacher_encoded["attention_mask"],
                        "teacher_prompt_length": max_teacher_prompt_len,
                        "teacher_prompt_lengths_per_example": torch.tensor(teacher_prompt_lengths),
                    }
                )
                for key, value in self._extract_multimodal_fields(teacher_encoded).items():
                    result[f"teacher_prompt_{key}"] = value

        return result
