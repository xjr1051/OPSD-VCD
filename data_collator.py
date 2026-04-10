import torch
import random
import numpy as np
from PIL import Image, ImageFilter


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
        mask_ratio=0.25,
        blur_radius=2.0,
        use_privileged_visual_teacher=False,
        privileged_visual_field="privileged_visual_evidence",
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
        self.mask_ratio = mask_ratio
        self.blur_radius = blur_radius
        self.use_privileged_visual_teacher = use_privileged_visual_teacher
        self.privileged_visual_field = privileged_visual_field

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

        if self.reason_first and self.enable_vcd_opsd:
            raise ValueError(
                "reason_first and enable_vcd_opsd cannot be enabled together in this baseline skeleton."
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

        # Set padding side explicitly for consistency
        print(f"[DataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[DataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[DataCollator] Reason first mode: {self.reason_first}")
        print(f"[DataCollator] VCD-OPSD mode: {self.enable_vcd_opsd}")
        if self.enable_vcd_opsd:
            print(
                f"[DataCollator] VCD view fields: good={self.good_view_field}, bad={self.bad_view_field}"
            )
            print(f"[DataCollator] View pair config: {self.view_pairs}")
            print(f"[DataCollator] View field prefix: {self.view_field_prefix}")
            print(f"[DataCollator] Pair sampling strategy: {self.pair_sampling_strategy}")
        print(f"[DataCollator] Privileged visual teacher: {self.use_privileged_visual_teacher}")
        if self.use_privileged_visual_teacher:
            print(f"[DataCollator] Privileged field: {self.privileged_visual_field}")
        print(f"[DataCollator] Image perturbation pairs: {self.use_image_perturbation_pairs}")
        if self.use_image_perturbation_pairs:
            print(f"[DataCollator] Image field: {self.image_field}")
            print(
                f"[DataCollator] Perturb params: noise_std={self.noise_std}, mask_ratio={self.mask_ratio}, blur_radius={self.blur_radius}"
            )

    @staticmethod
    def _parse_view_pairs(view_pairs):
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

        available_pairs = [pair for pair in self.view_pairs if self._pair_available(feature, pair)]
        if not available_pairs:
            return None

        if self.pair_sampling_strategy == "first":
            return available_pairs[0]
        if self.pair_sampling_strategy == "round_robin":
            return available_pairs[example_idx % len(available_pairs)]
        return random.choice(available_pairs)

    @staticmethod
    def _to_pil_image(image_obj):
        if isinstance(image_obj, Image.Image):
            return image_obj.convert("RGB")
        if isinstance(image_obj, np.ndarray):
            if image_obj.dtype != np.uint8:
                image_obj = np.clip(image_obj, 0, 255).astype(np.uint8)
            return Image.fromarray(image_obj).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image_obj)}")

    def _apply_perturbation(self, image_obj, view_tag):
        image = self._to_pil_image(image_obj)
        tag = str(view_tag).strip().lower()

        if tag == "clean":
            return image
        if tag == "noise":
            arr = np.asarray(image).astype(np.float32)
            noise = np.random.normal(0.0, self.noise_std, size=arr.shape)
            arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
            return Image.fromarray(arr)
        if tag == "mask":
            arr = np.asarray(image).copy()
            h, w = arr.shape[:2]
            mask_h = max(1, int(h * self.mask_ratio))
            mask_w = max(1, int(w * self.mask_ratio))
            top = random.randint(0, max(0, h - mask_h))
            left = random.randint(0, max(0, w - mask_w))
            arr[top : top + mask_h, left : left + mask_w] = 0
            return Image.fromarray(arr)
        if tag == "blur":
            return image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        # Unknown tags default to clean so the training loop remains robust.
        return image

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
        kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": max_prompt_len,
            "return_tensors": "pt",
        }
        if images is not None:
            encoded = self.tokenizer(text=prompts, images=images, **kwargs)
        else:
            encoded = self.tokenizer(prompts, **kwargs)
        return encoded

    def __call__(self, features):

        batch_size = len(features)

        # Prepare student and teacher prompts using chat template (matching evaluation)
        student_prompts = []
        teacher_prompts = []
        teacher_reasoning_prompts = []  # NEW: for reason_first mode
        teacher_good_prompts = []
        teacher_bad_prompts = []
        student_images = []
        teacher_images = []
        teacher_good_images = []
        teacher_bad_images = []

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
                        teacher_img = self._apply_perturbation(raw_image, teacher_view_tag)
                        student_img = self._apply_perturbation(raw_image, student_view_tag)

                        # Keep text condition shared by default in image-perturbation mode.
                        problem_good_view = problem
                        problem_bad_view = problem

                        teacher_good_images.append(teacher_img)
                        teacher_bad_images.append(student_img)
                        student_images.append(student_img)
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
                            teacher_bad_user_message = (
                                f"Problem: {problem_bad_view}\n\n"
                                f"Here is a reference solution to this problem:\n"
                                f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                                f"{self.transition_prompt}\n"
                                f"Please reason step by step, and put your final answer within \\boxed{{}}."
                            )
                    teacher_good_messages = [{"role": "user", "content": teacher_good_user_message}]
                    teacher_bad_messages = [{"role": "user", "content": teacher_bad_user_message}]

                    teacher_good_prompt = self.tokenizer.apply_chat_template(
                        teacher_good_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    )
                    teacher_bad_prompt = self.tokenizer.apply_chat_template(
                        teacher_bad_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    )
                    teacher_good_prompts.append(teacher_good_prompt)
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

        # Tokenize WITHOUT padding first to get true lengths
        has_student_images = len(student_images) == len(student_prompts) and len(student_images) > 0
        if has_student_images:
            student_encoded_no_pad = self.tokenizer(
                text=student_prompts,
                images=student_images,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
        else:
            student_encoded_no_pad = self.tokenizer(
                student_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]

        # Find max lengths in this batch
        max_student_prompt_len = max(student_prompt_lengths)

        # Tokenize WITH padding to max length in batch
        student_encoded = self._tokenize_with_optional_images(
            student_prompts,
            max_student_prompt_len,
            images=student_images if has_student_images else None,
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
            reasoning_encoded_no_pad = self.tokenizer(
                teacher_reasoning_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            reasoning_prompt_lengths = [len(ids) for ids in reasoning_encoded_no_pad["input_ids"]]
            max_reasoning_prompt_len = max(reasoning_prompt_lengths)

            reasoning_encoded = self.tokenizer(
                teacher_reasoning_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_reasoning_prompt_len,
                return_tensors="pt",
            )

            # Tokenize transition prompt (this will be appended after reasoning)
            # Don't use chat template here - just the raw text
            transition_text = f"\n{self.transition_prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            transition_encoded = self.tokenizer(
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
                if self.use_image_perturbation_pairs:
                    teacher_good_encoded_no_pad = self.tokenizer(
                        text=teacher_good_prompts,
                        images=teacher_good_images,
                        padding=False,
                        truncation=True,
                        max_length=self.max_length,
                    )
                else:
                    teacher_good_encoded_no_pad = self.tokenizer(
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

                if self.use_image_perturbation_pairs:
                    teacher_bad_encoded_no_pad = self.tokenizer(
                        text=teacher_bad_prompts,
                        images=teacher_bad_images,
                        padding=False,
                        truncation=True,
                        max_length=self.max_length,
                    )
                else:
                    teacher_bad_encoded_no_pad = self.tokenizer(
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

                # Keep teacher_prompts aliases for backward compatibility in downstream logging.
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
                    teacher_encoded_no_pad = self.tokenizer(
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
