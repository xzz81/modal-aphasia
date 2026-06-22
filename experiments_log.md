## 2026-06-22 - Original Synthetic Evaluation Boundary Correction

- Goal: Evaluate the text-adv Emu3.5 checkpoint and the original modal_aphasia image-adv checkpoint with modal_aphasia's original synthetic benchmark code.
- Change: Added `scripts/eval_emu35_original_synthetic.py`, which uses `InferenceTextOutputBuilder.build_concepts_description_mc` for text MC and `InferenceImageOutputBuilder.build_synthetic_concepts` plus `grade_synthetic_images.grade_result` for image grading. This replaces the earlier custom symmetric evaluation for this comparison.
- Data/model paths: text-adv checkpoint `model/finetuned/modal_aphasia_symmetric/emu35_lora_core_c/checkpoint-200`; image-adv checkpoint `model/finetuned/modal_aphasia_original/synth_concepts_lora_a6000`; original synthetic dataset exposed as `data/synthetic_images -> data/synthetic_images_384`.
- Validation: Dry-run produced 44 text MC rows and 840 image rows from the original builders. Smoke completed for text-adv and image-adv. Full text MC completed for both checkpoints.
- Result: Full original text MC: text-adv 36/44 overall, 20/22 synthetic-query; image-adv-original 8/44 overall, 2/22 synthetic-query. The copied a6000 full image benchmark reference is under `outputs/eval/a6000_modal_memory_lora_full_reference`.
- Running: Full original synthetic image evaluation for text-adv is running in tmux session `codex-modal-original-image-eval`; output `outputs/eval/emu35_original_synthetic_text_adv_image_full_20260622_170000`; log `runs/emu35_original_synthetic_text_adv_image_full_20260622_170000.log`.
- Interpretation: The text-adv condition now has a confirmed text-side advantage under the original modal_aphasia MC metric. The original a6000 checkpoint remains the image-adv condition; it should be compared using its original full synthetic image benchmark, not the earlier anchor-only symmetric image-adv run.
- Next step: Monitor the text-adv full image run, then merge text-adv image metrics with the a6000 image-adv reference for the final modal-gap case table/report.

## 2026-06-22 - Original Synthetic Image Generation Evaluation for Both Checkpoints

- Goal: Run the actual modal_aphasia image generation benchmark for both checkpoints, using `InferenceImageOutputBuilder.build_synthetic_concepts` and `grade_synthetic_images`.
- Protocol: 840 fake-word prompts from the original synthetic train/test splits. Each prompt asks the model to generate an image; generated images are classified for `color`, `pattern`, `position`, and `shape`.
- Checkpoints: text-adv `model/finetuned/modal_aphasia_symmetric/emu35_lora_core_c/checkpoint-200`; image-adv-original `model/finetuned/modal_aphasia_original/synth_concepts_lora_a6000`.
- Running: text-adv generation eval is running in tmux `codex-modal-original-image-eval` on pc-super GPU1, output `outputs/eval/emu35_original_synthetic_text_adv_image_full_20260622_170000`, log `runs/emu35_original_synthetic_text_adv_image_full_20260622_170000.log`.
- Running: image-adv-original generation eval is running in tmux `codex-modal-original-image-adv-eval` on pc-super GPU0, output `outputs/eval/emu35_original_synthetic_image_adv_full_20260622_173500`, log `runs/emu35_original_synthetic_image_adv_full_20260622_173500.log`.
- Next step: Monitor both tmux jobs until `summary.json` is written, then compare all-attribute and per-attribute image generation accuracy for the two checkpoints.

## 2026-06-22 - Evaluation Code Cleanup

- Goal: Remove non-original or misleading evaluation entry points before committing.
- Change: Removed non-original diagnostic evaluation code and the earlier custom symmetric evaluation launcher/code. Kept only the text-adv dataset/training utilities and the original modal_aphasia synthetic generation/text-MC evaluation script.
- Reason: The official comparison should use modal_aphasia's original `concepts_description_mc` text MC and `synthetic_concepts` image generation benchmarks. Non-original diagnostics should not be present in the committed code because they can be mistaken for the main benchmark.
