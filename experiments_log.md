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

## 2026-06-22 - h200 Handoff and 8-GPU Original Synthetic Eval

- Goal: Move the modal_aphasia Emu3.5 evaluation from pc-super to h200 and run the original synthetic image-generation benchmark faster.
- Change: Handoff source to h200 under `/home/chen/workplace/umm/modal-aphasia`; use conda env `emu35-modal`; download `BAAI/Emu3.5` and `BAAI/Emu3.5-VisionTokenizer` directly on h200; copy only the two LoRA adapter payloads from pc-super.
- Runtime setup: Added h200-local image sharding arguments to `scripts/eval_emu35_original_synthetic.py` and a merge helper `scripts/merge_emu35_original_synthetic_shards.py`. The benchmark builder/grader remains modal_aphasia's original `InferenceImageOutputBuilder.build_synthetic_concepts` plus classifier grading.
- Validation: Dry-run confirmed 44 text MC rows and 840 image rows. Sharding dry-run confirmed 8 shards x 105 image rows. Smoke eval completed for both checkpoints and wrote generated-image/classifier summaries.
- Running: h200 supervisor `runs/run_h200_8gpu_eval_20260622_1218.sh` first runs image-adv-original across GPUs 0-7, then text-adv across GPUs 0-7. Logs are under `runs/h200_original_synthetic_*_8gpu_shards_20260622_1218_shard_*.log`.
- Output paths: image-adv shards `outputs/eval/h200_original_synthetic_image_adv_8gpu_shards_20260622_1218`; merged image-adv output `outputs/eval/h200_original_synthetic_image_adv_8gpu_merged_20260622_1218`; text-adv shards `outputs/eval/h200_original_synthetic_text_adv_8gpu_shards_20260622_1218`; merged text-adv output `outputs/eval/h200_original_synthetic_text_adv_8gpu_merged_20260622_1218`.
- Current status: image-adv-original 8 shards are running on h200 GPUs 0-7, each using about 26GB GPU memory. Text-adv will start automatically after image-adv merges.

## 2026-06-22 - h200 Two-Tasks-Per-GPU Eval Update

- Goal: Use the remaining h200 GPU memory to run both checkpoint conditions concurrently.
- Change: Stopped the sequential 8-GPU supervisor shell while leaving the already-running image-adv shard processes alive. Launched text-adv shards on the same GPUs 0-7 and started `runs/run_h200_2pergpu_eval_20260622_1228.sh` to watch both sets and merge both outputs.
- Runtime setup: Each GPU now runs one `image_adv_original` shard and one `text_adv` shard. `TORCHINDUCTOR_COMPILE_THREADS=4` is set for the newly launched text-adv shard supervisor to reduce compile-worker pressure.
- Output paths: image-adv merged output remains `outputs/eval/h200_original_synthetic_image_adv_8gpu_merged_20260622_1218`; text-adv merged output is `outputs/eval/h200_original_synthetic_text_adv_8gpu_merged_20260622_1228`.
- Current status: 16 eval processes are running. GPUs 0-7 each use about 53.5GB memory with high utilization. Image-adv shards have started writing generated images; text-adv shards have loaded the model and are entering generation.

## 2026-06-22 - Dynamic Image Resplit After Text Completion

- Goal: Once text-adv finishes, keep h200 GPUs saturated by redistributing unfinished image-adv rows so each GPU again runs two image workers.
- Change: Added `--image-rows-file` support to `scripts/eval_emu35_original_synthetic.py`, added `scripts/recover_and_split_image_rows.py`, stopped the previous two-task watcher, and launched `runs/run_h200_dynamic_image_resplit_20260622_2055.sh`.
- Runtime plan: The dynamic watcher waits for all text-adv shard processes to finish, merges text-adv, stops the old image-adv shard processes, recovers already generated PNGs into a graded recovered shard, splits remaining image sample IDs into 16 residual row-list files, and launches two residual image workers per GPU.
- Output paths: recovered/residual image shards `outputs/eval/h200_original_synthetic_image_adv_dynamic_shards_20260622_2055`; final dynamic image-adv merge `outputs/eval/h200_original_synthetic_image_adv_dynamic_merged_20260622_2055`; text merge remains `outputs/eval/h200_original_synthetic_text_adv_8gpu_merged_20260622_1228`.
- Current status: dynamic watcher `runs/h200_dynamic_image_resplit_supervisor_20260622_2055.log` is waiting for the final text-adv shard to finish. Existing image-adv workers continue running until that trigger fires.
