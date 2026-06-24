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

## 2026-06-22 - h200 Original Synthetic Eval Final Results

- Goal: Preserve the completed h200 original synthetic benchmark comparison for text-adv vs image-adv-original.
- Command/config: Original modal_aphasia builders `concepts_description_mc` and `synthetic_concepts`; dynamic image residual phase used `--image-rows-file` with 16 residual shards after recovering 248 completed image rows.
- Output paths: text-adv final output `outputs/eval/h200_original_synthetic_text_adv_8gpu_merged_20260622_1228`; image-adv final output `outputs/eval/h200_original_synthetic_image_adv_dynamic_merged_20260622_2055`; saved bundle `outputs/eval/final_bundles/h200_original_synthetic_text_image_adv_20260622.tar.gz`.
- Result: text-adv text MC 35/44 = 79.5%, image all-attribute 0/840 = 0.0%. image-adv-original text MC 7/44 = 15.9%, image all-attribute 12/840 = 1.43%; per-attribute image accuracy color 34.0%, pattern 51.2%, position 34.2%, shape 15.1%.
- Interpretation: text-adv preserves text retrieval and loses image generation; image-adv-original preserves more image attributes but has weak exact all-attribute generation and poor text MC. The strict `image_all_correct && text_wrong` case count is 12/840 for image-adv-original and 0/840 for text-adv.
- Preservation: Bundle SHA256 `65e90b047ada25034d30942d1d66eacc9813e130bff52e59dff16e2a89b91b71`; copied to h200, pc-super, and local Codex workspace.

## 2026-06-24 - Tianyang Janus-Pro-7B Local Asset Setup

- goal: make Janus-Pro-7B available inside the tianyang modal-aphasia handoff for original Janus workflows.
- hypothesis or change: reuse existing `/cache/ummu/model/Janus-Pro-7B` and expose it as repo-relative `model/Janus-Pro-7B`; update Janus model loading to prefer the local repo asset when callers request `deepseek-ai/Janus-Pro-7B`.
- command/config: linked `model/Janus-Pro-7B -> /cache/ummu/model/Janus-Pro-7B`; validated `VLChatProcessor.from_pretrained(model/Janus-Pro-7B)` and `python3 -m py_compile modal_aphasia/janus/modeling_vlm.py modal_aphasia/janus/inference.py modal_aphasia/janus/train.py`.
- data/model paths: `model/Janus-Pro-7B`; physical asset on tianyang at `/cache/ummu/model/Janus-Pro-7B`.
- output paths: no runtime outputs; git-tracked code change in `modal_aphasia/janus/modeling_vlm.py`.
- result: local processor resolution works for `deepseek-ai/Janus-Pro-7B`, avoiding remote Hugging Face access for the processor/model path when the local asset exists.
- interpretation: Janus-Pro workflows in this project can now use the local tianyang asset consistently with repo-relative paths.
- next step: run a Janus smoke inference/training check if Janus-specific evaluation is needed.


## 2026-06-24 - Tianyang Janus-Pro Concepts 4GPU Smoke

- goal: validate Janus-Pro fine-tuning on modal_aphasia synthetic concepts benchmark data before launching longer runs.
- hypothesis or change: use tianyang 4xA100 with DeepSpeed ZeRO-2 instead of single-GPU training; disable SwanLab auto-integration with SWANLAB_MODE=disabled.
- command/config: /opt/venv/bin/python -m accelerate.commands.launch --config-file configs/concepts_janus/accelerate_config.yaml --num_processes 4 -m modal_aphasia.janus.train --setting synthetic_concepts_extended --aux-fraction 0.0 --num-prompt-permutations 1 --language-model-only --per-device-train-batch-size 1 --gradient-accumulation-steps 1.
- data/model paths: data/synthetic_images; model/Janus-Pro-7B; output model/finetuned_janus/concepts_smoke/janus-concepts-smoke-4gpu-20260624.
- output paths: runs/janus_concepts_smoke_4gpu_20260624.log; model/finetuned_janus/concepts_smoke/janus-concepts-smoke-4gpu-20260624.
- result: completed 168/168 steps, train_loss=2.86806549344744, eval_val_loss improved to 2.1915745735168457; final checkpoint saved.
- interpretation: Janus-Pro local asset, dataset builder, VQ image-token collator, Trainer, and 4GPU DeepSpeed launch are working on tianyang.
- next step: launch longer synthetic_concepts_extended run with 24 prompt permutations and prepare faces dataset from the real a6000 faces_raw.zip.

## 2026-06-24 - Tianyang Janus-Pro Concepts Full No-Aux Launch and Faces Data Prep

- goal: launch Janus-Pro fine-tuning on a full modal_aphasia synthetic concepts setting and prepare a second benchmark dataset family, faces.
- hypothesis or change: run synthetic_concepts_extended with the original 24 prompt permutations on tianyang 4xA100; keep aux_fraction=0.0 because BLIP3o auxiliary download from hf-mirror timed out repeatedly. Transfer real faces_raw.zip from a6000 and generate data/faces with the project script.
- command/config: concepts launched with /opt/venv/bin/python -m accelerate.commands.launch --config-file configs/concepts_janus/accelerate_config.yaml --num_processes 4 -m modal_aphasia.janus.train --setting synthetic_concepts_extended --num-prompt-permutations 24 --aux-fraction 0.0 --per-device-train-batch-size 1 --gradient-accumulation-steps 1 --language-model-only --save-final-model. Faces generated by modal_aphasia.data.generate_faces_dataset from data/faces_cache/full_resolution.
- data/model paths: data/synthetic_images; data/faces; data/faces_raw.zip copied from a6000:/workspace/home/AAAI 2027/modal-aphasia/misc/faces_raw.zip; model/Janus-Pro-7B.
- output paths: runs/janus_concepts_full_noaux_24perm_20260624.log; model/finetuned_janus/concepts/janus-concepts-24perm-noaux-seed178430-20260624; runs/generate_faces_dataset_20260624.log; data/faces.
- result: concepts training entered the training loop with 4014 steps and reached at least step 31 with loss decreasing 5.2823 -> 3.7412. Faces dataset generation succeeded with 600 records, output size about 226MB.
- interpretation: two benchmark data families are now usable on tianyang; concepts is actively training, faces is ready for Janus-Pro fine-tuning after GPU release.
- next step: monitor concepts to completion; queue or launch faces training on the same 4GPU DeepSpeed setup.

## 2026-06-24 - Tianyang Janus-Pro Faces Queue

- goal: schedule a second modal_aphasia Janus-Pro benchmark fine-tuning family after concepts releases the GPUs.
- hypothesis or change: use original faces setting and 100 epochs, with tianyang-safe 4GPU DeepSpeed ZeRO-2 and per_device_train_batch_size=1 instead of the original 8GPU batch size 4.
- command/config: queued tmux session janus-faces-queued waits for janus-concepts-24perm-noaux-seed178430-20260624 to exit, then launches /opt/venv/bin/python -m accelerate.commands.launch --config-file configs/faces_janus/accelerate_config.yaml --num_processes 4 -m modal_aphasia.janus.train --setting faces --num-epochs 100 --aux-fraction 0.0 --language-model-only --per-device-train-batch-size 1 --gradient-accumulation-steps 1 --save-final-model.
- data/model paths: data/faces; model/Janus-Pro-7B.
- output paths: runs/janus_faces_queued_20260624.log; model/finetuned_janus/faces/janus-faces-seed178430-20260624.
- result: queued, pending completion of the active concepts training job.
- interpretation: concepts and faces are now both staged under native Janus training settings, with tianyang memory adjustments recorded.
- next step: monitor concepts completion, then verify faces starts and reaches first loss logs.

## 2026-06-24 - Tianyang Janus-Pro Safety Data Prep and Queue

- goal: add the modal_aphasia safety image benchmark family to the tianyang Janus-Pro fine-tuning queue.
- hypothesis or change: materialize the original 50-image safety cache from misc/safety_images_meta.jsonl, preserving sha256 hashes; then queue safety_unsafe training after concepts and faces to avoid GPU contention.
- command/config: scripts/download_safety_images.py --cache-dir data/safety_images_cache; DATA_ROOT=data python -m modal_aphasia.data.generate_safety_dataset --cache-dir data/safety_images_cache --output-dir data/safety_images; queued /opt/venv/bin/python -m accelerate.commands.launch --config-file configs/safety/accelerate_config.yaml --num_processes 4 -m modal_aphasia.janus.train --setting safety_unsafe --num-epochs 6 --aux-fraction 0.0 --per-device-train-batch-size 1 --gradient-accumulation-steps 1 --language-model-only --save-final-model.
- data/model paths: data/safety_images_cache has 50 raw Unsplash files matching misc/safety_images_meta.jsonl sha256; data/safety_images has 50 processed examples; model/Janus-Pro-7B.
- output paths: runs/generate_safety_dataset_20260624.log; runs/janus_safety_queued_20260624.log; model/finetuned_janus/safety/janus-safety-unsafe-seed178430-20260624.
- result: safety cache verified 50/50, generated dataset saved with 50 prompt/image records, and tmux session janus-safety-queued is waiting for janus-faces-queued to finish before launching.
- interpretation: three Janus-Pro benchmark data families are now staged on tianyang: synthetic concepts, faces, and safety_unsafe. Safety uses aux_fraction=0.0 because BLIP aux remains unavailable on this host.
- limitation: current Janus SettingType implements safety_unsafe only; the safety_refusal aligned phase referenced by configs/safety/train.sh is not implemented in modal_aphasia/janus/train.py and was not queued.
- next step: monitor concepts completion, verify faces starts and finishes, then verify safety_unsafe starts and reaches first eval/loss logs.


## 2026-06-24 - Tianyang Janus-Pro Guarded Queue Repair

- goal: ensure the queued faces and safety_unsafe fine-tuning stages only start after the previous Janus-Pro stage truly succeeds.
- hypothesis or change: replace the original process/session-only waits with guarded tmux queues that also require `Finished training` in the previous log and final checkpoint files `config.json` plus `model.safetensors.index.json`.
- command/config: killed only `janus-faces-queued` and `janus-safety-queued`; relaunched `/tmp/janus_faces_guarded.sh` and `/tmp/janus_safety_guarded.sh`. Concepts training in `janus-concepts-full` was left untouched.
- data/model paths: concepts output `model/finetuned_janus/concepts/janus-concepts-24perm-noaux-seed178430-20260624`; faces output `model/finetuned_janus/faces/janus-faces-seed178430-20260624`; safety output `model/finetuned_janus/safety/janus-safety-unsafe-seed178430-20260624`.
- output paths: `runs/janus_faces_queued_20260624.log`; `runs/janus_safety_queued_20260624.log`.
- result: guarded queues are active and waiting; current concepts training was still running at about step 955/4014 when verified.
- interpretation: this prevents a failed concepts or faces run from silently triggering the next benchmark family and hiding the failure.
- next step: monitor concepts completion, then verify the guarded faces queue starts and reaches first loss/eval logs.

## 2026-06-24 - Tianyang Janus-Pro Concepts Full Completion and Faces Start

- goal: verify completion of the full synthetic concepts Janus-Pro benchmark fine-tuning stage and confirm the next benchmark family starts through the guarded queue.
- hypothesis or change: no setting change; continue the previously launched `synthetic_concepts_extended` 24-prompt-permutation no-aux run until final checkpoint save, then let the guarded faces queue start automatically.
- command/config: monitored `runs/janus_concepts_full_noaux_24perm_20260624.log`; training command remained `/opt/venv/bin/python -m accelerate.commands.launch --config-file configs/concepts_janus/accelerate_config.yaml --num_processes 4 -m modal_aphasia.janus.train --setting synthetic_concepts_extended --num-prompt-permutations 24 --aux-fraction 0.0 --language-model-only --per-device-train-batch-size 1 --gradient-accumulation-steps 1 --save-final-model`.
- data/model paths: `data/synthetic_images`; `model/Janus-Pro-7B`; concepts checkpoint `model/finetuned_janus/concepts/janus-concepts-24perm-noaux-seed178430-20260624`; faces dataset `data/faces`.
- output paths: concepts log `runs/janus_concepts_full_noaux_24perm_20260624.log`; faces queue log `runs/janus_faces_queued_20260624.log`.
- result: concepts completed 4014/4014 steps with `train_runtime=9656.9447`, `train_loss=0.27583238417156974`, and `Finished training`. Final checkpoint contains `config.json`, `model.safetensors.index.json`, seven safetensors shards, tokenizer files, and processor files. The guarded faces queue detected the successful checkpoint and logged `starting_faces_2026-06-24T14:47:58+08:00`.
- interpretation: the first Janus-Pro benchmark family finished successfully on tianyang; guard conditions worked as intended and released the next benchmark family only after the final concepts checkpoint was complete.
- next step: monitor faces until it reaches stable loss/eval logs and final checkpoint, then verify the safety_unsafe guarded queue starts.

## 2026-06-24 - Tianyang Janus-Pro Faces Standard Global-Batch Restart

- goal: restart faces Janus-Pro fine-tuning with optimizer-step semantics matching the original faces benchmark config.
- hypothesis or change: keep `num_epochs=100` but compensate for tianyang 4GPU memory-safe microbatching by using `gradient_accumulation_steps=8`, so effective global batch is `4 GPUs * 1 per-device * 8 accumulation = 32`, matching the original `8 GPUs * 4 per-device`.
- command/config: stopped the previous faces run that used `--num_processes 4 --per-device-train-batch-size 1 --gradient-accumulation-steps 1`; relaunched guarded faces with `/opt/venv/bin/python -m accelerate.commands.launch --config-file configs/faces_janus/accelerate_config.yaml --num_processes 4 -m modal_aphasia.janus.train --output-model-id janus-faces-seed178430-20260624 --seed 178430 --num-epochs 100 --aux-fraction 0.0 --learning-rate 1e-5 --learning-rate-scheduler linear --warmup-steps 25 --language-model-only --per-device-train-batch-size 1 --gradient-accumulation-steps 8 --save-strategy no --eval-steps 20 --setting faces --output-root model/finetuned_janus/faces --save-final-model`.
- data/model paths: `data/faces`; `model/Janus-Pro-7B`; intended faces checkpoint `model/finetuned_janus/faces/janus-faces-seed178430-20260624`.
- output paths: current corrected log `runs/janus_faces_queued_20260624.log`; archived wrong-effective-batch log `runs/janus_faces_queued_20260624.aborted_effective_batch4_20260624_153001.log`; archived partial output `model/finetuned_janus/faces/janus-faces-seed178430-20260624.aborted_effective_batch4_20260624_153001`.
- result: corrected guarded faces run started at `2026-06-24T15:31:04+08:00`. Verification showed command-line `--gradient-accumulation-steps 8`, Trainer total `0/1900`, and all 4 A100 GPUs allocated about 57-64GB with 100% utilization. The 1900-step total is expected because Trainer uses `ceil(600/32)=19` optimizer steps per epoch for 100 epochs.
- interpretation: the previous 15000-step faces run should not be treated as a faithful original-batch reproduction. The active run now matches the original effective global batch and per-epoch optimizer-step semantics.
- next step: monitor the corrected faces run through first loss/eval logs and final checkpoint; then let guarded safety_unsafe launch.
