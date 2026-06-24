
## 2026-06-24 - Tianyang Janus-Pro smoke training unblock

- Symptom: Janus-Pro smoke training blocked first on an interactive SwanLab login prompt, then single-GPU training OOMed at optimizer step with about 79GB allocated.
- Cause: transformers auto-detected swanlab in the tianyang environment; full Janus-Pro language-model training does not fit on one A100 80GB with Adam optimizer state.
- Repair: set SWANLAB_MODE=disabled and launch via 4 GPU DeepSpeed ZeRO-2 using configs/concepts_janus/accelerate_config.yaml with --num_processes 4.
- Verification: runs/janus_concepts_smoke_4gpu_20260624.log completed 168/168 steps and saved model/finetuned_janus/concepts_smoke/janus-concepts-smoke-4gpu-20260624.

## 2026-06-24 - Tianyang safety image cache materialization unblock

- Symptom: tianyang direct Unsplash download timed out earlier, and two current Unsplash download endpoints returned bytes whose sha256 did not match misc/safety_images_meta.jsonl.
- Cause: the safety benchmark requires exact cached image bytes; some Unsplash image URLs need the metadata ixid on the images.unsplash.com URL, and one image needs the no-crop variant to reproduce the recorded hash.
- Repair: downloaded the cache locally with sha256 verification, added scripts/download_safety_images.py to try the original download link plus ixid/no-crop fallback URLs, then rsynced the verified cache to data/safety_images_cache on tianyang.
- Verification: tianyang verified all 50 raw files against misc/safety_images_meta.jsonl, and modal_aphasia.data.generate_safety_dataset saved data/safety_images with 50 records.

## 2026-06-24 - Tianyang Janus benchmark queue guard

- Symptom: queued faces and safety_unsafe tmux stages would start when the previous process/session ended, even if the previous training failed before saving a final checkpoint.
- Cause: the first queue used process/session disappearance as the only readiness condition.
- Repair: replaced only the waiting tmux sessions with guarded scripts under /tmp that check `Finished training` plus final checkpoint files before launching the next benchmark family; left the active concepts training process untouched.
- Verification: `janus-faces-queued` and `janus-safety-queued` are running guarded scripts and waiting; `janus-concepts-full` continued progressing to about step 955/4014 after the queue replacement.

## 2026-06-24 - Tianyang Janus faces effective-batch correction

- Symptom: the queued faces Janus-Pro run used `--num_processes 4 --per-device-train-batch-size 1 --gradient-accumulation-steps 1 --num-epochs 100`, producing 15000 optimizer steps instead of matching the original faces config effective global batch of 32.
- Cause: the tianyang memory-safe launch reduced per-device batch size and GPU count but did not compensate with gradient accumulation.
- Repair: stopped the wrong-effective-batch faces run, archived its log as `runs/janus_faces_queued_20260624.aborted_effective_batch4_20260624_153001.log`, archived the waiting safety log, moved the partial faces output aside, and restarted guarded faces with `--num_processes 4 --per-device-train-batch-size 1 --gradient-accumulation-steps 8 --eval-steps 20`.
- Verification: the corrected faces process is running with `--gradient-accumulation-steps 8`; the Trainer progress total is `0/1900`, matching the original global-batch-32 optimizer-step semantics under per-epoch ceiling, and all 4 A100 GPUs showed active memory allocation and 100% utilization when checked.
