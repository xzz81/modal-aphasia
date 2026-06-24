
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
