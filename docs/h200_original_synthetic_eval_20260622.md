# h200 Original Synthetic Evaluation, 2026-06-22

This note records the finalized Emu3.5 modal_aphasia original synthetic evaluation artifacts.

## Protocol

- Text benchmark: `InferenceTextOutputBuilder.build_concepts_description_mc`
- Image benchmark: `InferenceImageOutputBuilder.build_synthetic_concepts`
- Image grading: modal_aphasia synthetic classifier for color, pattern, position, and shape
- Checkpoints:
  - `text_adv`: `model/finetuned/modal_aphasia_symmetric/emu35_lora_core_c/checkpoint-200`
  - `image_adv_original`: `model/finetuned/modal_aphasia_original/synth_concepts_lora_a6000`

## Final Outputs

- `outputs/eval/h200_original_synthetic_text_adv_8gpu_merged_20260622_1228`
- `outputs/eval/h200_original_synthetic_image_adv_dynamic_merged_20260622_2055`
- Bundle: `outputs/eval/final_bundles/h200_original_synthetic_text_image_adv_20260622.tar.gz`
- Bundle SHA256: `65e90b047ada25034d30942d1d66eacc9813e130bff52e59dff16e2a89b91b71`

## Results

| condition | Text MC overall | fake-to-real | real-to-fake | Image all-attr | color | pattern | position | shape |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| text_adv | 35/44 = 79.5% | 19/22 = 86.4% | 16/22 = 72.7% | 0/840 = 0.0% | 5.7% | 3.3% | 0.5% | 3.7% |
| image_adv_original | 7/44 = 15.9% | 2/22 = 9.1% | 5/22 = 22.7% | 12/840 = 1.43% | 34.0% | 51.2% | 34.2% | 15.1% |

`image_all_correct && text_wrong`:

- `text_adv`: 0/840
- `image_adv_original`: 12/840

## Execution Notes

The final h200 run used dynamic image resharding:

1. Run text and image shards concurrently across GPUs 0-7.
2. After text shards finished, recover completed image PNGs from old workers.
3. Split remaining image sample IDs into 16 residual shards.
4. Run two image workers per GPU.
5. Merge recovered, residual, and text-only shards with `scripts/merge_emu35_original_synthetic_shards.py`.

The dynamic split manifest is included in the final bundle.
