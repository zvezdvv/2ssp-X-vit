# 2SSP ViT Pruning Report (20250923-232427)

## Config
- model: google/vit-base-patch16-224
- target_sparsity: 0.2
- freeze_backbone: True
- replace_classifier: False
- use_adapter: False
- adapter_reduction: None
- eval_batches: 5
- min_remaining: 512
- cifar_load: True

## Parameters reduction
- Stage-1 (Width): 85.81M -> 82.82M (3.5%)
- Stage-2 (Depth): 82.82M -> 69.14M (16.5%)
- Final result: 85.81M -> 69.14M (19.4%)

## Latency
- Baseline: 44.53 ms
- Stage-1 (Width): 43.18 ms (-3.0%)
- Stage-2 (Depth): 40.66 ms (-5.8%)
- Final change: -8.7%

## Accuracy
- Baseline: 0.9375
- Stage-1 (Width): 0.9062 (drop: 3.33%)
- Stage-2 (Depth): 0.7969 (drop: 12.07%)
- Final change: 15.0%

## Auto-allocation plan
- Target sparsity: 0.2
- Blocks total: 12
- Blocks to prune (Stage-2): 2 (0.1667)
- Per-block neurons to prune (Stage-1): 162
- Estimated total removed params: 17163672
- Estimation error (params): 2403

## Artifacts
- pruned_block_indices: [7, 8]
