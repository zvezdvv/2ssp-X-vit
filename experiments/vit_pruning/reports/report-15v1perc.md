# 2SSP ViT Pruning Report (20250924-233553)

## Config
- model: google/vit-base-patch16-224
- target_sparsity: 0.15
- freeze_backbone: True
- replace_classifier: False
- use_adapter: False
- adapter_reduction: None
- eval_batches: 5
- min_remaining: 512
- cifar_load: True

## Parameters reduction
- Stage-1 (Width): 85.81M -> 80.01M (6.7%)
- Stage-2 (Depth): 80.01M -> 73.41M (8.3%)
- Final result: 85.81M -> 73.41M (14.4%)

## Latency
- Baseline: 45.03 ms
- Stage-1 (Width): 45.25 ms (0.5%)
- Stage-2 (Depth): 41.39 ms (-8.5%)
- Final change: -8.1%

## Accuracy
- Baseline: 0.9375
- Stage-1 (Width): 0.8875 (drop: 5.33%)
- Stage-2 (Depth): 0.8438 (drop: 4.93%)
- Final change: 10.0%

## Auto-allocation plan
- Target sparsity: 0.15
- Blocks total: 12
- Blocks to prune (Stage-2): 1 (0.0833)
- Per-block neurons to prune (Stage-1): 314
- Estimated total removed params: 12879288
- Estimation error (params): 8336

## Artifacts
- pruned_block_indices: [7]
