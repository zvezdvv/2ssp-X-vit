# 2SSP ViT Pruning Report (20250925-001717)

## Config
- model: google/vit-base-patch16-224
- target_sparsity: 0.4
- freeze_backbone: True
- replace_classifier: False
- use_adapter: False
- adapter_reduction: None
- eval_batches: 5
- min_remaining: 512
- cifar_load: True

## Parameters reduction
- Stage-1 (Width): 85.81M -> 79.83M (7.0%)
- Stage-2 (Depth): 79.83M -> 53.47M (33.0%)
- Final result: 85.81M -> 53.47M (37.7%)

## Latency
- Baseline: 43.5 ms
- Stage-1 (Width): 41.48 ms (-4.6%)
- Stage-2 (Depth): 30.1 ms (-27.4%)
- Final change: -30.8%

## Accuracy
- Baseline: 0.9375
- Stage-1 (Width): 0.8719 (drop: 7.0%)
- Stage-2 (Depth): 0.3156 (drop: 63.8%)
- Final change: 66.33%

## Auto-allocation plan
- Target sparsity: 0.4
- Blocks total: 12
- Blocks to prune (Stage-2): 4 (0.3333)
- Per-block neurons to prune (Stage-1): 324
- Estimated total removed params: 34327344
- Estimation error (params): 4806

## Artifacts
- pruned_block_indices: [6, 7, 8, 9]
