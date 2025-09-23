# 2SSP ViT Pruning Report (20250923-220536)

## Config
- model: google/vit-base-patch16-224
- target_sparsity: 0.05
- freeze_backbone: True
- replace_classifier: False
- use_adapter: False
- adapter_reduction: None
- eval_batches: 5
- min_remaining: 512
- cifar_load: True

## Parameters reduction
- Stage-1 (Width): 85.81M -> 81.51M (5.0%)
- Stage-2 (Depth): 81.51M -> 81.51M (0.0%)
- Final result: 85.81M -> 81.51M (5.0%)

## Latency
- Baseline: 43.62 ms
- Stage-1 (Width): 42.45 ms (-2.7%)
- Stage-2 (Depth): 45.38 ms (6.9%)
- Final change: 4.0%

## Accuracy
- Baseline: 0.9375
- Stage-1 (Width): 0.8875 (drop: 5.33%)
- Stage-2 (Depth): 0.8875 (drop: 0.0%)
- Final change: 5.33%

## Auto-allocation plan
- Target sparsity: 0.05
- Blocks total: 12
- Blocks to prune (Stage-2): 0 (0.0000)
- Per-block neurons to prune (Stage-1): 233
- Estimated total removed params: 4297452
- Estimation error (params): 7135

## Artifacts
- pruned_block_indices: []
