# 2SSP ViT Pruning Report (20250923-225428)

## Config
- model: google/vit-base-patch16-224
- target_sparsity: 0.3
- freeze_backbone: True
- replace_classifier: False
- use_adapter: False
- adapter_reduction: None
- eval_batches: 5
- min_remaining: 512
- cifar_load: True

## Parameters reduction
- Stage-1 (Width): 85.81M -> 74.24M (13.5%)
- Stage-2 (Depth): 74.24M -> 61.99M (16.5%)
- Final result: 85.81M -> 61.99M (27.8%)

## Latency
- Baseline: 44.24 ms
- Stage-1 (Width): 39.96 ms (-9.7%)
- Stage-2 (Depth): 36.51 ms (-8.6%)
- Final change: -17.5%

## Accuracy
- Baseline: 0.9375
- Stage-1 (Width): 0.7188 (drop: 23.33%)
- Stage-2 (Depth): 0.5813 (drop: 19.13%)
- Final change: 38.0%

## Auto-allocation plan
- Target sparsity: 0.3
- Blocks total: 12
- Blocks to prune (Stage-2): 2 (0.1667)
- Per-block neurons to prune (Stage-1): 627
- Estimated total removed params: 25740132
- Estimation error (params): 1772

## Artifacts
- pruned_block_indices: [4, 6]
