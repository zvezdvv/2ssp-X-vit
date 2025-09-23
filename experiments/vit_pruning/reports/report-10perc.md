# 2SSP ViT Pruning Report (20250924-000210)

## Config
- model: google/vit-base-patch16-224
- target_sparsity: 0.1
- freeze_backbone: True
- replace_classifier: False
- use_adapter: False
- adapter_reduction: None
- eval_batches: 5
- min_remaining: 512
- cifar_load: True

## Parameters reduction
- Stage-1 (Width): 85.81M -> 84.31M (1.7%)
- Stage-2 (Depth): 84.31M -> 77.35M (8.3%)
- Final result: 85.81M -> 77.35M (9.9%)

## Latency
- Baseline: 43.82 ms
- Stage-1 (Width): 43.86 ms (0.1%)
- Stage-2 (Depth): 43.26 ms (-1.4%)
- Final change: -1.3%

## Accuracy
- Baseline: 0.9375
- Stage-1 (Width): 0.9281 (drop: 1.0%)
- Stage-2 (Depth): 0.9031 (drop: 2.69%)
- Final change: 3.67%

## Auto-allocation plan
- Target sparsity: 0.1
- Blocks total: 12
- Blocks to prune (Stage-2): 1 (0.0833)
- Per-block neurons to prune (Stage-1): 81
- Estimated total removed params: 8581836
- Estimation error (params): 1201

## Artifacts
- pruned_block_indices: [7]
