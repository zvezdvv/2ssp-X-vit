# 2SSP ViT Pruning Report (20250923-223804)

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
- Stage-1 (Width): 85.81M -> 72.75M (15.2%)
- Stage-2 (Depth): 72.75M -> 54.75M (24.7%)
- Final result: 85.81M -> 54.75M (36.2%)

## Latency
- Baseline: 43.79 ms
- Stage-1 (Width): 40.17 ms (-8.3%)
- Stage-2 (Depth): 31.61 ms (-21.3%)
- Final change: -27.8%

## Accuracy
- Baseline: 0.9375
- Stage-1 (Width): 0.6625 (drop: 29.33%)
- Stage-2 (Depth): 0.5031 (drop: 24.06%)
- Final change: 46.33%

## Auto-allocation plan
- Target sparsity: 0.4
- Blocks total: 12
- Blocks to prune (Stage-2): 3 (0.2500)
- Per-block neurons to prune (Stage-1): 708
- Estimated total removed params: 34321968
- Estimation error (params): 570

## Artifacts
- pruned_block_indices: [4, 10, 11]
