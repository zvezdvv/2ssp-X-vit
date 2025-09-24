# 2SSP ViT Pruning Report (20250925-000204)

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
- Stage-1 (Width): 85.81M -> 85.81M (0.0%)
- Stage-2 (Depth): 85.81M -> 57.45M (33.0%)
- Final result: 85.81M -> 57.45M (33.0%)

## Latency
- Baseline: 43.05 ms
- Stage-1 (Width): 44.95 ms (4.4%)
- Stage-2 (Depth): 31.53 ms (-29.9%)
- Final change: -26.8%

## Accuracy
- Baseline: 0.9375
- Stage-1 (Width): 0.9375 (drop: 0.0%)
- Stage-2 (Depth): 0.4125 (drop: 56.0%)
- Final change: 56.0%

## Auto-allocation plan
- Target sparsity: 0.3
- Blocks total: 12
- Blocks to prune (Stage-2): 4 (0.3333)
- Per-block neurons to prune (Stage-1): 0
- Estimated total removed params: 28351488
- Estimation error (params): 2609584

## Artifacts
- pruned_block_indices: [4, 6, 7, 8]
