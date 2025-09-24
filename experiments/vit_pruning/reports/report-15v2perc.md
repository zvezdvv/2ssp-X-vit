# 2SSP ViT Pruning Report (20250924-234848)

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
- Stage-1 (Width): 85.81M -> 85.81M (0.0%)
- Stage-2 (Depth): 85.81M -> 71.63M (16.5%)
- Final result: 85.81M -> 71.63M (16.5%)

## Latency
- Baseline: 43.93 ms
- Stage-1 (Width): 45.18 ms (2.9%)
- Stage-2 (Depth): 38.77 ms (-14.2%)
- Final change: -11.7%

## Accuracy
- Baseline: 0.9375
- Stage-1 (Width): 0.9375 (drop: 0.0%)
- Stage-2 (Depth): 0.85 (drop: 9.33%)
- Final change: 9.33%

## Auto-allocation plan
- Target sparsity: 0.15
- Blocks total: 12
- Blocks to prune (Stage-2): 2 (0.1667)
- Per-block neurons to prune (Stage-1): 0
- Estimated total removed params: 14175744
- Estimation error (params): 1304792

## Artifacts
- pruned_block_indices: [6, 8]
