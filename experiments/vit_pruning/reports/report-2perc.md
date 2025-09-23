# 2SSP ViT Pruning Report (20250923-213804)

## Config
- model: google/vit-base-patch16-224
- target_sparsity: 0.02
- freeze_backbone: True
- replace_classifier: True
- use_adapter: False
- adapter_reduction: None
- eval_batches: 5
- min_remaining: 512
- cifar_load: True

## Parameters reduction
- Stage-1 (Width): 85.81M -> 84.09M (2.0%)
- Stage-2 (Depth): 84.09M -> 84.09M (0.0%)
- Final result: 85.81M -> 84.09M (2.0%)

## Latency
- Baseline: 46.58 ms
- Stage-1 (Width): 45.22 ms (-2.9%)
- Stage-2 (Depth): 47.57 ms (5.2%)
- Final change: 2.1%

## Accuracy
- Baseline: 0.9375
- Stage-1 (Width): 0.925 (drop: 1.33%)
- Stage-2 (Depth): 0.925 (drop: 0.0%)
- Final change: 1.33%

## Auto-allocation plan
- Target sparsity: 0.02
- Blocks total: 12
- Blocks to prune (Stage-2): 0 (0.0000)
- Per-block neurons to prune (Stage-1): 93
- Estimated total removed params: 1715292
- Estimation error (params): 835

## Artifacts
- pruned_block_indices: []
- adapter_path: /Users/vladimirzvyozdkin/Studies_Hildesheim/SRP/2SSP implementation on VIT/2SSP/experiments/vit_pruning/artifacts/20250923-213804/adapter.pt
