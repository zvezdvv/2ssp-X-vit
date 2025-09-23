ViT pruning experiment (google/vit-base-patch16-224)

Overview
- Implements a Two-Stage Structured Pruning (2SSP) workflow for ViT:
  - Stage-1 (Width): prune MLP intermediate neurons across all blocks
  - Stage-2 (Depth): remove entire transformer blocks (depth pruning)
- Simplified configuration: specify a single global TARGET_SPARSITY and the system auto-balances width/depth to match the target with high accuracy.
- Consolidated reporting: parameters, latency, accuracy per stage and final, saved to experiments/vit_pruning/reports/.
- CIFAR-10 adapter/head artifacts can be saved automatically for reuse.
- Memory-friendly: you can limit CIFAR-10 train/test splits to avoid OOM.
- Controlled persistence: by default pruned model is NOT saved; saving is opt-in.

Quick start (recommended)
1) Install dependencies
- From project root:
  - pip install -r requirements.txt
  - Additionally for ViT/vision: pip install transformers datasets timm accelerate torchvision

2) Run auto 2SSP end-to-end (single TARGET_SPARSITY)
- Basic run with quick CIFAR-10 evaluation, head-only tuning (1 epoch), and save adapter:
  - python experiments/vit_pruning/auto_2ssp.py --target 0.143 --load-cifar --replace-classifier --do-finetune --freeze-backbone --save-adapter
- Memory-friendly run (limit CIFAR to 2% train / 5% test like in your notebook):
  - python experiments/vit_pruning/auto_2ssp.py --target 0.143 --load-cifar --cifar-train-pct 0.02 --cifar-test-pct 0.05 --replace-classifier --do-finetune --freeze-backbone
- Persist pruned model only when you want:
  - python experiments/vit_pruning/auto_2ssp.py --target 0.143 --load-cifar --cifar-train-pct 0.02 --cifar-test-pct 0.05 --replace-classifier --save-pruned-model
    - Will save to experiments/vit_pruning/pruned_models/<run_id> to avoid touching HF cache.

Parameters (auto_2ssp.py)
- --target: desired global sparsity (0..1), e.g. 0.143 for 14.3%
- --model: HF model id (default google/vit-base-patch16-224)
- --load-cifar: enables quick CIFAR-10 accuracy evaluation (by default uses 25%/25%)
- --cifar-train-pct: fraction of CIFAR-10 train split, e.g. 0.02 for 2%
- --cifar-test-pct: fraction of CIFAR-10 test split, e.g. 0.05 for 5%
- --replace-classifier: replace head with 10-class linear layer
- --use-adapter: alternatively use adapter bottleneck (instead of replacing head)
- --do-finetune: quick head/adapter fine-tuning (default 1 epoch)
- --freeze-backbone: keep ViT backbone frozen (train only head/adapter)
- --save-adapter: saves the trained head/adapter for later reuse
- --min-remaining: minimum MLP intermediate size per block after width pruning (default 512)
- --eval-batches: number of batches for quick accuracy estimation (default 5)
- --save-pruned-model: if set, persists the pruned model to --pruned-output-dir (otherwise pruned model is NOT saved)
- --pruned-output-dir: directory to save pruned model (default experiments/vit_pruning/pruned_models)

3) Outputs
- Reports: experiments/vit_pruning/reports/report-<run_id>.json and .md
  - Contains:
    - Parameters reduction:
      - Stage-1 (Width): P0 -> P1 and %
      - Stage-2 (Depth): P1 -> P2 and %
      - Final: P0 -> P2 and %
    - Latency (baseline, stage-1, stage-2 and deltas)
    - Accuracy (baseline, stage-1, stage-2 and drops)
    - Auto-allocation plan summary (chosen blocks to prune, per-block neurons to prune, estimated param removal)
- Artifacts: experiments/vit_pruning/artifacts/<run_id>/adapter.pt (if --save-adapter), plus pruned block indices in the report.
- Pruned model dir (only if you passed --save-pruned-model): experiments/vit_pruning/pruned_models/<run_id>.

Using from the notebook (manual flow)
- Your original way to limit CIFAR is fully supported and recommended for low memory:
  - train_ds = load_dataset('cifar10', split='train[:2%]')
  - test_ds  = load_dataset('cifar10', split='test[:5%]')
- If you prefer to call the new planner/pruners inside the notebook:
  Example cell snippet:
  - from src.vit_pruning import plan_2ssp_allocation, prune_vit_mlp_width, prune_vit_attention_blocks, count_total_params, save_report, compute_actual_sparsity, _get_encoder
  - TARGET_SPARSITY = 0.143
  - plan = plan_2ssp_allocation(base_model, TARGET_SPARSITY, min_remaining=512)
  - # Stage-1 (width)
  - B = len(_get_encoder(base_model).layer)
  - base_model = prune_vit_mlp_width(base_model, n_to_prune_per_block=[plan.per_block_neurons_to_prune]*B, min_remaining=512)
  - # Stage-2 (depth)
  - depth_fraction = plan.blocks_to_prune / max(1, B)
  - result = prune_vit_attention_blocks(base_model, sparsity=depth_fraction, dataloader=test_loader, device=device, batch_limit=5)
  - base_model = result["model"]
- To avoid persisting a pruned model from the notebook: do not call save_pretrained; the model remains in memory for measurement only. If you DO want to keep it, call:
  - base_model.save_pretrained('experiments/vit_pruning/pruned_models/<your_tag>')

Why per-stage “sparsity percentages” differ from global parameter reduction
- If you set “10% width sparsity” naively and apply it only to MLP layers, you are zeroing 10% of MLP neurons, but MLP weights are only a portion of the whole model parameters. The global reduction becomes smaller than 10%.
- Similarly, “10% depth sparsity” (e.g., remove 10% of blocks) maps to parameter reduction proportional to the average parameters per block. Due to discrete removal (whole blocks, integer neurons) and architectural constraints (hidden sizes, heads), exact matching is not guaranteed.
- The auto-allocation planner minimizes this mismatch by computing parameter budgets and distributing pruning across both stages to hit the target global sparsity closely, with local search tweaks to reduce residual error.

Notes on accuracy/latency
- Structured pruning gives real compute reduction and latency gains (vs. unstructured masking).
- Accuracy typically drops more during depth pruning. The planner’s initial allocation often prunes a small number of blocks and compensates the rest with width pruning to preserve more accuracy at the same total sparsity.

Saving the CIFAR-10 adapter
- When running the script with --save-adapter, the trained head/adapter is saved to:
  - experiments/vit_pruning/artifacts/<run_id>/adapter.pt
- The saved payload includes:
  - classifier/adapter state_dict
  - classifier_type, num_labels, hidden_size, timestamp
  - extra metadata (model name, target sparsity, head/adapter config)

Examples
- 14.3% global sparsity, small CIFAR-10 loaders, head-only training 1 epoch:
  - python experiments/vit_pruning/auto_2ssp.py --target 0.143 --load-cifar --cifar-train-pct 0.02 --cifar-test-pct 0.05 --replace-classifier --do-finetune --freeze-backbone --save-adapter
- 25% sparsity, adapter head, quick eval, do not persist pruned model:
  - python experiments/vit_pruning/auto_2ssp.py --target 0.25 --load-cifar --cifar-train-pct 0.02 --cifar-test-pct 0.05 --use-adapter
- Persist pruned model for reuse later:
  - python experiments/vit_pruning/auto_2ssp.py --target 0.25 --load-cifar --cifar-train-pct 0.02 --cifar-test-pct 0.05 --replace-classifier --save-pruned-model

Troubleshooting
- If accuracy is extremely low with REPLACE_CLASSIFIER and no fine-tuning, remember the new head is randomly initialized. Use --do-finetune and optionally --freeze-backbone for a quick head-only adaptation.
- If the achieved sparsity differs slightly from the target (e.g., ±0.2–0.5%), this is due to discrete constraints (integer blocks/neurons). The planner performs a small local search to minimize this error.
- If memory is tight, reduce --cifar-train-pct and --cifar-test-pct, reduce batch sizes in your notebook loaders, or run without --load-cifar to skip accuracy measurements (still get params/latency reports).

Files of interest
- experiments/vit_pruning/auto_2ssp.py — command-line entry point implementing the simplified single-parameter workflow
- src/vit_pruning.py — primitives:
  - plan_2ssp_allocation — compute Stage-1/Stage-2 amounts for a given target
  - prune_vit_mlp_width — width pruning (supports exact per-block neuron counts)
  - prune_vit_attention_blocks — depth pruning by removing whole blocks, optionally evaluating impact
  - evaluate_top1 — quick top-1 accuracy estimation
  - save_cifar_adapter, save_report — for artifacts and consolidated reporting
