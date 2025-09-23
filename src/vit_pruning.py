import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
import copy

__all__ = [
    "prune_vit_mlp_width",
    "evaluate_top1",
    "prune_vit_attention_blocks"
]

@torch.no_grad()
def _gather_mlp_pairs(vit_model) -> List[Tuple[nn.Linear, nn.Linear]]:
    """Return (intermediate_dense, output_dense) linear layer pairs for each ViT encoder block.

    Supports huggingface ViTModel / ViTForImageClassification structures.
    """
    mlp_pairs = []
    encoder = getattr(vit_model, "vit", getattr(vit_model, "base_model", vit_model))
    encoder = encoder.encoder if hasattr(encoder, "encoder") else getattr(vit_model, "encoder")
    for layer in encoder.layer:  # type: ignore[attr-defined]
        inter_dense = layer.intermediate.dense
        out_dense = layer.output.dense
        mlp_pairs.append((inter_dense, out_dense))
    return mlp_pairs

@torch.no_grad()
def prune_vit_mlp_width(vit_model, sparsity: float, strategy: str = "l1", min_remaining: int = 256):
    """Width pruning of MLP intermediate dimension across all ViT blocks.

    Args:
        vit_model: HuggingFace ViT (ViTModel or ViTForImageClassification)
        sparsity: fraction (0-1) of intermediate neurons to remove.
        strategy: currently only 'l1'
        min_remaining: lower bound for remaining neurons after pruning each block.
    """
    assert 0.0 <= sparsity < 1.0, "sparsity must be in [0,1)"
    mlp_pairs = _gather_mlp_pairs(vit_model)

    for inter_dense, out_dense in mlp_pairs:
        W_int: torch.Tensor = inter_dense.weight  # [intermediate, hidden]
        B_int: torch.Tensor = inter_dense.bias    # [intermediate]
        W_out: torch.Tensor = out_dense.weight    # [hidden, intermediate]

        if strategy == "l1":
            importance = W_int.abs().sum(dim=1)
        else:
            raise ValueError(f"Unknown strategy {strategy}")

        n_channels = W_int.size(0)
        n_prune = int(n_channels * sparsity)
        if n_channels - n_prune < min_remaining:
            n_prune = max(0, n_channels - min_remaining)
        if n_prune == 0:
            continue

        keep_idx = torch.argsort(importance, descending=True)[: n_channels - n_prune]
        keep_idx, _ = torch.sort(keep_idx)

        new_W_int = W_int[keep_idx].clone()
        new_B_int = B_int[keep_idx].clone() if B_int is not None else None
        new_W_out = W_out[:, keep_idx].clone()

        hidden = W_int.size(1)
        new_intermediate = new_W_int.size(0)

        inter_dense.weight = nn.Parameter(new_W_int)
        if new_B_int is not None:
            inter_dense.bias = nn.Parameter(new_B_int)
        inter_dense.out_features = new_intermediate
        inter_dense.in_features = hidden

        out_dense.weight = nn.Parameter(new_W_out)
        out_dense.in_features = new_intermediate

    return vit_model

@torch.no_grad()
def evaluate_top1(model, dataloader, device: str = "cuda", max_batches: int | None = None, progress: bool = False):
    """Compute top-1 accuracy.

    Args:
        model: classification model with .logits output
        dataloader: iterable yielding dict with pixel_values, labels
        device: target device
        max_batches: limit number of batches (for quick estimation)
        progress: show tqdm progress bar
    """
    model.eval()
    correct = 0
    total = 0
    autocast_device = device if device.startswith("cuda") or device.startswith("mps") else "cpu"
    iterator = dataloader
    if progress:
        try:
            from tqdm.auto import tqdm  # lazy import
            iterator = tqdm(dataloader, total=(max_batches if max_batches is not None else None), desc="eval")
        except Exception:
            pass
    for i, batch in enumerate(iterator):
        if max_batches is not None and i >= max_batches:
            break
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type=autocast_device, enabled=True):
            out = model(pixel_values=pixel_values)
            logits = out.logits
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(1, total)


@torch.no_grad()
def _get_encoder(vit_model):
    """Get encoder module from different ViT model variants.
    
    Supports huggingface ViTModel / ViTForImageClassification structures.
    """
    if hasattr(vit_model, "vit"):
        # ViTForImageClassification has a vit attribute
        base = vit_model.vit
    elif hasattr(vit_model, "base_model"):
        # Some variants have base_model
        base = vit_model.base_model
    else:
        # Otherwise, assume model is already encoder-like
        base = vit_model
    
    # Get encoder from base
    encoder = base.encoder if hasattr(base, "encoder") else base
    return encoder
    
@torch.no_grad()
def _get_attention_blocks(vit_model) -> List:
    """Return list of attention block objects from ViT model.
    
    Supports huggingface ViTModel / ViTForImageClassification structures.
    """
    blocks = []
    encoder = _get_encoder(vit_model)
    
    for layer in encoder.layer:  # type: ignore[attr-defined]
        blocks.append(layer.attention)
    
    return blocks


@torch.no_grad()
def prune_vit_attention_blocks(
    vit_model, 
    sparsity: float, 
    dataloader=None, 
    device: str = "cuda", 
    batch_limit: int = 5,
    metric_fn=None
) -> Dict[str, Any]:
    """Remove entire attention blocks (Depth Pruning, Stage-2 of 2SSP).
    
    Args:
        vit_model: HuggingFace ViT model (ViTModel or ViTForImageClassification)
        sparsity: Fraction (0-1) of attention blocks to remove
        dataloader: Optional dataloader for metric-based selection
        device: Device for evaluation
        batch_limit: Max batches to process for evaluation
        metric_fn: Custom metric function to evaluate block importance
                  If None, uses top1 accuracy from evaluate_top1
    
    Returns:
        Dict with pruned model and pruning information
    """
    assert 0.0 <= sparsity < 1.0, "sparsity must be in [0,1)"
    
    # Create a copy of the model for calibration
    vit_model.eval()
    model_copy = copy.deepcopy(vit_model)
    model_copy.eval()
    
    # Get encoder and attention blocks
    encoder = _get_encoder(vit_model)
    encoder_copy = _get_encoder(model_copy)
    
    num_blocks = len(encoder.layer)
    num_to_prune = max(0, min(num_blocks - 1, int(num_blocks * sparsity)))
    
    if num_to_prune == 0:
        print("No attention blocks to prune based on sparsity.")
        return {"model": vit_model, "pruned_indices": [], "original_metrics": None, "final_metrics": None}
    
    # Store original encoder blocks for the copy model
    original_layer_modules = [copy.deepcopy(layer) for layer in encoder_copy.layer]
    
    # If no dataloader provided, use block position as a proxy for importance
    if dataloader is None:
        print("No dataloader provided. Using position-based heuristic for pruning.")
        # Heuristic: assume middle blocks are less important than edge blocks
        # This is a common pattern in transformers but might not be optimal
        importance_scores = [(i if i < num_blocks/2 else num_blocks - i) for i in range(num_blocks)]
        to_prune = sorted(range(num_blocks), key=lambda i: importance_scores[i])[:num_to_prune]
        original_metrics = None
        final_metrics = None
    else:
        # If dataloader provided, evaluate importance by impact on accuracy
        print(f"Evaluating {num_blocks} attention blocks using accuracy...")
        
        # Baseline metrics with the original model
        original_metrics = evaluate_top1(vit_model, dataloader, device, max_batches=batch_limit, progress=True)
        print(f"Baseline accuracy: {original_metrics:.4f}")
        
        # Evaluate importance of each block by temporarily modifying them
        impact_scores = []
        
        # Create identity function for attention that matches the interface of ViTAttention.forward
        def identity_attention(hidden_states, head_mask=None, output_attentions=False):
            # Simple identity function - return input without attention
            # For ViT: AttentionOutput needs to return the correct structure
            # Input shape: [batch_size, seq_len, hidden_size]
            
            outputs = (hidden_states,)
            
            # Добавим weights внимания, если требуется output_attentions
            if output_attentions:
                # Создаем фиктивные веса внимания нулевого размера
                # Обычно это тензор размера [batch_size, num_heads, seq_len, seq_len]
                batch_size = hidden_states.shape[0]
                seq_len = hidden_states.shape[1]
                dummy_attn = torch.zeros(
                    (batch_size, 1, seq_len, seq_len), 
                    device=hidden_states.device
                )
                outputs = outputs + (dummy_attn,)
                
            return outputs
            
        # Test removing each block individually
        for block_idx in range(num_blocks):
            print(f"Testing block {block_idx}/{num_blocks}...", end="\r")
            
            # Reset model to original state
            model_copy = copy.deepcopy(vit_model)
            model_copy.eval()
            encoder_copy = _get_encoder(model_copy)
            
            # Create a patch for the forward method of the attention module
            target_block = encoder_copy.layer[block_idx].attention
            
            # Save original method
            original_forward = target_block.forward
            
            # Сохраним оригинальный метод forward
            original_attention_forward = target_block.forward
            
            try:
                # Временно заменяем на identity функцию
                target_block.forward = identity_attention
                
                # Evaluate
                score = evaluate_top1(model_copy, dataloader, device, max_batches=batch_limit, progress=False)
                impact = original_metrics - score
                impact_scores.append(impact)
            except Exception as e:
                print(f"Error evaluating block {block_idx}: {e}")
                # Если произошла ошибка, предполагаем низкое влияние (меньше удалять)
                impact_scores.append(0.0)
            finally:
                # Восстанавливаем оригинальный метод, чтобы избежать утечек памяти
                if hasattr(target_block, 'forward'):
                    target_block.forward = original_attention_forward
            
            # Clean up to avoid memory leaks
            del model_copy, encoder_copy, target_block
            
        print(" " * 50, end="\r")  # Clear progress line
        
        # Print impact scores
        for i, impact in enumerate(impact_scores):
            print(f"Block {i}: Impact {impact:.4f}")
            
        # Select blocks with lowest impact to prune
        to_prune = sorted(range(num_blocks), key=lambda i: impact_scores[i])[:num_to_prune]
        print(f"Selected blocks to prune: {to_prune}")
    
    # Perform actual pruning on original model
    print(f"Performing actual pruning of {len(to_prune)} blocks...")
    
    # 1. Create new layer list without pruned blocks
    keep_indices = [i for i in range(num_blocks) if i not in to_prune]
    new_layers = nn.ModuleList([copy.deepcopy(encoder.layer[i]) for i in keep_indices])
    
    # 2. Replace the layer list with pruned version
    encoder.layer = new_layers
    
    # 3. Update model config to reflect new depth
    if hasattr(vit_model, "config"):
        vit_model.config.num_hidden_layers = len(new_layers)
    
    # Final evaluation if dataloader provided
    if dataloader is not None:
        final_metrics = evaluate_top1(vit_model, dataloader, device, max_batches=batch_limit, progress=True)
        print(f"Final accuracy after pruning: {final_metrics:.4f}")
        if original_metrics is not None:
            print(f"Accuracy change: {final_metrics - original_metrics:.4f}")
    else:
        final_metrics = None
    
    return {
        "model": vit_model, 
        "pruned_indices": to_prune, 
        "original_metrics": original_metrics,
        "final_metrics": final_metrics
    }
