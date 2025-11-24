import torch
from math import ceil
from time import time
from enum import Enum

# model configurator
# vit has q, k, v, att_proj, fc1, fc2 matrices to optimize
# any limitations on how these matrices can be optimized?
# fc1 is (emb, hid) dimension, fc2 is (hid, emb)
# q, k, v is (emb, h*hid), proj is (h*hid, emb)
class Attn:
    def __init__(self, emb_dim, head_dim, n_heads):
        self.q = torch.randn((n_heads*head_dim, emb_dim))
        self.k = torch.randn((n_heads*head_dim, emb_dim))
        self.v = torch.randn((n_heads*head_dim, emb_dim))
        self.p = torch.randn((n_heads*head_dim, emb_dim))
        
        self.fc1 = torch.randn((4*emb_dim, emb_dim))
        self.fc2 = torch.randn((emb_dim, 4*emb_dim))

class VisionModel:
    def __init__(self, emb_dim, head_dim, n_heads):
        self.de = emb_dim
        self.dh = head_dim
        self.nh = n_heads
        self.nb = 12

        self.bs = []
        for _ in range(self.nb):
            self.bs.append(Attn(emb_dim, head_dim, n_heads))

class PruningTypes(Enum):
    DEPTH = 0
    WIDTH = 1
    HEAD = 2 # valid only for attention
    NONE = 3 # means doesn't do pruning

class PruningInterface:
    # make sure to implement all functions and variables in this interface
    # the way how self.nn is used is up to you, just make sure to include it
    # it could be vit from transformers or timm, you decide
    # correctly assign pruning types for your method.
    # PruningTypes.None means pruning this structure is unsupported
    def __init__(self, model, pruning_dataloader):
        self.nn = model
        self.dl = pruning_dataloader
        self.att_prune_type = PruningTypes.DEPTH
        self.mlp_prune_type = PruningTypes.WIDTH

    # Your algorithm should return importance metric according to this format
    # format is designed in a way that is most efficient for that pruning type
    # code below is just a description for format of importance metrics
    # you can change it as you like. 
    # Lower importance means can be pruned earlier
    def fit(self):
        match self.att_prune_type:
            case PruningTypes.DEPTH:
                self.att_importance = torch.randn((self.nn.n_blocks,))
            case PruningTypes.HEAD :
                self.att_importance = [torch.randn((b.n_heads,)) for b in self.nn.blocks]
            case PruningTypes.WIDTH:
                # [q, k] and [v, proj] matrices are interrelated
                # their neurons should be pruned together
                # so we only need 2 tensors per block instead of 4
                self.att_importance = [[
                    torch.randn(b.q.shape[:1]), torch.randn(b.v.shape[:1])
                ] for b in self.nn.blocks]
                # width pruning in this case should reduce hidden dimension of 
                # q,k,v matrices. Not embedding dimension.
                # reducing embedding dimension is possible, but troublesome
                # let us know if your pruning algorithm does that
            case _:
                self.att_importance = None

        match self.mlp_prune_type:
            case PruningTypes.DEPTH:
                self.mlp_importance = torch.randn((self.nn.n_blocks,))
            case PruningTypes.WIDTH:
                # fc1 and fc2 are interrelated, their neurons are pruned together
                # so we only need 1 tensor per block instead of 2
                self.mlp_importance = [torch.randn(b.fc1.shape[:1]) for b in self.nn.blocks]
                # same consideration goes for reducing embedding dimension
                # just like in attention width pruning
                # let us know if your pruning algorithm does that
            case _:
                self.mlp_importance = None

        return self.att_importance, self.mlp_importance
        
# 5 types of pruning
# depth, width, head pruning(att), n:m block sparsity, unstructured
# interconnected groups for different kinds of pruning. these groups must be adjusted together
# DEPTH: [q, k, v, proj], [fc1, fc2]
# WIDTH: [q,k], [v,proj], [fc1, fc2]
# HEAD : [q, k], [v, proj]. this is a special form of width pruning
# BLOCK: [q, k]
class DepthPruning:
    def __init__(self, model, random=True):
        self.nn = model
        self.random = random

    def fit(self):
        if self.random:
            self.at_ord = torch.randperm(self.nn.nb)
            self.fc_ord = torch.randperm(self.nn.nb)
        else:
            self.at_ord = torch.arange(self.nn.nb)
            self.fc_ord = torch.arange(self.nn.nb)

    def mask_at(self, sparsity):
        masklist = self.at_ord[ :ceil(sparsity * self.nn.nb)]
        c = [i in masklist for i in torch.arange(self.nn.nb)]
        return [[
            torch.full(at.q.shape, 1 if c[i] else 0, dtype=torch.bool),
            torch.full(at.k.shape, 1 if c[i] else 0, dtype=torch.bool),
            torch.full(at.v.shape, 1 if c[i] else 0, dtype=torch.bool),
            torch.full(at.p.shape, 1 if c[i] else 0, dtype=torch.bool),
        ] for i, at in enumerate(self.nn.bs)]
    
    def mask_at_dw(self, sparsity):
        n = ceil(sparsity * self.nn.nb)
        return [[
            torch.full((1,), 1 if i in self.at_ord[:n] else 0, dtype=torch.bool)
        ] * 4 for i in range(self.nn.nb)]

    def mask_fc(self, sparsity):
        masklist = self.fc_ord[ :ceil(sparsity * self.nn.nb)]
        c = [i in masklist for i in torch.arange(self.nn.nb)]
        return [[
            torch.full(fc.fc1.shape, 1 if c[i] else 0, dtype=torch.bool),
            torch.full(fc.fc2.shape, 1 if c[i] else 0, dtype=torch.bool),
        ] for i, fc in enumerate(self.nn.bs)]
    
    def mask_fc_dw(self, sparsity):
        n = ceil(sparsity * self.nn.nb)
        return [[
            torch.full((1,), 1 if i in self.fc_ord[:n] else 0, dtype=torch.bool)
        ] * 2 for i in range(self.nn.nb)]

class WidthPruning:
    def __init__(self, model, random=True):
        self.nn = model
        self.random = random

    def fit(self):
        if self.random:
            self.qk_ord = [torch.randperm(  self.nn.dh) for _ in range(self.nn.nb)]
            self.vp_ord = [torch.randperm(  self.nn.dh) for _ in range(self.nn.nb)]
            self.fc_ord = [torch.randperm(4*self.nn.de) for _ in range(self.nn.nb)]
        else:
            self.qk_ord = torch.arange(  self.nn.dh).repeat(self.nn.nb, 1)
            self.vp_ord = torch.arange(  self.nn.dh).repeat(self.nn.nb, 1)
            self.fc_ord = torch.arange(4*self.nn.de).repeat(self.nn.nb, 1)
    
    def mask_at(self, sparsity):
        masks, n = [], ceil(sparsity * self.nn.dh)
        for i, at in enumerate(self.nn.bs):
            q = torch.zeros(at.q.shape, dtype=torch.bool)
            k = torch.zeros(at.k.shape, dtype=torch.bool)
            v = torch.zeros(at.v.shape, dtype=torch.bool)
            p = torch.zeros(at.p.shape, dtype=torch.bool)
            for h in range(self.nn.nh):
                q[self.qk_ord[i][:n] + h * self.nn.dh] = 1
                k[self.qk_ord[i][:n] + h * self.nn.dh] = 1
                v[self.vp_ord[i][:n] + h * self.nn.dh] = 1
                p[self.vp_ord[i][:n] + h * self.nn.dh] = 1
            masks.append([q, k, v, p])
        return masks
    
    def mask_at_dw(self, sparsity):
        masks, n = [], ceil(sparsity * self.nn.dh)
        for i, at in enumerate(self.nn.bs):
            q = torch.zeros(at.q.shape[:1], dtype=torch.bool)
            k = torch.zeros(at.k.shape[:1], dtype=torch.bool)
            v = torch.zeros(at.v.shape[:1], dtype=torch.bool)
            p = torch.zeros(at.p.shape[:1], dtype=torch.bool)
            for h in range(self.nn.nh):
                q[self.qk_ord[i][:n] + h * self.nn.dh] = 1
                k[self.qk_ord[i][:n] + h * self.nn.dh] = 1
                v[self.vp_ord[i][:n] + h * self.nn.dh] = 1
                p[self.vp_ord[i][:n] + h * self.nn.dh] = 1
            masks.append([q, k, v, p])
        return masks
    
    def mask_fc(self, sparsity):
        masks, n = [], ceil(sparsity * 4 * self.nn.de)
        for i, fc in enumerate(self.nn.bs):
            fc1 = torch.zeros(fc.fc1.shape, dtype=torch.bool)
            fc2 = torch.zeros(fc.fc2.shape, dtype=torch.bool)
            fc1[self.fc_ord[i][:n], :] = 1
            fc2[:, self.fc_ord[i][:n]] = 1
            masks.append([fc1, fc2])
        return masks
    
    def mask_fc_dw(self, sparsity):
        masks, n = [], ceil(sparsity * 4 * self.nn.de)
        for i, fc in enumerate(self.nn.bs):
            fc1 = torch.zeros(fc.fc1.shape[:1], dtype=torch.bool)
            fc2 = torch.zeros(fc.fc2.shape[1:], dtype=torch.bool)
            fc1[self.fc_ord[i][:n]] = 1
            fc2[self.fc_ord[i][:n]] = 1
            masks.append([fc1, fc2])
        return masks

class HeadPruning:
    def __init__(self, model):
        self.m = model

    def fit(self):
        self.orders = [[i for i in range(self.m.n_heads)]] * self.m.n_blocks

    def mask_at(self, sparsity):
        masks = []
        n = ceil(sparsity * self.m.n_heads)
        for i in range(self.m.n_blocks):
            q = torch.zeros(self.m.blocks[i].q.shape, dtype=torch.bool)
            k = torch.zeros(self.m.blocks[i].k.shape, dtype=torch.bool)
            v = torch.zeros(self.m.blocks[i].v.shape, dtype=torch.bool)
            p = torch.zeros(self.m.blocks[i].p.shape, dtype=torch.bool)

            for j in range(n):
                h, d = self.orders[i][j], self.m.head_dim
                q[h*d:(h+1)*d] = 1
                k[h*d:(h+1)*d] = 1
                v[h*d:(h+1)*d] = 1
                p[h*d:(h+1)*d] = 1

            masks.append([q, k, v, p])
        
        return masks

class BlockPruning:
    pass


class Auto2SSPInterface(PruningInterface):
    def __init__(self, model, pruning_dataloader, device=None, importance_mode="copy", batch_limit=5, min_remaining=256, error_policy="raise"):
        super().__init__(model, pruning_dataloader)
        # Declare pruning capabilities of the method
        self.att_prune_type = PruningTypes.DEPTH
        self.mlp_prune_type = PruningTypes.WIDTH

        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
        self.importance_mode = importance_mode
        self.batch_limit = batch_limit
        self.min_remaining = min_remaining
        # How to handle evaluation errors in 'copy' mode: 'raise' (default) or 'heuristic'
        self.error_policy = error_policy

        # Make local package importable and cache references to helpers from src.vit_pruning
        import sys
        from pathlib import Path
        ROOT = Path(__file__).resolve().parent
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src.vit_pruning import _get_encoder, evaluate_top1, _gather_mlp_pairs
        self._get_encoder = _get_encoder
        self._evaluate_top1 = evaluate_top1
        self._gather_mlp_pairs = _gather_mlp_pairs

        # Optional: activation-driven FFN importance (may be unavailable)
        try:
            from src.vit_pruning import _compute_ffn_activation_importance
            self._compute_ffn_activation_importance = _compute_ffn_activation_importance
        except Exception:
            self._compute_ffn_activation_importance = None

    def _num_blocks(self):
        enc = self._get_encoder(self.nn)
        if hasattr(enc, "layer"):
            return len(enc.layer)
        if hasattr(enc, "blocks"):
            return len(enc.blocks)
        raise AttributeError("Unsupported ViT model structure: expected encoder.layer or blocks")

    def _compute_mlp_importance(self):
        # Prefer calibration-driven activation L2 metric if dataloader provided
        if self.dl is not None and self._compute_ffn_activation_importance is not None:
            try:
                imps = self._compute_ffn_activation_importance(
                    self.nn, self.dl, device=self.device, batch_limit=self.batch_limit, progress=True
                )
                if isinstance(imps, list) and len(imps) > 0:
                    # List[Tensor[d_int]] per block
                    return [t.detach().to("cpu") for t in imps]
            except Exception:
                pass  # fallback to weight L1

        # Fallback: per-neuron L1 norm of fc1 weights
        pairs = self._gather_mlp_pairs(self.nn)
        out = []
        for inter_dense, _ in pairs:
            W_int = inter_dense.weight  # [intermediate, hidden]
            imp = W_int.abs().sum(dim=1).detach().to("cpu")
            out.append(imp)
        return out

    def _compute_att_depth_importance(self):
        B = self._num_blocks()
        # Heuristic if no dataloader or requested explicitly
        if self.importance_mode.lower() == "heuristic" or self.dl is None:
            # center-most blocks get highest importance; edges lowest
            scores = [(i if i < B/2 else B - i) for i in range(B)]
            return torch.tensor(scores, dtype=torch.float32)

        # Copy-replace evaluation impact
        import copy
        import torch.nn as nn

        class HFAttentionBypass(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, hidden_states, head_mask=None, output_attentions: bool = False, *args, **kwargs):
                zeros = torch.zeros_like(hidden_states)
                if output_attentions:
                    return (zeros, None)
                return (zeros,)

        class TimmAttentionBypass(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x, *args, **kwargs):
                return torch.zeros_like(x)

        heuristic_scores = [(i if i < B/2 else B - i) for i in range(B)]

        # Baseline accuracy
        try:
            baseline = float(self._evaluate_top1(self.nn, self.dl, device=self.device, max_batches=self.batch_limit, progress=False))
        except Exception as e:
            if getattr(self, "error_policy", "raise") == "raise":
                raise e
            else:
                # Fallback to heuristic for all blocks to avoid mixing incomparable scales
                return torch.tensor(heuristic_scores, dtype=torch.float32)

        enc = self._get_encoder(self.nn)
        impacts = []
        for i in range(B):
            try:
                mcopy = copy.deepcopy(self.nn)
                enc_copy = self._get_encoder(mcopy)
                if hasattr(enc_copy, "layer") and hasattr(enc_copy.layer[i], "attention"):
                    enc_copy.layer[i].attention = HFAttentionBypass()
                elif hasattr(enc_copy, "blocks") and hasattr(enc_copy.blocks[i], "attn"):
                    enc_copy.blocks[i].attn = TimmAttentionBypass()
                score = float(self._evaluate_top1(mcopy, self.dl, device=self.device, max_batches=self.batch_limit, progress=False))
                impact = max(0.0, baseline - score)
                impacts.append(impact)
            except Exception as e:
                if getattr(self, "error_policy", "raise") == "raise":
                    raise e
                else:
                    # Fallback to heuristic for all blocks
                    return torch.tensor(heuristic_scores, dtype=torch.float32)

        return torch.tensor(impacts, dtype=torch.float32)

    def fit(self):
        self.att_importance = self._compute_att_depth_importance()
        self.mlp_importance = self._compute_mlp_importance()
        return self.att_importance, self.mlp_importance

def count_pruned(masks):
    pruned, total = 0, 0
    for b in masks: 
        for m in b: 
            pruned += m.sum()
            total  += m.numel()
    return pruned / total

def conjunction(m_a, m_b, n_submasks):
    for i, m in enumerate(m_b):
        for j in range(n_submasks): m_a[i][j] &= m[j]
    return m_a

# conjunction algorithm
def mask_conjunction(model, methods, target, init_sparsity=None, random=True):
    '''
        methods: tuple (method_class, prunes_att, prunes_mlp)
        init_sparsity: tuple (attention, mlp) initial sparsity, for optimization
    '''
    sparsity_step = 2e-3
    atspinit, fcspinit = [target]*2 if init_sparsity is None else init_sparsity

    pruners = [m[0](model, random) for m in methods]
    for p in pruners: p.fit()

    # att masking phase
    # blocks x (q, k, v, p)
    at_sparsity, ef_sparsity, n_matrices = atspinit, 0, 4
    while ef_sparsity < target:
        masks = [p.mask_at(at_sparsity) for p, f in zip(pruners, methods) if f[1]]
        
        conjs = masks[0]
        for m in masks[1:]: conjs = conjunction(conjs, m, n_matrices)
        
        ef_sparsity = count_pruned(conjs)

        if at_sparsity >= 1: break
        at_sparsity += sparsity_step
        if at_sparsity > 1: at_sparsity = 1
    at_ef_sparsity = ef_sparsity

    # mlp masking phase
    # blocks x (fc1, fc2)
    fc_sparsity, ef_sparsity, n_matrices = fcspinit, 0, 2
    while ef_sparsity < target:
        masks = [p.mask_fc(fc_sparsity) for p, f in zip(pruners, methods) if f[2]]
        
        conjs = masks[0]
        for m in masks[1:]: conjs = conjunction(conjs, m, 2)
        
        ef_sparsity = count_pruned(conjs)

        if fc_sparsity >= 1: 
            break
        fc_sparsity += sparsity_step
        if fc_sparsity > 1: fc_sparsity = 1
    fc_ef_sparsity = ef_sparsity

    return at_sparsity, fc_sparsity, float(at_ef_sparsity), float(fc_ef_sparsity)


def test_unstr_mask_conj(emb_dim, head_dim, num_heads, num_steps, methods, random=True):
    targets, ats, fcs, at_ef, fc_ef = [0], [0], [0], [0], [0]
    step = 1 / num_steps
    for i in range(num_steps):
        target = (i+1) * step
        print(f"Sparsity {target*100:3.0f}%: ", end="")
        results = mask_conjunction(
            VisionModel(emb_dim, head_dim, num_heads),
            methods, target, (ats[-1], fcs[-1]), random
        )
        targets.append(round(target, 5))
        ats.append(round(results[0], 5))
        fcs.append(round(results[1], 5))
        at_ef.append(round(results[2], 5))
        fc_ef.append(round(results[3], 5))
        print(f"Att: {results[0]*100:4.1f}%    Att Eff: {results[2]*100:4.1f}%    MLP: {results[1]*100:4.1f}%    MLP Eff: {results[3]*100:4.1f}%")
    return targets, ats, fcs, at_ef, fc_ef

if __name__ == "__main__":
    start = time()
    sps, ats, fcs, at_ef, fc_ef = test_unstr_mask_conj(
        768, 64, 12, 100,
        [[DepthPruning, True, True], [WidthPruning, True, True]], False
    )
    end   = time()

    print(f'Time to run test: {round(end-start, 3):.3f} s', end='\n\n')
    print(sps)
    print(ats)
    print(fcs)
    print(at_ef)
    print(fc_ef)
