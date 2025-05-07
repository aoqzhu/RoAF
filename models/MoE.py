import torch
import torch.nn.functional as F
from torch import Tensor, nn
def load_balanced_loss(router_probs, expert_mask):
    num_experts = expert_mask.size(-1)

    density = torch.mean(expert_mask, dim=0)
    density_proxy = torch.mean(router_probs, dim=0)
    loss = torch.mean(density_proxy * density) * (num_experts ** 2)

    return loss

class Gate(nn.Module):
    def __init__(
            self,
            dim,
            num_experts: int,
            top_k: int,
            capacity_factor: float = 1.0,
            epsilon: float = 1e-6,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor, use_aux_loss=True):
        # Compute gate scores
        gate_scores = F.softmax(self.w_gate(x), dim=-1)
        loss_gate_scores = gate_scores
        # Determine the top-1 expert for each token
        capacity = int(self.capacity_factor * x.size(0))

        top_k_scores, top_k_indices = gate_scores.topk(self.top_k, dim=-1)  # 3/4

        mask = torch.zeros_like(gate_scores).scatter_(
            1, top_k_indices, 1
        )

        # Combine gating scores with the mask
        masked_gate_scores = gate_scores * mask

        # Denominators
        denominators = (
                masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        )

        # Norm gate scores to sum to the capacity
        gate_scores = (masked_gate_scores / denominators) * capacity

        if use_aux_loss:
            # load = gate_scores.sum(0)  # Sum over all examples
            # importance = gate_scores.sum(1)  # Sum over all experts

            # # Aux loss is mean suqared difference between load and importance
            # loss = ((load - importance) ** 2).mean()
            loss = load_balanced_loss(loss_gate_scores, mask)

            return gate_scores, loss

        return gate_scores, None


class FeedForward(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mu = nn.Linear(input_dim, output_dim)
        self.logvar = nn.Linear(input_dim, output_dim)

    def reparameterise(self, mu, std):
        eps = torch.randn_like(std)
        return mu + std * eps

    def KL_loss(self, mu, logvar):
        return (-(1 + logvar - mu.pow(2) - logvar.exp()) / 2).mean(dim=-1).mean()

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.reparameterise(mu, torch.exp(0.5 * logvar))
        kl_loss = self.KL_loss(mu, logvar)
        return z, kl_loss, torch.exp(logvar)


class MoE(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            output_dim: int,
            num_experts: int,
            top_k: int,
            capacity_factor: float = 1.0,
            mult: int = 4,
            use_aux_loss: bool = True,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.mult = mult
        self.top_k = top_k
        self.use_aux_loss = use_aux_loss

        self.experts = nn.ModuleList(
            [
                FeedForward(dim, dim)
                for _ in range(num_experts)
            ]
        )
        self.gate = Gate(
            dim,
            num_experts,
            top_k,
            capacity_factor,

        )

    def forward(self, x: Tensor):
        gate_scores, loss = self.gate(
            x, use_aux_loss=self.use_aux_loss
        )
        expert_outputs = []
        loss_kl = []
        Uncertainty = []
        for expert_output, kl_loss, sigma in [expert(x) for expert in self.experts]:
            expert_outputs.append(expert_output)
            loss_kl.append(kl_loss)
            Uncertainty.append(sigma)
        loss_KL = 0
        for i in range(self.num_experts):
            loss_KL += loss_kl[i]

        loss = loss + (loss_KL) / self.num_experts
        Uncertainty = torch.stack(Uncertainty, dim=0)

        if torch.isnan(gate_scores).any():
            print("NaN in gate scores")
            gate_scores[torch.isnan(gate_scores)] = 0

        inv_var = 1.0 / (Uncertainty + 1e-6)
        inv_var = inv_var.permute(1, 2, 3, 0)
        stacked_expert_outputs = torch.stack(expert_outputs, dim=-1)

        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[torch.isnan(stacked_expert_outputs)] = 0

        gate_scores = gate_scores.unsqueeze(2).expand(-1, -1, inv_var.size(2),
                                                      -1)
        weights = gate_scores * inv_var
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
        moe_output = torch.sum(gate_scores * weights * stacked_expert_outputs, dim=-1)

        return moe_output, loss


class MoE_block(nn.Module):
    def __init__(
            self,
            dim: int,
            heads: int,
            dim_head: int,
            mult: int = 4,
            dropout: float = 0.1,
            num_experts: int = 4,
            top_k=3,
            *args,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mult = mult
        self.top_k = top_k
        self.dropout = dropout
        self.moe = MoE(
            dim, dim * mult, dim, num_experts, top_k,*args, **kwargs
        )

        self.add_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor):
        out, loss = self.moe(x)

        return self.add_norm(x + out), loss
