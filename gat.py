import torch
import torch.nn as nn

class ParallelMultiHeadMLP(nn.Module):
    def __init__(self, num_heads, node_dim):
        super().__init__()
        self.num_heads = num_heads
        self.node_dim = node_dim
        
        # Define 3D weight tensors: [num_heads, in_features, out_features]
        # Layer 1: node_dim*2 -> node_dim*4
        self.w1 = nn.Parameter(torch.empty(num_heads, node_dim * 2, node_dim * 4))
        # Layer 2: node_dim*4 -> node_dim*2
        self.w2 = nn.Parameter(torch.empty(num_heads, node_dim * 4, node_dim * 2))
        # Layer 3: node_dim*2 -> node_dim
        self.w3 = nn.Parameter(torch.empty(num_heads, node_dim * 2, node_dim))
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialization happens once, so a loop here is perfectly fine
        for i in range(self.num_heads):
            nn.init.orthogonal_(self.w1[i])
            nn.init.orthogonal_(self.w2[i])
            nn.init.orthogonal_(self.w3[i])

    def forward(self, x):
        # x shape: [edges, node_dim * 2]
        
        # Layer 1: Multiply edges(e) features(i) by heads(h) features(i) to get out(o)
        # Result shape: [edges, num_heads, node_dim * 4]
        h1 = torch.einsum('ei,hio->eho', x, self.w1)
        h1 = self.lrelu(h1)
        
        # Layer 2
        # Result shape: [edges, num_heads, node_dim * 2]
        h2 = torch.einsum('ehi,hio->eho', h1, self.w2)
        h2 = self.lrelu(h2)
        
        # Layer 3
        # Result shape: [edges, num_heads, node_dim]
        out = torch.einsum('ehi,hio->eho', h2, self.w3)
        return out

class GAT(nn.Module):
    def __init__(self, node_dim, num_heads, device):
        super().__init__()
        self.node_dim = node_dim
        self.num_heads = num_heads
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.device = device

        # 1. Use the new parallelized MLP
        self.parallel_decoder = ParallelMultiHeadMLP(self.num_heads, self.node_dim).to(self.device)
        
        self.final_projection = nn.Sequential(
            nn.Linear(self.num_heads * self.node_dim, self.node_dim, bias=False),
            nn.LeakyReLU(negative_slope=0.1)
        ).to(self.device)
        
        nn.init.orthogonal_(self.final_projection[0].weight)

    def forward(self, V, E, q, V_agent):
        v, q_attn, q_sign, dst = self._get_params(V, E, q, V_agent) 

        alpha = self._calculate_attention(q_attn, v, dst)
        independent_sign = self._calculate_sign(q_sign, v)      
        
        # 2. Process all heads in parallel (No for loop!)
        # Output shape is directly [edges, num_heads, node_dim]
        transformed_v = self.parallel_decoder(v) 
        
        messages = alpha.unsqueeze(-1) * transformed_v * independent_sign.unsqueeze(-1)  
        
        aggregated = self._aggregate_messages(messages, dst)

        output_mask = (V_agent[:, 1] == 1) 
        aggregated[output_mask] = torch.tanh(aggregated[output_mask])
        return aggregated

    def _calculate_attention(self, q_attn, v, dst):
        v_expanded = v.unsqueeze(1) 
        attn_scores = (q_attn * v_expanded).sum(dim=-1)  
        attn_scores = attn_scores / torch.sqrt(torch.tensor(self.node_dim * 2.0, device=v.device))
        attn_scores = self.lrelu(attn_scores) 

        max_per_dst = torch.zeros((v.shape[0], self.num_heads), device=v.device)
        dst_expanded = dst.unsqueeze(1).expand(-1, self.num_heads)
        max_per_dst.scatter_reduce_(0, dst_expanded, attn_scores, reduce='amax', include_self=False)

        exp_attn = torch.exp(attn_scores - max_per_dst[dst])
        sum_exp = torch.zeros((v.shape[0], self.num_heads), device=v.device)
        sum_exp.scatter_add_(0, dst_expanded, exp_attn)

        return exp_attn / (sum_exp[dst] + 1e-8)

    def _calculate_sign(self, q_sign, v):
        v_expanded = v.unsqueeze(1) 
        sign_scores = (q_sign * v_expanded).sum(dim=-1) 
        return torch.tanh(sign_scores)

    def _aggregate_messages(self, messages, dst):
        # Gather messages targeting the same destination node
        aggregated = torch.zeros((dst.max() + 1, self.num_heads, self.node_dim), device=messages.device)
        dst_meta_expanded = dst.view(-1, 1, 1).expand(-1, self.num_heads, self.node_dim)
        aggregated.scatter_add_(0, dst_meta_expanded, messages)
        
        # 4. CONCATENATE instead of averaging
        # Flatten [nodes, num_heads, node_dim] into [nodes, num_heads * node_dim]
        concat_aggregated = aggregated.view(-1, self.num_heads * self.node_dim)
        
        # 5. Pass through the final shared projection network -> [nodes, node_dim]
        final_out = self.final_projection(concat_aggregated)
        
        return final_out

    def _get_params(self, V, E, q, V_agent):
        agent_id = E[:, 0]
        src = E[:, 1]
        dst = E[:, 2]

        v_in = V[src]   
        v_out = V[dst]  
        v = torch.cat([v_in, v_out], dim=-1) 

        q_agent = q[agent_id - 1]  
        q_attn = q_agent[:, :, :self.node_dim * 2]  
        q_sign = q_agent[:, :, self.node_dim * 2:]  

        return v, q_attn, q_sign, dst        

    def _create_mlp(self):
        # Helper function to generate identical architectures with distinct weights
        mlp = nn.Sequential(
            nn.Linear(self.node_dim * 2, self.node_dim * 4, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(self.node_dim * 4, self.node_dim * 2, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(self.node_dim * 2, self.node_dim, bias=False)
        ).to(self.device)
        
        for layer in mlp:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)

        return mlp