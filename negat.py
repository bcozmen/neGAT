import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

class GAT(nn.Module):
    def __init__(self, node_dim, num_heads, device):
        super().__init__()
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.activation = nn.Tanh()
        self.device = device
        
        self.message_transformation = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim * 4, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(node_dim * 4, node_dim * 2, bias=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(node_dim * 2, node_dim, bias=False)
        ).to(device)
        #nn.init.orthogonal_(self.message_transformation.weight)
        nn.init.orthogonal_(self.message_transformation[0].weight)
        nn.init.orthogonal_(self.message_transformation[2].weight)
        nn.init.orthogonal_(self.message_transformation[4].weight)
    def forward(self, V, E, q, V_agent):
        agent_id = E[:, 0]
        src = E[:, 1]
        dst = E[:, 2]

        v_in = V[src]   
        v_out = V[dst]  
        v = torch.cat([v_in, v_out], dim=-1) 

        q_agent = q[agent_id - 1]  

        q_attn = q_agent[:, :, :self.node_dim * 2]  
        q_sign = q_agent[:, :, self.node_dim * 2:]  

        v_expanded = v.unsqueeze(1) 

        attn_scores = (q_attn * v_expanded).sum(dim=-1)  
        attn_scores = attn_scores / torch.sqrt(torch.tensor(self.node_dim * 2.0, device=V.device))
        attn_scores = self.lrelu(attn_scores)  

        max_per_dst = torch.zeros((V.shape[0], self.num_heads), device=V.device)
        dst_expanded = dst.unsqueeze(1).expand(-1, self.num_heads)
        max_per_dst.scatter_reduce_(0, dst_expanded, attn_scores, reduce='amax', include_self=False)
        
        exp_attn = torch.exp(attn_scores - max_per_dst[dst])  
        sum_exp = torch.zeros((V.shape[0], self.num_heads), device=V.device)
        sum_exp.scatter_add_(0, dst_expanded, exp_attn)

        alpha = exp_attn / (sum_exp[dst] + 1e-8)  

        sign_scores = (q_sign * v_expanded).sum(dim=-1) 
        independent_sign = torch.tanh(sign_scores)      

        transformed_v = self.message_transformation(v) 
        
        transformed_v = transformed_v.unsqueeze(1)     
        alpha_expanded = alpha.unsqueeze(-1)           
        sign_expanded = independent_sign.unsqueeze(-1)  

        messages = alpha_expanded * transformed_v * sign_expanded  
        
        aggregated = torch.zeros((V.shape[0], self.num_heads, self.node_dim), device=V.device)
        dst_meta_expanded = dst.view(-1, 1, 1).expand(-1, self.num_heads, self.node_dim)
        aggregated.scatter_add_(0, dst_meta_expanded, messages)

        aggregated = aggregated.mean(dim=1)  

        output_mask = (V_agent[:, 1] == 1)  
        aggregated[output_mask] = self.activation(aggregated[output_mask])
        
        return aggregated

class PhysicsEngine(nn.Module):
    def __init__(self, num_agents=1000, num_heads=4, node_dim=8, 
                 n_env_inputs=1, n_agent_inputs=1, n_agent_outputs=1, device='cuda'):
        super().__init__()
        self.num_agents = num_agents
        self.num_heads = num_heads
        self.node_dim = node_dim
        self.n_env_inputs = n_env_inputs
        self.n_agent_inputs = n_agent_inputs
        self.n_agent_outputs = n_agent_outputs
        self.device = torch.device(device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu')

        self.gat_layer = GAT(node_dim=self.node_dim, num_heads=self.num_heads, device=self.device)
        self.generate_graph()
        self.step = 0

        self.phase_steps = 200

    def get_environment_inputs(self):
        env_inputs = torch.zeros((self.n_env_inputs, self.node_dim), device=self.device)
        phase_signal = np.sin(2 * np.pi * self.step / self.phase_steps)
        env_inputs[:, 0] = phase_signal  
        return env_inputs

    def forward(self):
        # 1. Clone the master feature matrix
        V = self.V.clone()
        
        # 2. Inject global environment inputs (Phase/Season signal)
        env_inputs = self.get_environment_inputs()
        self.step += 1
        V[:self.n_env_inputs] = env_inputs

        # 3. THE FIX: Explicitly ensure GAT sees the updated physics features
        # (Positions X, Y and Energy are stored in the first 3 indices of input nodes)
        # This gives the GAT the "sensory data" it needs to navigate.
        inp_mask = (self.V_agent[:, 1] == 0) & (self.V_agent[:, 0] > 0)
        # V[inp_mask] already contains the retained positions from self.V, 
        # but we must make sure we don't accidentally wipe them out.

        # 4. Pass the updated state through the Graph Neural Network
        final_message = self.gat_layer(V, self.E, self.q, self.V_agent)

        # 5. Update agent output nodes with the network's directional intentions
        output_mask = (self.V_agent[:, 1] == 1)
        V[output_mask] += final_message[output_mask]

        # 6. Apply physics engine rules (updates positions & alters energy)
        V = self._apply_physics(V)
        
        # 7. Apply a slight dampening/decay to the output nodes to prevent exploding gradients
        V[output_mask] *= 0.8  
        
        # 8. Commit the changes back to the master state
        self.V = V.detach()
        
        # 9. Handle ecology loops
        deaths = self.delete()      
        self.repopulate()  
        return deaths

    def _apply_physics(self, V):
        inp_mask = (self.V_agent[:, 1] == 0) & (self.V_agent[:, 0] > 0)
        inp_indices = torch.where(inp_mask)[0]
        inp_agent_ids = self.V_agent[inp_indices, 0] - 1
        
        global_out_indices = inp_agent_ids * self.n_agent_outputs + self.n_env_inputs + (self.num_agents * self.n_agent_inputs)
        
        # Physics position transitions
        V[inp_indices, 0] = V[inp_indices, 0] + V[global_out_indices, 0] * 0.1
        V[inp_indices, 1] = V[inp_indices, 1] + V[global_out_indices, 1] * 0.1  

        V[inp_indices, 0] = torch.clamp(V[inp_indices, 0], 0.0, 1.0)
        V[inp_indices, 1] = torch.clamp(V[inp_indices, 1], 0.0, 1.0)

        x_coord, y_coord = V[inp_indices, 0], V[inp_indices, 1]

        target_center = (0.5, 0.5)
        ring_inner_radius = 0.30
        ring_outer_radius = 0.45
        ring_distance = torch.sqrt((x_coord - target_center[0])**2 + (y_coord - target_center[1])**2)
        x_mask = (ring_distance >= ring_inner_radius) & (ring_distance <= ring_outer_radius)
        not_in_ring = 1 - x_mask.float()
        # 2. Dynamic Smooth Target Y Transition
        phase_signal = np.sin(2 * np.pi * self.step / self.phase_steps)
        
        # Smoothly shifts target Y midline between 0.25 (winter/bottom) and 0.75 (summer/top)
        target_y = 0.5 + (phase_signal * 0.25)
        
        # Calculate individual distance error to the moving target latitude line
        y_distance_error = torch.abs(y_coord - target_y)
        
        # Background base metabolic loss outside everything
        energy_change = torch.full_like(V[inp_indices, 2], -0.1)
        
        # High contrast reward inside the ring scaled smoothly by target Y distance proximity
        # Base ring buffer value = 0.26. Maximum penalty subtraction for missing altitude = -0.20
        ring_reward = 0.2 - (y_distance_error * 2.0)
        ring_reward = torch.clamp(ring_reward, -0.20, 0.20)
        
        # Combine both ecological constraints
        energy_change = ring_reward - not_in_ring * 0.15  # Bonus for being in the correct x zone, even if y is off
        V[inp_indices, 2] += energy_change

        V[inp_indices, 2] = torch.clamp(V[inp_indices, 2], -2, 1.0)
        
        return V

    def delete(self):
        inp_mask = (self.V_agent[:, 1] == 0) & (self.V_agent[:, 0] > 0)
        inp_indices = torch.where(inp_mask)[0]
        energies = self.V[inp_indices, 2]
        
        dead_agents_mask = energies <= 0.0
        self.dead_agents_indices = torch.where(dead_agents_mask)[0]
        
        if len(self.dead_agents_indices) > 0:
            dead_agent_ids = self.dead_agents_indices + 1
            is_dead_node = torch.isin(self.V_agent[:, 0], dead_agent_ids)
            self.V[is_dead_node] = 0.0

        return len(self.dead_agents_indices)

    def repopulate(self):
        if not hasattr(self, 'dead_agents_indices') or len(self.dead_agents_indices) == 0:
            return  
            
        num_to_repopulate = len(self.dead_agents_indices)
        
        inp_mask = (self.V_agent[:, 1] == 0) & (self.V_agent[:, 0] > 0)
        energies = self.V[inp_mask][:, 2]
        living_agents_indices = torch.where(energies > 0.0)[0] 

        global_inp_indices = self.dead_agents_indices * self.n_agent_inputs + self.n_env_inputs
        
        new_inputs = torch.zeros((num_to_repopulate, self.node_dim), device=self.device)
        new_inputs[:, :2] = torch.rand((num_to_repopulate, 2), device=self.device)  
        new_inputs[:, 2] = 1.0
        
        self.V[global_inp_indices] = new_inputs

        global_out_indices = self.dead_agents_indices * self.n_agent_outputs + self.n_env_inputs + (self.num_agents * self.n_agent_inputs)
        self.V[global_out_indices] = torch.zeros((num_to_repopulate, self.node_dim), device=self.device)

        if len(living_agents_indices) > 0:
            parent_samples = torch.randint(0, len(living_agents_indices), (num_to_repopulate,), device=self.device)
            living_agent_ids = self.V_agent[inp_mask, 0] - 1
            parent_ids = living_agent_ids[parent_samples]
            
            mutation_rate = 0.20 if (num_to_repopulate > self.num_agents * 0.5) else 0.04
            mutation = torch.randn(num_to_repopulate, self.num_heads, self.node_dim * 4, device=self.device) * mutation_rate
            
            self.q[self.dead_agents_indices] = self.q[parent_ids] + mutation
        else:
            self.q[self.dead_agents_indices] = torch.randn(num_to_repopulate, self.num_heads, self.node_dim * 4, device=self.device) * 2.0
            
        self.dead_agents_indices = torch.empty(0, dtype=torch.long, device=self.device)

    def generate_graph(self):
        self.V, self.V_agent = self._generate_V()
        self.E = self._generate_E()
        self.q = torch.randn(self.num_agents, self.num_heads, self.node_dim * 4, device=self.device)

    def _generate_E(self):
        env_nodes = torch.arange(self.n_env_inputs, device=self.device)
        out_nodes_global = torch.arange(self.num_agents * self.n_agent_outputs, device=self.device) + self.n_env_inputs + self.num_agents * self.n_agent_inputs
        
        src1 = env_nodes.repeat_interleave(self.num_agents * self.n_agent_outputs)
        dst1 = out_nodes_global.repeat(self.n_env_inputs)
        agent_id1 = torch.arange(1, self.num_agents + 1, device=self.device).repeat_interleave(self.n_agent_outputs).repeat(self.n_env_inputs)
        E_env_to_agent = torch.stack([agent_id1, src1, dst1], dim=1)
        
        local_in = torch.arange(self.n_agent_inputs, device=self.device)
        local_out = torch.arange(self.n_agent_outputs, device=self.device)
        src2_local = local_in.repeat_interleave(self.n_agent_outputs)
        dst2_local = local_out.repeat(self.n_agent_inputs)
        
        agent_offsets_in = torch.arange(self.num_agents, device=self.device) * self.n_agent_inputs + self.n_env_inputs
        agent_offsets_out = torch.arange(self.num_agents, device=self.device) * self.n_agent_outputs + self.n_env_inputs + (self.num_agents * self.n_agent_inputs)
        
        edges_per_agent = self.n_agent_inputs * self.n_agent_outputs
        src2 = src2_local.repeat(self.num_agents) + agent_offsets_in.repeat_interleave(edges_per_agent)
        dst2 = dst2_local.repeat(self.num_agents) + agent_offsets_out.repeat_interleave(edges_per_agent)
        agent_id2 = torch.arange(1, self.num_agents + 1, device=self.device).repeat_interleave(edges_per_agent)
        E_agent_to_agent = torch.stack([agent_id2, src2, dst2], dim=1) 

        src3 = out_nodes_global
        dst3 = out_nodes_global
        agent_id3 = torch.arange(1, self.num_agents + 1, device=self.device).repeat_interleave(self.n_agent_outputs)
        E_self_recurrent = torch.stack([agent_id3, src3, dst3], dim=1)
        
        return torch.cat([E_env_to_agent, E_agent_to_agent, E_self_recurrent], dim=0)

    def _generate_V(self):
        inp_env_nodes = torch.zeros((self.n_env_inputs, self.node_dim), device=self.device)
        inp_agent_nodes = torch.zeros((self.num_agents * self.n_agent_inputs, self.node_dim), device=self.device)
        inp_agent_nodes[:, :2] = torch.rand_like(inp_agent_nodes[:, :2])  
        inp_agent_nodes[:, 2] = 1.0

        out_agent_nodes = torch.zeros((self.num_agents * self.n_agent_outputs, self.node_dim), device=self.device)
        V = torch.cat([inp_env_nodes, inp_agent_nodes, out_agent_nodes], dim=0)
        
        V_agent_env = torch.zeros((self.n_env_inputs,), dtype=torch.long, device=self.device)  
        V_agent_inp = torch.arange(1, self.num_agents + 1, device=self.device).repeat_interleave(self.n_agent_inputs)  
        V_agent_out = torch.arange(1, self.num_agents + 1, device=self.device).repeat_interleave(self.n_agent_outputs)  
        V_agent = torch.cat([V_agent_env, V_agent_inp, V_agent_out], dim=0)

        V_type_inp = torch.zeros((self.n_env_inputs + self.num_agents * self.n_agent_inputs,), dtype=torch.long, device=self.device)  
        V_type_out = torch.ones((self.num_agents * self.n_agent_outputs,), dtype=torch.long, device=self.device)  
        V_type = torch.cat([V_type_inp, V_type_out], dim=0)

        return V, torch.stack([V_agent, V_type], dim=1)

# --- Simulation Execution ---
model = PhysicsEngine(num_agents=1000)
percentiles = torch.tensor([0, 25, 50, 75, 100], device=model.device)
avg_energy_percentiles = []
deaths_over_time = []

# Extended runtime to allow generational convergence
for t in tqdm(range(600)):
    deaths = model()
    deaths_over_time.append(deaths)

    inp_mask = (model.V_agent[:, 1] == 0) & (model.V_agent[:, 0] > 0)
    energies = model.V[inp_mask][:, 2].cpu().numpy()
    
    avg_percentiles = np.percentile(energies, percentiles.cpu().numpy())
    avg_energy_percentiles.append(avg_percentiles)

# Plotting Results
va = model.V_agent
inp_mask = (va[:, 1] == 0) & (va[:, 0] > 0)

x = model.V[inp_mask][:, 0].cpu()
y = model.V[inp_mask][:, 1].cpu()
energy = model.V[inp_mask][:, 2].cpu()

theta = np.linspace(0, 2*np.pi, 100)
plt.plot(0.5 + 0.30 * np.cos(theta), 0.5 + 0.30 * np.sin(theta), 'r--', alpha=0.5, label='Ring Boundaries')
plt.plot(0.5 + 0.45 * np.cos(theta), 0.5 + 0.45 * np.sin(theta), 'r--', alpha=0.5)

current_phase = np.sin(2 * np.pi * model.step / model.phase_steps)
target_y_line = 0.5 + (current_phase * 0.25)
plt.axhline(y=target_y_line, color='blue', linestyle=':', label='Target Latitude Line')

plt.scatter(x, y, c=energy, cmap='viridis', s=10)
plt.colorbar(label='Energy Level')
plt.title(f'Final Positions at Step {model.step} (Smooth Phase Val: {current_phase:.3f})')
ll = plt.legend(loc='lower left')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

avg_energy = np.array(avg_energy_percentiles)
plt.plot(avg_energy[:, 2], label='Median Energy')
plt.fill_between(range(len(avg_energy)), avg_energy[:, 0], avg_energy[:, 4], color='blue', alpha=0.2, label='Min-Max Range')
plt.title('Energy Percentiles Over Migration Cycles')
plt.xlabel('Time Step')
plt.ylabel('Energy Level')
plt.legend()
plt.show()

plt.plot(deaths_over_time, label='Deaths per Step')
plt.title('Agent Deaths Over Migration Cycles')
plt.xlabel('Time Step')
plt.ylabel('Number of Deaths')
plt.legend()
plt.show()