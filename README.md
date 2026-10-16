# neGAT: Neuroevolution of Graph-Attention Networks

neGAT is an experimental population simulation in which agents use graph-attention networks to control their movement and evolve behavior over time. Each agent has a small genome made up of attention queries. Agents that lose all of their energy are removed and replaced by descendants of surviving agents, allowing useful behaviors to spread through the population.

The project explores a different approach to neuroevolution: instead of evolving every neural-network weight independently, it keeps the message-processing machinery shared and evolves how each agent selects and interprets information.

## Core Idea

The model separates **message transformation** from **information routing**:

* **Shared message network:** Every agent uses the same multi-head nonlinear transformation to turn a pair of connected node states into a message. The shared network provides a common communication language across the population.
* **Agent-specific queries:** Each agent has attention and sign queries that determine which incoming messages matter and whether they should contribute positively or negatively.
* **Graph-based state:** Environment inputs, agent inputs, and agent outputs are represented as nodes. Edges connect these nodes so that information can flow from the environment into agents, through each agent's controller, and back through recurrent output nodes.

Since the message network is shared, evolution primarily changes **where information flows and how it is weighted**, rather than producing an entirely unrelated neural network for every individual. This can allow agents to form selective communication patterns, feedback loops, and simple memory-like behaviors.

## Simulation Loop

At each simulation step:

1. The environment provides a changing periodic signal.
2. The graph-attention layer reads the current node states and computes messages for each destination node.
3. Messages are combined across attention heads and used to update each agent's output nodes.
4. The physics system applies the agents' movement decisions and keeps positions within the simulation area.
5. Agents receive or lose energy based on their position relative to the target region, their alignment with the moving environmental signal, and local crowding.
6. Agents whose energy reaches zero are removed. Their input and output state is reset, and their query parameters are initialized from surviving agents with variation.

The resulting population is therefore shaped by both the controller's behavior and the environment's ecological pressures.

## Graph Structure

The current graph contains three types of connections:

* Environment nodes send the global signal to agent output nodes.
* Each agent's input nodes connect to its output nodes.
* Output nodes have recurrent self-connections, allowing state from a previous step to influence later decisions.

The graph is built for the configured population and node dimensions when the simulation starts. The project leaves room for future experiments with changing graph topologies, but the current evolutionary process focuses on agent-specific attention queries and inherited variation.

## Running the Example

The main entry point is `PhysicsEngine` in `negat.py`. A minimal run looks like this:

```python
from negat import PhysicsEngine

model = PhysicsEngine(num_agents=256)
model.train(50_000)
model.plot_graph()
```

`train()` advances the simulation for the requested number of steps. `plot_graph()` displays the current agent positions and energy levels, the evolution of population energy over time, and the number of deaths at each step.

The default device uses CUDA when it is available and otherwise falls back to the CPU. The number of agents, attention heads, node dimensions, and environment/controller input and output sizes can be adjusted through the `PhysicsEngine` constructor.

## Project Layout

* `gat.py` — Multi-head graph-attention and shared message-processing layers.
* `negat.py` — Population management, graph construction, physics, energy, death, and repopulation logic.
* `example.ipynb` — Interactive training and visualization example.

## Planned: Self-Evolving Topology

A next step is to let the controller influence not only the agents' behavior, but also the structure of their internal graph. A secondary network could identify candidate neurons or nodes that should form new connections, allowing each agent's brain topology to evolve alongside its attention queries.

The existing connections already have attention-based scaling, so a full Hebbian learning rule may not be the best fit. Instead, future structural-plasticity experiments could combine attention with longer-term connection activity and usefulness:

* identify candidate neurons or nodes that would benefit from a new connection;
* strengthen connections that are consistently useful;
* weaken connections that remain inactive or provide little useful signal;
* remove edges that have not been updated or used for a long period; and
* remove neurons that no longer have outgoing edges or otherwise become disconnected from the computation.

The exact mechanism is intentionally undecided. The goal is for the control system to participate in topological evolution, so that agents can develop, simplify, and reorganize their own internal processing networks rather than only changing the parameters attached to a fixed graph.

## Status

neGAT is a research prototype intended for experimentation with graph-based neuroevolution. Its main purpose is to make it easy to test new environmental pressures, graph structures, controller signals, and evolutionary mechanisms rather than to provide a production-ready training framework.

