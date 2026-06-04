# neGAT: Neuroevolution of Graph-Attention Networks

This simulation combines Graph Attention Networks (GAT) with structural genetic mutations (NEAT-style). Instead of optimizing fixed networks, agents evolve their own internal topologies by dynamically adding hidden nodes and routing recurrent edges.

## How It Works

The framework separates the processing mechanics from the information routing:

* **Shared Layer (`W_msg`):** A fixed, deep non-linear MLP. It is identical across all agents and never mutates, acting as a permanent, universal communication protocol.
* **Evolved Routing ($q$ and Topologies):** Individual agent genomes control their attention queries ($q$) and explicit graph wiring ($|V|$ nodes, $|E|$ edges).

Because `W_msg` is universally shared, agents can eventually transmit vectors to one another without mutations turning their messages into gibberish. They adapt entirely by changing **where** data flows—organically building internal memory loops, delay lines, and logic filters out of a fixed set of operations.

## Roadmap

* **Inter-Agent Messaging:** Dynamic k-NN spatial edge generation for direct agent-to-agent vector broadcasting.
* **Topological Crossover:** Innovation tracking to safely blend distinct graph structures.