# CONVERGENT

## Multi-Agent Coherence Through Ambient Intent Awareness

*A Technical Whitepaper*

**James C. Young**
AI Enablement & Workflow Analyst
[github.com/AreteDriver/convergent](https://github.com/AreteDriver/convergent)

*February 2026*

---

## Abstract

Current multi-agent AI frameworks coordinate agents through centralized supervision or explicit message passing. Both approaches create scaling bottlenecks: centralized supervisors become throughput constraints as agent count grows, and message-passing protocols generate quadratic communication overhead. This paper introduces Convergent, a coordination library that replaces explicit coordination with ambient intent awareness. Agents publish decisions to a shared intent graph, observe relevant decisions by other agents, and independently adjust their behavior to maintain coherence. The mechanism is inspired by biological stigmergy (indirect coordination through environmental modification) and flocking behaviors (coherent group movement from local rules). Convergent is implemented as a Rust core with Python bindings, designed as a composable primitive that integrates with existing orchestration frameworks.

---

## 1. The Parallel Coordination Problem

Multi-agent AI systems face a fundamental coordination challenge: how do multiple agents working on related components produce coherent output without serializing their work? The problem is particularly acute in code generation, where architectural decisions by one agent (data model choices, API shapes, dependency selections) directly constrain the viable options for other agents.

Three dominant approaches exist in current frameworks, each with structural limitations:

| Approach | Mechanism | Scaling Behavior | Failure Mode |
|----------|-----------|------------------|--------------|
| Centralized Supervisor | Single agent routes all tasks and resolves conflicts | Linear bottleneck. Supervisor becomes throughput ceiling. | Supervisor context window overflow at scale |
| Message Passing | Agents negotiate directly via structured messages | O(n^2) communication. Token cost grows quadratically. | Livelock from circular negotiations |
| Sequential Pipeline | Agents work in strict order, each consuming prior output | Linear time regardless of task decomposability | No parallelism. Early errors cascade. |

All three approaches treat coordination as a communication problem: agents must talk to each other (or to a coordinator) to align. Convergent reframes coordination as a perception problem: agents must be able to see the landscape of decisions and respond to it.

---

## 2. Theoretical Foundations

### 2.1 Stigmergy

Stigmergy, a term coined by Pierre-Paul Grasse in 1959, describes a mechanism of indirect coordination where agents communicate through modifications to a shared environment rather than through direct interaction. The canonical example is ant trail pheromones: an ant deposits a chemical trail as it moves between a food source and the nest. Other ants detect the trail and follow it, depositing additional pheromones and reinforcing the path. No ant communicates with any other ant. Coordination emerges from the shared environment.

The critical insight is that stigmergic coordination scales naturally. Adding more ants to the system does not require more communication channels. Each ant follows the same simple rules, and the environmental signals (pheromone concentration) self-organize into efficient collective behavior.

### 2.2 Flocking and Boid Behaviors

Craig Reynolds' 1987 boid model demonstrated that realistic flocking behavior emerges from three local rules applied by each individual: separation (avoid crowding nearby neighbors), alignment (steer toward the average heading of nearby neighbors), and cohesion (steer toward the average position of nearby neighbors). No bird has a global view of the flock. No coordinator assigns positions. Each bird follows local rules relative to its neighbors, and coherent global behavior emerges.

Convergent adapts these principles to the domain of multi-agent AI coordination:

| Boid Rule | Biological Behavior | Software Analog |
|-----------|-------------------|-----------------|
| Separation | Avoid crowding nearby neighbors | Avoid duplicating another agent's work. If an intent exists for a component, don't build a competing version. |
| Alignment | Steer toward average heading of neighbors | Adopt compatible interfaces. If neighboring agents converge on a data model, align with it. |
| Cohesion | Steer toward average position of neighbors | Converge on shared abstractions. Move toward the emerging architectural consensus. |

### 2.3 From Biology to Software Architecture

The translation from biological stigmergy to software agent coordination requires replacing biological mechanisms with computational equivalents:

| Biological Mechanism | Computational Equivalent | Convergent Component |
|---------------------|-------------------------|---------------------|
| Pheromone trails | Entries in a shared data structure | Intent Graph nodes |
| Pheromone concentration | Confidence metric on shared entries | Stability Score (0.0-1.0) |
| Pheromone evaporation | Time-based decay of unconfirmed entries | Stability decay for unvalidated intents |
| Ant sensing radius | Relevance query over shared state | Intent Resolver semantic matching |
| Trail reinforcement | Dependent adoption increases confidence | Stability increase when consumed by others |

---

## 3. Core Architecture

Convergent consists of three components that together form a minimal but complete coordination primitive: the Intent Graph (shared state), the Stability Scorer (confidence mechanism), and the Intent Resolver (perception mechanism).

### 3.1 The Intent Graph

The intent graph is a directed graph where nodes represent agent decisions and edges represent relationships between decisions (dependencies, conflicts, refinements). It serves as the shared environment that agents read from and write to -- the computational equivalent of the pheromone field.

Each intent node contains structured information about what an agent is doing, what it provides to other agents, what it requires from other agents, and how confident it is in the decision:

| Field | Type | Purpose |
|-------|------|---------|
| action | string | Specific, verifiable description of what the agent is doing |
| category | enum | decision, interface, dependency, constraint |
| provides | list[string] | What this intent makes available to other agents |
| requires | list[string] | What this intent needs from other agents |
| constraints | list[string] | Rules this intent imposes on the system |
| stability | float (0.0-1.0) | Confidence score reflecting decision maturity |
| evidence | list[string] | What supports the claimed stability score |
| files_affected | list[string] | Which files this intent touches (separation enforcement) |

Agents read the graph frequently (before each major decision point) and write to it less frequently (when they commit a significant decision). This read-heavy, write-light pattern aligns with the stigmergic model where environmental reading is continuous but environmental modification is intermittent.

### 3.2 The Stability Scorer

Stability is the mechanism by which the intent graph distinguishes between tentative explorations and committed decisions. It serves the same function as pheromone concentration in ant trails: higher stability signals attract alignment from other agents.

Stability scores range from 0.0 to 1.0 and are calculated from multiple evidence sources:

| Evidence Type | Stability Range | Rationale |
|--------------|----------------|-----------|
| Speculative intent | 0.1 - 0.3 | Agent has expressed an intention but produced no artifacts. Others should be aware but not dependent. |
| Code committed | 0.5 - 0.7 | Agent has produced code implementing the intent. Reversal is costly but possible. |
| Tests passing | 0.7 - 0.85 | Implementation is validated. Other agents can safely depend on the interface. |
| Consumed by others | 0.85 - 1.0 | Other agents have adopted this decision. Reversal would cascade. Effectively locked. |

Stability also decays over time if not reinforced, analogous to pheromone evaporation. An intent published early in execution that is never confirmed by code or tests gradually loses its influence on other agents' decisions. This prevents stale, abandoned intents from constraining the system indefinitely.

### 3.3 The Intent Resolver

The intent resolver is the perception mechanism: the computational equivalent of an ant sensing pheromone trails. When an agent reaches a decision point, it queries the intent resolver to understand the current landscape of decisions relevant to its work.

Resolution operates in two phases. Phase 1 (structural matching) compares intent subjects using string similarity and schema structure overlap. This works without LLM dependencies and is sufficient for most coordination scenarios. Phase 2 (semantic matching, planned) introduces LLM-powered evaluation where an LLM determines whether two intents are semantically related even when they use different terminology.

The resolver returns a ranked list of relevant intents with their stability scores, enabling the agent to make informed decisions: adopt high-stability interfaces, avoid duplicating active work, and respect published constraints.

---

## 4. Worked Convergence Example

Consider three agents building a web application: Agent A (authentication), Agent B (user management), and Agent C (API layer). Without coordination, they independently choose incompatible data models, duplicate shared utilities, and define conflicting API contracts.

With Convergent, the workflow proceeds as follows:

**Step 1: Intent Publication.** Agent A publishes an intent: "Creating User model with email, name, role fields. Provides: User model, AuthService.authenticate() method. Stability: 0.2 (speculative)." Agent B, working on user management, reaches a decision point about its own User model.

**Step 2: Intent Resolution.** Agent B queries the intent resolver before designing its data model. The resolver returns Agent A's intent as highly relevant (subject overlap: "User model"). Agent B sees that Agent A is already defining a User model with specific fields. Rather than creating a competing definition, Agent B aligns: it adopts Agent A's schema and adds its own fields (preferences, last_login) as extensions.

**Step 3: Stability Reinforcement.** Agent B publishes its own intent referencing Agent A's User model in its "requires" field. This dependency causes Agent A's stability score to increase (now consumed by another agent, moving from 0.2 to 0.6). Agent C, building the API layer, queries the resolver and sees two aligned intents with moderate-to-high stability. It adopts both interfaces confidently.

**Step 4: Convergence.** By the time all three agents complete their work, they have independently converged on a coherent architecture -- not because a supervisor told them to, but because each agent perceived the landscape of decisions and responded to it. The intent graph records the full decision history, providing an auditable trace of how coherence emerged.

This is exactly how flocking works in nature. Birds don't send messages to coordinate formation. They each follow simple rules relative to their neighbors, and coherent behavior emerges.

---

## 5. Conflict Resolution

When two agents publish conflicting intents (incompatible data models, overlapping file changes, contradictory constraints), the intent resolver detects the conflict through structural or semantic overlap analysis. Resolution follows a stability-weighted protocol:

If one intent has significantly higher stability (difference > 0.3), the lower-stability intent yields. The agent with the lower-stability intent is notified of the conflict and expected to realign. If stability scores are comparable, the older intent (first-mover) takes precedence. The newer agent adapts to the established decision. In rare cases where neither rule resolves the conflict, the system escalates to human review or, if integrated with an orchestrator like Gorgon, to a supervisor agent.

This approach avoids the livelock problem that occurs in peer-to-peer negotiation, where two agents continuously adjust in response to each other without reaching agreement. The stability score provides an asymmetry that breaks the cycle: one decision is always more established than the other.

---

## 6. Implementation Details

### 6.1 Rust Core

The intent graph, stability scorer, and graph query engine are implemented in Rust for performance. Graph operations (node insertion, edge creation, stability calculation, neighborhood queries) must be fast enough that agents can resolve against the graph before every decision without introducing meaningful latency. The Rust implementation achieves sub-millisecond query times for graphs with up to ten thousand nodes, which exceeds the expected scale of most multi-agent workflows.

The Rust core is compiled as a Python extension module via PyO3 and Maturin, exposing a clean Python API while retaining native performance.

### 6.2 Python Semantic Layer

The semantic resolution layer is implemented in Python for flexibility and LLM integration. Phase 1 uses structural matching: intent subjects are compared using string similarity and schema structure overlap. Phase 2 will introduce LLM-powered semantic matching, where an LLM evaluates whether two intents are semantically related even when they use different terminology.

This phased approach ensures the core system works without LLM dependencies (important for testing, offline use, and cost control) while preserving a clear path to more sophisticated matching.

### 6.3 Project Status

| Component | Status | Language |
|-----------|--------|----------|
| Intent Graph (core) | Implemented | Rust |
| Stability Scorer | Implemented | Rust |
| Intent Resolver (structural) | Implemented | Python + Rust |
| Intent Resolver (semantic/LLM) | Planned (Phase 2) | Python |
| PyO3 Bindings | Implemented | Rust -> Python |
| Gorgon Integration | Implemented (Phase 3) | Python |
| Benchmarks & Evaluation | Planned | Python |

---

## 7. Differentiation

Convergent occupies a distinct position in the multi-agent AI landscape. It is not an orchestration framework (it does not manage workflows), nor a communication protocol (agents do not send messages to each other). It is a coordination primitive: a shared data structure with well-defined read/write semantics that enables coherent parallel work.

| Dimension | Convergent | LangGraph | CrewAI | AutoGen |
|-----------|-----------|-----------|--------|---------|
| Coordination Model | Emergent (stigmergy) | Graph-based state machine | Role-based delegation | Conversational negotiation |
| Agent Communication | Indirect (shared state) | State transitions | Task handoffs | Direct messages |
| Scaling Behavior | O(n) -- reads scale linearly | Depends on graph topology | Linear (supervisor bottleneck) | O(n^2) message overhead |
| Parallelism | Native -- agents work independently | Possible with parallel branches | Sequential by default | Possible but complex |
| Token Overhead | Low -- read-only queries | Moderate -- state serialization | Moderate -- role prompts | High -- full conversation history |
| Implementation | Rust core + Python | Python | Python | Python |

Convergent's key advantage is composability. It does not compete with orchestration frameworks -- it enhances them. A Gorgon workflow can use Convergent for its coordination layer. A LangGraph application could integrate the intent graph as a shared state mechanism. The library is designed to be embedded, not to replace existing infrastructure.

---

## 8. Gorgon Integration

Convergent is designed as the coordination primitive for the Gorgon multi-agent orchestration framework. In this integration, Gorgon manages the workflow lifecycle (agent instantiation, budget management, checkpoint/resume, quality gates) while Convergent manages inter-agent coherence.

The integration points are:

- **Intent publication hooks:** Gorgon agents automatically publish intents at decision points defined in the YAML workflow. The orchestrator does not need to know the content of the intents -- it only provides the timing signals.
- **Resolution queries:** Before each major agent action, the Gorgon workflow engine triggers an intent resolution query, injecting relevant context from other agents into the active agent's prompt.
- **Conflict escalation:** When Convergent's stability-weighted resolution cannot resolve a conflict, it escalates to Gorgon's supervisor agent or triggers a human-in-the-loop review.
- **Audit trail:** The intent graph persists alongside Gorgon's SQLite checkpoint data, providing a complete record of how architectural decisions emerged and evolved.

This separation of concerns -- Gorgon for orchestration, Convergent for coordination -- follows the single responsibility principle and allows either component to be used independently or replaced without affecting the other.

---

## 9. Future Work

- **Semantic resolution (Phase 2):** LLM-powered intent matching that detects semantic relationships between intents using different terminology. An agent describing "authentication tokens" should match with an agent describing "session management" when the concepts overlap.
- **Benchmark suite:** Comparative evaluation of convergent coordination versus centralized supervision and message-passing across standardized multi-agent tasks, measuring coherence quality, token efficiency, wall-clock time, and scaling behavior.
- **Visualization dashboard:** Real-time visualization of the intent graph showing decision evolution, stability trajectories, conflict resolution events, and convergence patterns.
- **Cross-framework adapters:** Integration modules for LangGraph, CrewAI, and AutoGen that allow these frameworks to use Convergent's intent graph as a coordination layer without modifying their core architectures.

---

## 10. Conclusion

The multi-agent AI field has treated coordination as a communication problem: agents need to talk to each other to align. Convergent proposes that coordination is a perception problem: agents need to see the landscape of decisions and respond to it. This reframing, inspired by biological stigmergy and flocking behaviors, enables a coordination mechanism that scales naturally with agent count, avoids the token overhead of message passing, and eliminates the bottleneck of centralized supervision.

The intent graph, stability scorer, and intent resolver form a minimal but complete coordination primitive. Agents publish decisions, observe decisions, and adjust independently. Coherent output emerges from local rules applied to shared state -- exactly as it does in the biological systems that inspired the architecture.

Convergent does not claim to solve all multi-agent coordination challenges. It introduces a specific, well-defined primitive -- ambient intent awareness -- that complements rather than replaces existing approaches. Its value is most pronounced in scenarios where multiple agents must work in parallel on related components, which is precisely the scenario that current frameworks handle least effectively.

*Coordination is not a communication problem. It is a perception problem. Convergent provides the perception.*

---

## References

1. Grasse, P.-P. (1959). La reconstruction du nid et les coordinations interindividuelles chez Bellicositermes natalensis et Cubitermes sp. La theorie de la stigmergie. *Insectes Sociaux*, 6(1), 41-80.
2. Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model. *ACM SIGGRAPH Computer Graphics*, 21(4), 25-34.
3. Valckenaers, P., Van Brussel, H., Kollingbaum, M., & Bochmann, O. (2001). Multi-agent coordination and control using stigmergy applied to manufacturing control. *European Agent Systems Summer School*.
4. Dorigo, M., & Stutzle, T. (2004). *Ant Colony Optimization*. MIT Press.
5. Parunak, H. V. D. (2006). A survey of environments and mechanisms for human-human stigmergy. In D. Weyns et al. (Eds.), *Environments for Multi-Agent Systems II*. Springer.
6. Thangaraju, V. (2025). Cognitive RPA Swarms: Leveraging ML-Driven Collective Intelligence for Decentralized Process Automation. *Transactions on Engineering and Computing Sciences*, 13(01), 181-194.
7. Schmid, S. et al. (2023). MOSAIK: An Agent-Based Decentralized Control System With Stigmergy. *ESWC 2023*.
8. S-MADRL Framework (2025). Deep Reinforcement Learning for Multi-Agent Coordination using Stigmergic Communication. *Artificial Life and Robotics*.

---

**Repository:** [github.com/AreteDriver/convergent](https://github.com/AreteDriver/convergent)
**License:** MIT
