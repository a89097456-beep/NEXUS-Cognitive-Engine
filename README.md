NEXUS Cognitive Engine

An experimental model-based reinforcement learning engine written in C++17.

I started NEXUS with a question: can an agent learn enough about its environment to predict what might happen next — and use those predictions to make better decisions?

NEXUS is my attempt to explore that idea.

Rather than relying entirely on trial and error, the engine learns an internal model of the environment. It can use that model to simulate possible outcomes, evaluate different action sequences, and plan ahead.

How it works

NEXUS combines several learning and planning mechanisms:

- Learned World Model — an ensemble of neural networks trained to predict how the environment responds to actions.
- CEM / MPC Planning — searches through possible action sequences using the learned model to find promising decisions.
- Double Q-Learning — estimates the value of states and actions through a separate value-learning mechanism.
- Random Network Distillation (RND) — provides intrinsic rewards to encourage exploration of unfamiliar states.
- Vector Quantization & Symbolic Rule Induction — experimental mechanisms for converting continuous observations into discrete representations and discovering patterns.

Environments

I built several environments to experiment with the engine:

- NumberChain
- ModChain
- Switches
- MountainCar
- Pendulum

NEXUS also includes a bridge for working with Gymnasium environments.

What I'm exploring

The interesting part isn't just whether an agent can complete a task. I want to understand how its internal mechanisms interact.

Does a learned world model actually improve planning? Can intrinsic exploration help it discover better strategies? Can symbolic representations reveal useful structure in what the agent observes?

These are some of the questions I'm investigating with NEXUS.

Technical details

- Language: C++17
- Dependencies: C++ standard library
- Architecture: Model-based reinforcement learning
- Planning: Cross-Entropy Method (CEM) and Model Predictive Control (MPC)
- Learning: World-model ensemble, Double Q-Learning and RND
- Project format: Single-file experimental implementation

Current status

NEXUS is an ongoing experimental project. The current implementation is a research prototype, and there is still a lot to investigate, test and improve.

The source code is available in this repository. I'm continuing to develop the engine and explore how far this approach can go.
