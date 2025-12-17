# Assignment 4: DDPG, TD Learning, and VLA Models for TurtleBot3 Navigation

**Student:** Andrew Boksz  
**Course:** DS685-851: AI For Robotics  
**Date:** December 17, 2025

* * *

## Overview

This assignment explores deterministic policy gradient methods (DDPG), temporal-difference (TD) learning, and their application to continuous control problems in robotics, particularly for TurtleBot3 navigation and vision-language-action (VLA) models.

* * *

## Question 1 (Deterministic policy gradient vs TD backup in continuous control)

### Part 1: Why Deterministic Policy Gradient Avoids Action Space Integration

The deterministic policy gradient theorem states that:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\mu} [\nabla_\theta\mu_\theta(s)\nabla_a Q^\mu(s,a)|_{a=\mu_\theta(s)}]
$$

This formula differs from stochastic policy gradients because the policy $\mu_\theta(s)$ is deterministic, meaning it produces a single action for each state, rather than a probability distribution over actions. The gradient computation takes advantage of the chain rule by determining how the action-value function changes with respect to actions ($\nabla_a Q^\mu(s,a)$), then propagating this gradient through the policy's mapping from states to actions ($\nabla_\theta\mu_\theta(s)$).

On the contrary, stochastic policy gradients require computing $\mathbb{E}_{a \sim \pi(\cdot|s)}[\cdot]$, which requires integration or sampling over the entire action distribution. For continuous action spaces like TurtleBot3's velocity commands (linear velocity $v \in \mathbb{R}$ and angular velocity $\omega \in \mathbb{R}$), this would require either Monte Carlo sampling across infinite dimensions or analytical integration. These are both computationally impractical for real-time control.

The deterministic formulation avoids the computational burden by evaluating the Q-function gradient at exactly one point, the action selected by the deterministic policy, which avoids any explicit integration over the action space.

### Part 2: Why TD(0) Cannot Produce Continuous Control Commands

The TD(0) state-value update is defined as:

$$
V(s) \leftarrow V(s) + \alpha(R_{t+1} + \gamma V(S_{t+1}) - V(s))
$$

This update rule learns the expected cumulative return from each state under the current policy, but it has no mechanism for action selection. The value function $V(s)$ explains how valuable a state is, but cannot define what action it should take.

TurtleBot3 navigation needs continuous control outputs: specific values for linear velocity $v$ and angular velocity $\omega$ at every timestep. A TD(0) value function cannot directly produce these commands because:

1.  **No action argument**: The function $V(s)$ has no dependence on actions, so there's no way to evaluate different action choices
2.  **Optimization requirement**: Extracting a policy from $V(s)$ alone would require solving $\mu(s) = \arg\max_a \mathbb{E}[r(s,a) + \gamma V(s')]$ at every decision point, which demands a model of environment dynamics
3.  **Continuous optimization**: In continuous action spaces, this optimization problem becomes a complex continuous optimization at every timestep, making it computationally intensive for real-time robot control

### Part 3: Deterministic Actor as Natural Choice for TurtleBot3 Navigation

The TurtleBot3 DRL navigation repository requires smooth, continuous velocity commands to have stable robot motion and prevent mechanical stress on actuators. A deterministic actor $\mu_\theta(s): \mathcal{S} \rightarrow \mathcal{A}$ is the natural choice for this control problem due to the following reasons:

**Direct control mapping**: The actor network directly maps sensor observations (LIDAR scans, goal relative position, previous velocities) to continuous action outputs $(v, \omega)$ without requiring additional optimization or planning steps. This enables real-time control at the frequencies required for safe navigation (typically 10-20 Hz).

**Smooth trajectory generation**: Deterministic policies deliver consistent actions for similar states, which is crucial for generating smooth, executable trajectories. Stochastic policies with high variance could produce erratic movements that damage hardware or violate kinematic constraints of the robot platform.

**Gradient-based refinement**: The Q-function gradient $\nabla_a Q^\mu(s,a)$ provides actionable signals for improving the policy. Near obstacles, negative gradients push the actor toward lower velocities or increased turning. Near goals with clear paths, positive gradients promote faster, more direct motion. This gradient flow allows the actor to learn nuanced control behaviors that balance safety and efficiency.

* * *

## Question 2 (TD critic target in DDPG vs TD(0) and stability)

### Part 1: Comparing TD Target Formulations

**DDPG TD Target:**

$$
y = r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s'))
$$

**TD(0) Value Prediction:**

$$
y_{\text{TD}(0)} = r + \gamma V(s')
$$

**Q-learning Target:**

$$
y_{\text{Q-learning}} = r + \gamma \max_{a'} Q(s', a')
$$

These three formulas differ in three key ways:

**(i) Dependence on current policy**: DDPG's target is explicitly policy-dependent. DDPG evaluates the Q-function at the action selected by the target policy $\mu_{\theta'}(s')$. This makes the target shift as the policy improves. TD(0) assumes a fixed evaluation policy and simply predicts values under that policy. Q-learning is off-policy and policy-independent, meaning that it estimates the optimal value function regardless of the behavior policy.

**(ii) Use of max operator**: Q-learning's $\max_{a'}$ operator selects the highest Q-value among all possible next actions. This introduces systematic overestimation bias because the max operation boosts estimation noise. Therefore, if several actions have similar true values but noisy estimates, the max will tend to select whichever estimate happens to be highest due to noise rather than true value. DDPG avoids this bias by evaluating Q at a single action chosen by the target policy, which doesn't inherently favor overestimation.

**(iii) Role of target networks**: DDPG employs slowly-updated target networks with parameters $\phi'$ and $\theta'$ that are updated through Polyak averaging:

$$
\phi' \leftarrow \tau\phi + (1-\tau)\phi', \quad \tau \ll 1
$$

This stabilization mechanism prevents the known "moving target" problem. This is where both the predictor (current Q-network) and the target (bootstrap value) change simultaneously, which can cause oscillations or divergence. The target networks remain relatively stable over many updates, providing a consistent regression target for the critic.

### Part 2: Stability Considerations for Sim-to-Real Transfer

**Overestimation bias from max operator**: The max operator in Q-learning systematically overestimates action values because it cannot distinguish between high true values and high estimation noise. That's especially dangerous for collision avoidance tasks because if the critic overestimates the safety of risky actions, the policy may become too aggressive, leading to collisions when deployed on real hardware where environmental uncertainty exceeds simulation.

**Target network stabilization**: By freezing target network parameters for multiple gradient updates (typically hundreds or thousands), the bootstrap target $y$ remains stable even as the main networks learn. This prevents oscillations that arise when the target shifts after every update. Niu et al. (2021) demonstrated that this stability is critical for sim-to-real transfer with TurtleBot3. Unstable critics cause policy fluctuations in simulation that completely fail when transferred to physical robots due to the reality gap.

**TD(0) stability vs. control capability**: Pure TD(0) value prediction is more stable than action-value methods since it doesn't involve policy optimization over actions and doesn't use a max operator. However, this stability compromises control capability. TD(0) by itself cannot specify what actions to take, requiring a separate actor or planning module. For robot navigation, we need both stability and direct control, motivating the actor-critic architecture.

### Part 3: Robotics Perspective on Critic Stability

When transferring policies from simulation to real TurtleBot3 hardware, critic instability is shown as a serious practical problem:

**Safety hazards**: Unstable value estimates cause weird/random policy updates during training. If the policy hasn't converged to stable behaviors in simulation, it may display sudden direction changes, excessive speeds near obstacles, or complete navigation failures when deployed on real hardware.

**Hardware damage**: Oscillating policies can generate high-frequency velocity commands that exceed the robot's actuator capabilities or create mechanical stress. Real motors have inertia, friction, and response delays that simulation doesn't perfectly capture. Unstable policies that work in idealized simulation can cause overheating, mechanical wear, or even permanent damage to real servos.

**Deployment failure**: The reality gap between simulation and physical deployment means that policies must be robust to modeling errors, sensor noise, and unmodeled dynamics. A critic that hasn't converged to stable, accurate value estimates will provide poor guidance for the actor, causing policies that perform well in simulation to fail right away on real robots.

Niu et al. (2021) emphasized that stable critic training allows for zero-shot sim-to-real transfer, where policies trained entirely in simulation can successfully navigate on physical TurtleBot3 robots without any additional fine-tuning. This can only work when the critic provides stable, reliable value estimates throughout training.

* * *

## Question 3 (Reward structure, exploration, and conservative behaviors)

### Part 1: Interaction Between Reward Structure and Deterministic Policy

In collision avoidance tasks for TurtleBot3, the reward function typically takes the form similar to:

$$
r(s,a) = \begin{cases} 
-100 & \text{if collision occurs} \\
-\lambda \cdot d_{\min}^{-1} & \text{if } d_{\min} < d_{\text{safe}} \text{ (near-collision penalty)} \\
+\alpha \cdot \Delta d_{\text{goal}} & \text{if progress toward goal} \\
+R_{\text{goal}} & \text{if goal reached}
\end{cases}
$$

where $d_{\min}$ represents the minimum distance to any obstacle and $\Delta d_{\text{goal}}$ represents progress toward the goal position.

The critic $Q^\mu(s,a)$ learns to assign values to state-action pairs based on these rewards. Near obstacles, actions that move closer receive low Q-values due to collision penalties. Actions that maintain safe distances while progressing toward the goal receive high Q-values. The gradient $\nabla_a Q^\mu(s,a)$ captures the local sensitivity of the Q-function with respect to action changes:

- **Near obstacles**: $\nabla_a Q^\mu(s,a)$ points away from collision-inducing actions (e.g., negative gradient on forward velocity, positive gradient on turning rate to steer away)
- **In open space**: $\nabla_a Q^\mu(s,a)$ points toward efficient goal-reaching actions (e.g., positive gradient on forward velocity in goal direction)

The actor update follows this gradient through the chain rule:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\mu} [\nabla_\theta\mu_\theta(s)\nabla_a Q^\mu(s,a)|_{a=\mu_\theta(s)}]
$$

The magnitude and direction of $\nabla_a Q^\mu(s,a)$ directly determines how the policy parameters $\theta$ are adjusted. If the gradient consistently points toward conservative actions due to the reward structure, the policy will converge to conservative behaviors.

### Part 2: Convergence to Conservative Behaviors with Weak Exploration

Deterministic policies can be stuck in conservative local optima when exploration is lacking, especially under the following conditions:

**Sparse positive rewards dominate**: If a robot only reaches goals during training sometimes, while collision penalties are frequent and large, then the critic mostly learns the conservative actions (so... minimal velocities, cautious turning) that are safe. The gradient $\nabla_a Q^\mu(s,a)$ becomes overpowered by obstacle-avoidance signals rather than goal-reaching signals.

**Weak exploration fails to discover better actions**: The exploration mechanism in DDPG adds noise to actions: $a = \mu_\theta(s) + \mathcal{N}(0, \sigma^2)$ or uses Ornstein-Uhlenbeck noise for temporal correlation. If $\sigma$ is too small or decays too quickly, the policy never explores the space of higher velocities or more aggressive maneuvers that might actually be optimal. Without experiencing these actions, the critic cannot learn their true values, and the actor gradient never points toward them.

**Gradient magnitude imbalance**: Near obstacles, the reward penalties can be very large (for example, $around 100$ for collision, and around 10\$ for being within 0.5m). In open space, progress rewards might be low (for example, around a 1\$ more per meter toward the goal). This creates large negative gradients near obstacles but small positive gradients in open space. The actor learns primarily from the large negative signals, causing it to output minimal actions to "play it safe."

**Example scenario**: Consider a narrow corridor where the optimal policy requires moderately high forward velocity to effectively traverse the space. If exploration noise is weak and the robot only experiences slow velocities during training, the critic never learns that faster speeds that are safe in the corridor. The gradient $\nabla_a Q^\mu(s,a)$ learned from slow-speed experience points toward even slower speeds (to maximize safety margin), and the policy converges to creeping motion even though faster motion would be both safe and more efficient.

### Part 3: Exploration in Stochastic TD Methods

Stochastic policies include randomness into the policy itself rather than adding noise externally. For example, a Gaussian stochastic policy might be like:

$$
\pi_\theta(a|s) = \mathcal{N}(a | \mu_\theta(s), \sigma_\theta(s))
$$

where both the mean $\mu_\theta(s)$ and standard deviation $\sigma_\theta(s)$ can be learned.

**Qualitative differences in exploration patterns:**

**(1) Inherent vs. external exploration**: In stochastic policies, randomness is part of the policy itself. The distribution $\pi_\theta(a|s)$ inherently specifies which actions to try and how often. In deterministic DDPG, the policy $\mu_\theta(s)$ is completely deterministic, and exploration comes only from externally added noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ that's independent of the learning algorithm.

**(2) State-dependent exploration**: Stochastic policies can learn state-dependent exploration strategies. In states with high uncertainty (e.g., unfamiliar configurations), the policy can maintain high $\sigma_\theta(s)$ to explore more. In well-understood states, $\sigma_\theta(s)$ can be small for exploitation. Deterministic policies with fixed noise schedules cannot adapt exploration to state characteristics.

**(3) Policy gradient signals**: Stochastic policy gradients, for example in REINFORCE or PPO, use $\nabla_\theta \log \pi_\theta(a|s)$ to increase the probability of actions that receive high returns. This is fundamentally different from following Q-function gradients $\nabla_a Q(s,a)$. The log-probability gradient directly changes the policy distribution shape, whereas the deterministic policy gradient modifies the policy output to maximize Q-values.

**(4) Exploration-exploitation trade-off**: Many stochastic methods include entropy regularization: $J(\theta) = \mathbb{E}[R] + \beta \mathcal{H}(\pi_\theta)$, where $\mathcal{H}$ is policy entropy. This rewards different action distributions, allowing for exploration. Deterministic policies have zero entropy by definition and rely entirely on external noise for diversity.

However, stochastic policies have their own limitations for continuous control: high variance in policy outputs can create jerky, inconsistent robot motions, and variance in policy gradient estimates can slow learning. This is why deterministic policies remain popular for robotic control despite their exploration limitations.

### Part 4: Silver et al. (2014) Framework and External Exploration Assumption

Silver et al. (2014) explicitly states that the deterministic policy gradient framework assumes exploration is provided through external mechanisms, not by the policy itself. They suggest exploration strategies like:

- Additive Gaussian noise: $a = \mu_\theta(s) + \epsilon, \epsilon \sim \mathcal{N}(0, \sigma^2 I)$
- Ornstein-Uhlenbeck process: temporally correlated noise that produces smoother exploration trajectories, important for physical systems with inertia

**Implications for real-time navigation:**

**(1) Manual tuning burden**: The exploration noise parameters ($\sigma$, correlation time for OU process, decay schedule) must be tuned carefully for each task and environment. There's no automatic mechanism to adapt exploration to task difficulty or learning progress.

**(2) Decay schedules**: Exploration noise typically needs to decrease over training (e.g., $\sigma_t = \sigma_0 \cdot \gamma^t$) to allow policy convergence. But if noise decays too quickly, the policy may get stuck in local optima before finding good behaviors. If noise decays too slowly, the policy never stabilizes and continues exhibiting noisy behavior even after learning.

**(3) Environment-specific requirements**: Different navigation environments need different exploration strategies. Narrow corridors might benefit from less noise to avoid collisions during exploration. Open spaces might need more noise to discover efficient trajectories. A single fixed noise schedule may be suboptimal across diverse environments.

**(4) Off-policy learning**: A key advantage of DDPG is that it can learn from any exploratory behavior policy. The actor can be updated using Q-values learned from highly exploratory trajectories, even though the actor itself is deterministic. This separation of exploration (during data collection) and exploitation (the learned deterministic policy) is a core design principle.

For TurtleBot3 navigation in practice, this means researchers must carefully tune exploration parameters for each new environment or task, and this tuning often requires significant trial and error or domain expertise.

* * *

## Question 4 (Demonstrations, TD critic, and the limits of pure TD learning)

### Part 1: Using Human Demonstrations to Initialize DDPG

Given a set of human demonstration trajectories $\mathcal{D}_{\text{demo}} = \{(s_t, a_t, r_t, s_{t+1})\}$, these data can bootstrap DDPG training in many ways:

**Critic initialization (Q-function pretraining):**

The demonstration data can be used to pre-fill the replay buffer before any environment interaction. The critic can then be pretrained by sampling from these demonstrations and performing TD updates:

$$
\mathcal{L}_Q = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}_{\text{demo}}} [(Q_\phi(s,a) - (r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s'))))^2]
$$

This provides the critic with a "warm start", which assigns initial value estimates based on demonstrated state-action pairs rather than starting from random initialization. The critic learns to assign appropriate values to the actions humans took, providing a foundation for further refinement through Reinforcement Learning.

**Actor initialization (behavioral cloning):**

The actor can be pretrained using supervised learning to imitate the demonstrator:

$$
\mathcal{L}_{\text{BC}} = \mathbb{E}_{(s,a) \sim \mathcal{D}_{\text{demo}}} [||\mu_\theta(s) - a||^2]
$$

This teaches the actor to reproduce human actions in similar states, providing a safe, reasonable initial policy before RL fine-tuning begins.

**Combined approach**: Niu et al. (2021) demonstrated that combining both approaches, pretraining both actor and critic from demonstrations, then continuing with DDPG fine-tuning in simulation and on real robots, significantly accelerates learning for TurtleBot3 collision avoidance. They reported 60-70% reduction in training time compared to learning from scratch, and more importantly, improved safety during the learning process since the initial policy starts from safe demonstrated behaviors.

### Part 2: TD Critic Updates for Gradual Error Correction

The TD critic update in DDPG takes the form:

$$
\phi \leftarrow \phi + \alpha(y - Q_\phi(s,a))\nabla_\phi Q_\phi(s,a)
$$

where the target is:

$$
y = r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s'))
$$

**How TD learning corrects errors in demonstrated values:**

Human demonstrations often contain suboptimal actions. Humans can sometimes be overly cautious, make mistakes, or exhibit inconsistent behavior across similar situations. If we only used supervised learning (behavioral cloning), the policy would inherit these suboptimalities. TD learning provides a mechanism for correction through bootstrapping.

**(1) Initial phase**: When the critic is first trained on demonstrations, it learns approximate values $Q_\phi(s,a)$ for demonstrated state-action pairs. These initial estimates may be inconsistent; some actions might be overvalued (human was too optimistic) or undervalued (human was too cautious).

**(2) Temporal consistency**: The TD target $y = r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s'))$ enforces the Bellman equation. The equation requires that the value of taking action $a$ in state $s$ should equal the immediate reward plus the discounted value of the resulting next state. If human actions were suboptimal, the Q-values won't satisfy this consistency.

**(3) Gradual correction**: The TD error $\delta = y - Q_\phi(s,a)$ measures inconsistency. Gradient descent on this error gradually adjusts $\phi$ to make the Q-function more self-consistent. Over many updates, values propagate backward through trajectories so if an action leads to poor outcomes, its Q-value decreases; if it leads to good outcomes better than the demonstrator expected, its Q-value increases.

**Influence on actor updates through** $\nabla_a Q_\phi(s,a)$:

The actor gradient is:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\mu} [\nabla_\theta\mu_\theta(s) \nabla_a Q_\phi(s,a)|_{a=\mu_\theta(s)}]
$$

As the critic's Q-values become corrected through TD learning, the action gradients $\nabla_a Q_\phi(s,a)$ change:

- If demonstrations were too conservative (low velocities), but exploration reveals that higher velocities are safe and efficient, the critic learns to assign higher Q-values to faster actions. The gradient $\nabla_a Q_\phi(s,a)$ then points toward higher velocities, and the actor update increases the policy's output velocities.
    
- If demonstrations contained risky actions that happened to succeed by luck, but further experience reveals they often lead to collisions, the critic decreases their Q-values. The gradient points away from risky actions, and the actor learns to be more cautious than the demonstrator.
    

This TD-based correction mechanism allows the policy to exceed demonstrator performance. The agent learns from both the demonstrations (initialization) and its own experience (RL fine-tuning), discovering strategies better than those shown by humans.

### Part 3: Why Pure TD(0) Value Learning is Insufficient for Continuous Control

**Hypothetical alternative: Learning only** $V_\psi(s)$ **with TD(0):**

Consider learning only a state-value function without an actor:

$$
V_\psi(s) \leftarrow V_\psi(s) + \alpha(r + \gamma V_\psi(s') - V_\psi(s))\nabla_\psi V_\psi(s)
$$

This update learns the expected return from each state under some policy (e.g., the demonstration policy), but provides no mechanism to produce continuous control commands.

**Why this approach is insufficient:**

**(1) Absence of direct mapping from values to continuous actions:**

The value function $V_\psi(s)$ outputs a single scalar for each state, a prediction of cumulative future reward. TurtleBot3 needs two continuous outputs at every timestep: linear velocity $v \in [-v_{\max}, v_{\max}]$ and angular velocity $\omega \in [-\omega_{\max}, \omega_{\max}]$. There is no direct way to convert a scalar value into these two continuous control signals.

A approach might be: "in state $s$, try different actions and pick the one that leads to states with highest $V(s')$." However, this needs:

- A model of environment dynamics to predict $s' = f(s,a)$ for any candidate action $a$
- Solving a continuous optimization problem $\mu(s) = \arg\max_a V(f(s,a))$ at every timestep

**(2) Difficulty of deriving policy without separate optimization:**

Even with a perfect dynamics model, extracting a policy from $V(s)$ requires solving:

$$
\mu(s) = \arg\max_a \mathbb{E}_{s' \sim p(\cdot|s,a)}[r(s,a) + \gamma V_\psi(s')]
$$

For continuous action spaces, this is a non-convex continuous optimization problem that must be solved at every decision point. Methods like cross-entropy method, gradient-based optimization, or sampling-based planning could be used, but all need significant computation (hundreds or thousands of function evaluations per decision), making real-time control at 10-20 Hz not optimal.

**(3) Missing action-derivative for policy refinement:**

The key insight of actor-critic methods is that the critic provides gradients with respect to actions: $\nabla_a Q(s,a)$. This gradient points in the direction of improved actions; it tells us exactly how to adjust the current action to increase expected return.

For a state-value function, this gradient doesn't exist:

$$
\nabla_a V(s) = 0 \quad \text{(}V\text{ has no action argument)}
$$

Without action gradients, we cannot use gradient-based methods to refine a policy. We're forced to use derivative-free optimization or model-based planning, both much more computationally expensive than a single forward pass through an actor network.

**Concrete example for TurtleBot3:**

Suppose $V_\psi(s)$ correctly learns that:

- States near the goal with no obstacles nearby have high value ($V \approx 100$)
- States near obstacles have low value ($V \approx -50$)
- States far from both goal and obstacles have intermediate value ($V \approx 0$)

This tells us which states are good and bad, but not what to do about it. When the robot is in a state with $V(s) = -50$ (near obstacle), we know this is bad, but $V$ doesn't tell us:

- Should we turn left or right to avoid the obstacle?
- How much should we turn? (small $\omega$ or large $\omega$?)
- Should we slow down, stop, or maintain speed?

The action-value function $Q(s,a)$ answers these questions. We can compare $Q(s, a_1)$ vs $Q(s, a_2)$ for different candidate actions. Even better, $\nabla_a Q(s,a)$ tells us exactly how to adjust any action to improve it. The state-value function provides no such guidance.

### Part 4: Conditions Favoring DDPG Actor-Critic Over Pure TD Learning

Integrating human demonstrations with DDPG-style actor-critic is expected to outperform pure TD value learning under the following conditions:

**(1) Continuous action spaces**: When actions are continuous (velocities, forces, joint angles), direct policy parameterization via an actor network is way more efficient than trying to optimize actions from value functions at every timestep.

**(2) Real-time control requirements**: Robot navigation requires decisions at 10-20 Hz. Actor networks provide actions via a single forward pass (milliseconds), while planning or optimization from value functions requires iterative computation (seconds or longer).

**(3) Demonstrations are suboptimal but informative**: When human demonstrations are safe and reasonable but aren't optimal, the actor-critic can start from this safe initialization and improve through RL. Pure TD learning from demonstrations would only learn to evaluate the demonstrated policy, not improve beyond it.

**(4) Sample efficiency matters**: Pretraining from demonstrations dramatically reduces the number of environment interactions needed.

**(5) Safety-critical tasks**: Starting from safe demonstrated behaviors reduces dangerous exploration. Pure RL from scratch might try collision-inducing actions frequently during early training, risking hardware damage. Demonstration initialization + RL fine-tuning maintains safety while still improving performance.

**(6) Sim-to-real transfer**: Policies that combine demonstrations (often from real robots) with simulated RL training can connect the reality gap better than pure simulation-trained policies. The demonstrations provide realistic behaviors that account for real-world constraints, while RL in simulation provides sample-efficient optimization.

For TurtleBot3 navigation specifically, Niu et al. (2021) demonstrated all these advantages, achieving successful zero-shot sim-to-real transfer (policies trained in simulation work immediately on real robots) by combining human demonstrations with DDPG training, something that pure TD value learning could not accomplish.

* * *

## Question 5 (DDPG and VLA generalization)

### Part 1: Fine-Tuning VLA Action Head with Deterministic Policy Gradient

Vision-Language-Action (VLA) models have a multi-component architecture:

- **Vision encoder**: Processes visual inputs (camera images, depth maps, LIDAR) through convolutional or transformer networks (e.g., ResNet, Vision Transformer)
- **Language encoder**: Processes natural language instructions or goals through language models (e.g., BERT, T5, GPT)
- **Fusion layer**: Combines visual and language representations into a unified feature space
- **Action head**: Maps fused features to continuous robot actions $a = \mu_\theta(z)$, where $z$ are the fused features

When the perception and language components are pretrained on large datasets and approximately frozen, we can fine-tune only the action head using deterministic policy gradients.

**Mathematical formulation:**

Let the pretrained, frozen encoders b=e $f_{\text{fixed}}(\cdot)$ and the action head be $\mu_\theta(\cdot)$. The full policy is:

$$
a = \mu_\theta(f_{\text{fixed}}(\text{image}, \text{language}))
$$

The deterministic policy gradient becomes:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\mu}[\nabla_\theta\mu_\theta(z)\nabla_a Q^\mu(s,a)|_{a=\mu_\theta(z)}]
$$

where $z = f_{\text{fixed}}(\text{image}, \text{language})$ are fixed features.

**How this works in practice:**

**(1) Forward pass:**

- Extract features: $z = f_{\text{fixed}}(\text{observations})$ (no gradient tracking needed for frozen encoders)
- Compute action: $a = \mu_\theta(z)$
- Evaluate critic: $Q_\phi(s,a)$

**(2) Backward pass for actor:**

- Compute $\frac{\partial Q_\phi}{\partial a}$ (gradient of Q with respect to actions)
- Compute $\frac{\partial \mu_\theta}{\partial \theta}$ (gradient of action head output with respect to its parameters)
- Chain rule: $\frac{\partial Q_\phi}{\partial \theta} = \frac{\partial Q_\phi}{\partial a} \cdot \frac{\partial \mu_\theta}{\partial \theta}$
- Update only $\theta$ (action head parameters); frozen encoders not updated

**Computational advantages:**

- Only the small action head network requires gradient computation and parameter updates
- Pretrained representations are preserved, avoiding catastrophic forgetting
- Training is much faster than end-to-end RL, which would need to backpropagate through vision and language encoders
- The frozen encoders provide stable feature representations, reducing non-stationarity in the RL problem

### Part 2: Why Deterministic Policy Gradients Suit VLA Fine-Tuning

**Argument for DDPG-style methods in VLA fine-tuning:**

**(1) Pre-calibrated action distributions**: VLA models pretrained on large demonstration datasets already learn reasonable mappings from (vision, language) inputs to actions. The action distribution is approximately calibrated. Commands are physically feasible, semantically aligned with language instructions, and visually grounded to scene geometry.

Fully stochastic policies with high entropy might destructively upset these mappings. High exploration variance could undo the vision-language alignment achieved during pretraining. Deterministic policies make targeted, gradient-based adjustments that preserve the overall structure while optimizing for task-specific rewards.

**(2) Precise, reward-aligned corrections**: The VLA model might perform well on average but have systematic biases (e.g., always turning too sharply, moving too slowly, poor performance in certain visual contexts). Deterministic policy gradients provide precise corrections:

$$
\Delta \theta \propto \nabla_\theta\mu_\theta(z)\nabla_a Q^\mu(s,a)
$$

This update adjusts the action head in exactly the direction that increases Q-values, making minimal changes to behaviors that already work well while correcting specific failure modes. Stochastic policies would make more diffuse updates that might degrade good behaviors while fixing bad ones.

**(3) Stability with frozen encoders**: When encoders are frozen, the feature space $z$ is fixed. This converts the RL problem into a low-dimensional one (only action head parameters $\theta$ are optimized). Deterministic policy gradients are known to be stable in such settings, while high-variance stochastic policy gradients might struggle with the reduced parameter space.

**(4) Compatibility with diverse pretraining objectives**: VLA pretraining often combines supervised learning (behavioral cloning), contrastive learning (aligning vision and language), and auxiliary tasks (predicting object properties, success probabilities). These objectives don't directly optimize for RL rewards. Deterministic policy gradients provide a clean signal that respects the pretrained structure while adding reward-based optimization.

Liu et al. (2025) found that RL fine-tuning (including deterministic policy gradient methods like DDPG and stochastic methods like PPO) significantly improves VLA generalization besides supervised fine-tuning only. They showed improvements in:

- Semantic understanding (following novel language instructions not in pretraining data)
- Execution robustness (handling visual perturbations, lighting changes, object rearrangements)
- Task success rates (achieving goals rather than just imitating demonstrations)

### Part 3: TD Critic Correcting Systematic Biases in Action Head

The TD critic $Q^\mu(s,a)$ learns from sparse or delayed rewards that represent actual task objectives, not just imitation of demonstrations. The action gradient $\nabla_a Q^\mu(s,a)$ identifies how actions should change to increase expected returns.

**Example 1: Overly conservative actions (too slow)**

**Bias**: The VLA model was pretrained on demonstrations from cautious humans who moved slowly to avoid mistakes. The action head learns to output low linear velocities $v$ even when higher speeds would be safe and efficient.

**RL fine-tuning correction**:

- **Reward signal**: The task reward includes a time penalty (e.g., $-0.1$ per timestep) to encourage faster task completion
- **Critic learning**: Through exploration (additive noise) and TD learning, the critic discovers that higher velocities lead to faster goal reaching and thus higher cumulative rewards. It assigns higher Q-values to faster actions: $Q(s, v_{\text{fast}}) > Q(s, v_{\text{slow}})$
- **Action gradient**: $\frac{\partial Q}{\partial v} > 0$ (positive gradient with respect to velocity)
- **Actor update**: The gradient $\nabla_\theta J = \nabla_\theta\mu_\theta(z) \cdot \frac{\partial Q}{\partial v}$ pushes the action head parameters to increase velocity outputs. Over many updates, the action head learns to produce faster velocities in safe contexts.

**Example 2: Overly aggressive actions (too much turning)**

**Bias**: The VLA model learned from demonstrations where operators made sharp turns, but this creates oscillations in certain scenarios (e.g., narrow corridors where overcorrection leads to zigzag paths).

**RL fine-tuning correction**:

- **Reward signal**: The task reward might include a smoothness penalty on angular velocity: $r \leftarrow r - \lambda|\omega|^2$
- **Critic learning**: Smooth trajectories with moderate turning receive higher cumulative rewards than jerky trajectories. The critic learns: $Q(s, \omega_{\text{moderate}}) > Q(s, \omega_{\text{sharp}})$
- **Action gradient**: $\frac{\partial Q}{\partial \omega} < 0$ when $\omega$ is large (negative gradient favoring smaller angular velocities)
- **Actor update**: $\nabla_\theta J = \nabla_\theta\mu_\theta(z) \cdot \frac{\partial Q}{\partial \omega}$ adjusts parameters to reduce angular velocity outputs in scenarios where overcorrection occurs

**Example 3: Distribution shift (new visual contexts)**

**Bias**: The VLA model was pretrained in well-lit indoor environments but performs poorly in darker settings or outdoor scenes. It might output inappropriate actions because the frozen visual encoder's features don't perfectly match the new context.

**RL fine-tuning correction**:

- Even with frozen encoders, the action head can learn to adapt. If certain feature patterns $z$ (e.g., low-brightness features indicating dark scenes) correlate with specific systematic errors, the critic will assign appropriate Q-values
- The action gradient $\nabla_a Q(s,a)$ captures context-specific corrections: in dark scenes, maybe the robot should move more slowly due to reduced perception
- The action head learns a mapping $\mu_\theta(z)$ that implicitly conditions on these context indicators, producing appropriate context-dependent behaviors

**Qualitative mechanism of gradient-based reshaping:**

The gradient $\nabla_a Q^\mu(s,a)$ acts as a "correction vector" in action space. At each state, it points in the direction of improved actions according to the learned value function. The actor update propagates this correction back through the action head network, adjusting the weights to shift the action distribution:

$$
\mu_\theta^{\text{new}}(z) = \mu_\theta^{\text{old}}(z) + \alpha \cdot \text{(correction signal)}
$$

This is totally different from supervised learning, which only teaches the network to match provided targets. The critic-guided correction dynamically adapts based on which behaviors lead to higher task rewards, enabling discovery of improvements that demonstrators never showed.

### Part 4: Distinguishing Supervised Pretraining from RL Fine-Tuning

**Supervised pretraining on demonstration:**

Objective: Minimize prediction error on demonstrated actions

$$
\mathcal{L}_{\text{supervised}} = \mathbb{E}_{(s,a) \sim \mathcal{D}_{\text{demo}}} [||\mu_\theta(s) - a||^2]
$$

**Characteristics:**

- Learns to imitate: The model tries to reproduce exactly what demonstrators did
- No notion of long-term consequences: Each action is treated independently; there's no consideration of whether an action leads to task success
- Distribution shift problem: When deployed, the robot encounters states not in the demonstration data (due to compounding errors). The supervised model has no principled way to handle these out-of-distribution states
- Cannot exceed demonstrator performance: The best possible outcome is perfect imitation. If demonstrations are suboptimal, the learned policy will be equally suboptimal

**RL fine-tuning with TD-based critic:**

Objective: Maximize expected cumulative reward

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^T \gamma^t r_t]
$$

**Characteristics:**

- Learns to optimize: The model adjusts actions to maximize task-specific rewards, not just match demonstrations
- Long-term credit assignment: TD learning propagates value information backward through time. Actions are evaluated based on their consequences over entire trajectories, not just immediate effects
- Handles distribution shift: RL explicitly trains on the distribution of states induced by the current policy. As the policy changes, training data adapts. The model learns to handle states it actually encounters during deployment
- Can exceed demonstrations: By exploring and learning from trial-and-error, RL can discover strategies superior to those in the demonstration data

* * *

## References

1.  D. Silver, G. Lever, N. Heess, T. Degris, D. Wierstra, and M. A. Riedmiller, "Deterministic policy gradient algorithms," *ICML*, vol. 32, no. 1, pp. 387–395, June 2014.
    
2.  D. Silver, G. Lever, N. Heess, T. Degris, D. Wierstra, and M. Riedmiller, "Deterministic Policy Gradient Algorithms," in *Proceedings of the 31st International Conference on Machine Learning*, 2014.
    
3.  H. Niu, Z. Ji, F. Arvin, B. Lennox, H. Yin, and J. Carrasco, "Accelerated Sim-to-real deep reinforcement learning: Learning collision avoidance from human player," *arXiv:2102.11312*, 2021.
    
4.  J. Liu et al., "What can RL bring to VLA generalization? An empirical study," *arXiv \[cs.LG\]*, 2025.
    
5.  R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press, 2018.
