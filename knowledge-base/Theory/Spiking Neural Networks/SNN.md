### Classical SNN

![[Nervous System.png]]

A spike (action potential) is initiated from the dendrites, travels down the axon (pre-synaptic cell), jumps across the synapse, where it connects to another neuron and potentially causes that to fire.

##### Membrane Potential:
Continous spike train over time with intervals.

![[Membrane Potential.png|697]]

##### Action Potential:
Single spike, representing an electric charge.

![[Action Potential.png]]

#### Leaky Integrate and Fire

**Membrane Potential V**:
$$\tau \frac{dV}{dt} = -V \rightarrow \text{Leak}$$
(τ represents the leak size)

When a neuron receives a spike, V increase by a synaptic weight W:
$$V \leftarrow V + W \rightarrow \text{Integrate}$$

When $V > V_t$ the neuron "fires a spike" and resets:
$$V \leftarrow 0 \rightarrow \text{Fire}$$

#### Pseudocode
```
For each time period t:
	Update V from value at time t to value at time t + dt:
		V(t+dt) = V(t)e^(-dt/tau) = alphaV(t)
	Process any incoming spikes.
		V = V + W
	Check if V crossed the threshold.
	If so:
		Emit a spike.
		Reset V.
```

For example:
$\tau = 10$ (membrane time constant)
dt = 1 ms (simulation time step)
$$V(t+dt) = V(t) \cdot e^{\frac{-dt}{\tau}} = V(t) \cdot e^{-\frac{1}{10}} = V(t) \cdot 0.904$$
(Retains 90.4% of voltage, leaks 9.6%)
This means that at each step, first let 9.6% of the voltage leak out of the membrane. Next, add W from incoming spikes. Finally, check if the membrane overflowed (> threshold).

### Euler Forward Pass
The full biophysically accurate version of the LIF:
$$u_{\text{new}} = u_t + \frac{\Delta t}{\tau_m} \Big[ -(u_t - u_{\text{rest}}) + R I_t \Big]$$
where u(t) is the membrane potential (V), $u_{rest}$ is the resting potential of the neuron, $\tau$ is a positive constant controlling the leak speed, and R is the resistance, which modules the impact of the input current I(t) on the membrane potential. We can see a full [[Neuron Simulation]].
More details about neural dynamics here: https://unaterzavia.com/neural-dynamics1/


#### 2D LIF
Add dynamic threhsold $V_t$.

**Threshold dynamics:**
$$\tau_t \frac{dV_t}{dt} = 1 - V_t$$
New spike threshold condition:
$$V > V_t$$
After a spike:
- $V \leftarrow 0$ 
- $V_t \leftarrow V_t + \delta V_t$

If $V_t > 1$, $1 - V_t$ becomes negative, causing a leak.
If $V_t < 1$, $1 - V_t$ becomes positive, resting.

#### Hodgkin-Huxley detailed models
$$I = C_m \frac{dV_m}{dt} + \bar{g}_K n^4 (V_m - V_K) + \bar{g}_{Na} m^3 h (V_m - V_{Na}) + \bar{g}_l (V_m - V_l)$$
#### Neuron Dynamics

1. **Bursting**

![[Bursting Spikes.png]]

2. **Type I/II**
3. Spatial Structures


#### Solving hard task with SNNs

##### Reservoir computing

![[Reservoir Computing.png]]

##### Surrogate Gradient Descent
![[Forward Pass.png]]

Loss is a big function $L(\theta, x)$ where $\theta$ is the vector of linear parameters. Composed of lots of simple functions (matrix multiply, sum).

Compute gradient of loss with respect to parameters:
$$[\nabla L]_i = \frac{\partial L}{\partial \theta_i}$$
Gradient descent with learning rate $\alpha$ :
$$\theta \rightarrow \theta - \alpha \nabla L$$
Chain rule:
$$x \xrightarrow{f} y \xrightarrow{g} z \quad \text{or} \quad z = g(f(x))$$
Gives:
$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx} \quad \text{or} \quad z' = g'(f(x)) \cdot f'(x)$$
Chaining the chain rule:
$$x_1 \rightarrow x_2 \rightarrow \dots \rightarrow x_n$$
Gives:
$$\frac{dx_n}{dx_1} = \frac{dx_n}{dx_{n-1}} \cdot \frac{dx_{n-1}}{dx_{n-2}} \dots \frac{dx_2}{dx_1}$$
Vectors:
Jacobian $J_{ij}^f = \frac{\partial f_i}{\partial x_j}$ 
Chain rule: $J^{g \circ f} = J^g \cdot J^f$

Pytorch:
Autograd calculates all this automatically.

#### Problems with SNNs
Threshold is V > 1, or Heaviside H(V - 1)
H'(V) = 0 for all V != 1
Gradients are all zero.

#### Solutions
1. Smooth threshold
$$\sigma(x) = \frac{1}{1 + e^{-\beta x}}$$
2. Surrogate gradient descent
Only smooth gradient (backwards pass)