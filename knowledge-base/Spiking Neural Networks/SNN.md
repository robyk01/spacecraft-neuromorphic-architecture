### 1. Classical SNN

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

