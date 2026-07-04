As we know, the LIS first order differential ecuation:
$$\tau_m \frac{du}{dt} = -[u(t) - u_{rest}] + RI(t)$$
The classic definition of a derivative:
$$
\frac{du}{dt} = \frac{u[t + \Delta t] - u[t]}{\Delta t}
$$
Then we discretize and solve for $u[t + \Delta t]$:
$$
u[t + \Delta t] = \frac{\tau_mu[t] - u(t)\Delta t + u_{rest}\Delta t + RI(t)\Delta t}{\tau_m}
$$$$ u_{\text{new}} = u_t + \frac{\Delta t}{\tau_m} \Big[ -(u_t - u_{\text{rest}}) + R I_t \Big]$$
So the code will look like:
```python
times = np.arange(0, T, delta_t)
U = np.zeros_like(times) # store voltages
I = np.zeros_like(times) # store input current
output_spike_times = []
u_t = 0

for t_idx in range(len(times)):
	u_t = (tau * u_t - delta_t * u_t + delta_t * u_rest + R * delta_t * I[t_idx]) / tau
	U[t_idx] = u_t

	if u_t >= threshold:

		output_spike_times.append(times[t_idx])
		u_t = 0
```
