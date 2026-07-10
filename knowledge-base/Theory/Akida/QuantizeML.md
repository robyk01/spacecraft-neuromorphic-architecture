### Quantization
**Quantization** is the process of mapping continuous or high-precision numerical values to smaller, discrete set of finite values.

It reduces the bit-width of model parameters (eg. converting 32-bit floating point numbers to 8-bit or 4-bit integers) so models size decreases. 
Because of the reduced bit-width, lower precision integer arithmetic require less processing power and runs significantly faster on hardware.


### QuantizeML toolkit
QuantizeML uses a FixedPoint representation in place of float values for layers inputs, weights and outputs. FixedPoint numbers are actually integers with static number of fractional bits.

##### Symmetric / Fixed-Point Quantization
In a symmetric setup, the floating-point range is clipped so that it sits perfectly centered around zero.
$$
x_{float} \approx x_{int} \cdot 2^{-x_{frac-bits}}
$$

The precision of the representation is directly related to the number of fractional bits. For example, representing PI, using an 8-bit FP with varying fractional bits:

| frac_bits | x_int | float value |
| --------- | ----- | ----------- |
| 1         | 6     | 3.0         |
| 3         | 25    | 3.125       |
| 6         | 201   | 3.140625    |
For short, to find the int value of a float number we just multiply the int with $2^{decimal_count}$ .
##### Asymmetric / Affine Quantization
In real-word neural networks, activations are rarely balanced around zero. For example, ReLU completely deletes negative numbers, meaning its output distribution might be strictly positive, like (0.0, 25.0).
$$x_{float} = S \times (x_{int} - Z)$$

$$x_{int} = \text{round}\left(\frac{x_{float}}{S}\right) + Z$$
- **Scale (S)**: A floating-point value that represents the exact step size between adjacent integer values.
- **Zero-Point (Z)**: An integer data type that acts as an adjustable anchor. It tells the processor which integer value corresponds to a real-world float of 0.0.


#### **Summary:**
While asymmetric quantization gives a significantly better mathematical precision, it adds extra transistor logic, registers, and clock overhead to every incoming spike computation loop. Akida's goal is ultra-low-power neuromorphic hardware deployment, so it sticks with symmetric quantization.

More informations here: https://doc.brainchipinc.com/user_guide/quantizeml.html
