The BrainChip CNN2SNN (Convolutional Neural Network to Spiking Neural Network) converts quantized CNN models obtained using [[QuantizeML]]  to a low-latency and low-power SNN for use with the Akida Runtime. 

### Conversion flow
CNN2SNN offers a simple convert function that takes a quantized model as input and converts it into an Akida runtime compatible network.

Example:
```py
model_akida = convert(quantized_model)
```

