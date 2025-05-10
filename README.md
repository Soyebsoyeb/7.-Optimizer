OPTIMIZERS IN NEURAL NETWORK


ABOUT THE STRUCTURE OF THE DATASET:->
🔹 Dense layers
🔹 Activation functions (ReLU & Softmax)
🔹 Loss function (Categorical Cross-Entropy)
🔹 Optimization (SGD)

Core Components:->
Layer_Dense: Fully connected neural network layer 🔗
Activation_ReLU: ReLU activation (with backpropagation) ⚡
Activation_Softmax: Stable softmax function 📈
Loss_CategoricalCrossentropy: Categorical loss with gradients 🎯
Activation_Softmax_Loss_CategoricalCrossentropy: Efficient combo 🔥


Training
✅ Complete forward & backward propagation
✅ SGD optimizer implementation
📊 Accuracy & loss tracking


🖼️ Visualization
Spiral data visualization 🌪️
Optional training progress plots 📉



Key Implementation Details
🔁 Forward Propagation
Layer math:
output = inputs @ weights + biases

ReLU:
max(0, x)

Softmax (w/ numerical stability):
exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)



🔄 Backward Propagation
GRADIENTS:->
Weights: inputs.T @ dvalues
Biases: np.sum(dvalues, axis=0)
Inputs: dvalues @ weights.T
ReLU derivative: 0 where input < 0
Combined Softmax + CrossEntropy: Optimized for performance ⚡


⚙️ Optimization

(1) GRADIENT DESCENT
Stochastic Gradient Descent (SGD):
layer.weights += -learning_rate * layer.dweights
layer.biases += -learning_rate * layer.dbiases
