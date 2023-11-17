import math
import random

dataset = []
for x in range(1, 20):
  for y in range(1, 20):
    dataset.append({'x': x, 'y': y, 'hypot': math.hypot(x, y)})

class Layer:
  def __init__(self, num_in, num_out, activation, activation_derivative) -> None:
    self.num_in = num_in
    self.num_out = num_out
    self.activation = activation
    self.activation_derivative = activation_derivative
    self.weights = [[random.uniform(-1, 1) for _ in range(num_out)] for _ in range(num_in)]
    self.weighted_inputs = []
    self.bias = [random.uniform(-1, 1) for _ in range(num_out)]
    self.cost_weighted_input_derivatives = []

  def forward(self, inputs):
    assert len(inputs) == self.num_in
    self.inputs = inputs
    self.weighted_inputs = []

    results = []
    for out in range(self.num_out):
      total = self.bias[out]
      for inp in range(len(inputs)):
        total += inputs[inp] * self.weights[inp][out]
      self.weighted_inputs.append(total)
      results.append(self.activation(total))

    return results 

  def backward(self, cost_gradient, learning_rate):
    self.cost_weighted_input_derivatives = [0] * self.num_out

    for out in range(self.num_out):
      self.cost_weighted_input_derivatives[out] = cost_gradient * self.activation_derivative(self.weighted_inputs[out])

      for inp in range(self.num_in):
        weight_gradient = self.cost_weighted_input_derivatives[out] * self.inputs[inp]
        self.weights[inp][out] -= learning_rate * weight_gradient
  
      self.bias[out] -= learning_rate * self.cost_weighted_input_derivatives[out]

  def backward_from(self, learning_rate, next_layer):
    self.cost_weighted_input_derivatives = [0] * self.num_out

    for out in range(self.num_out):
      gradient = 0

      for next_out in range(next_layer.num_out):
        gradient += next_layer.weights[out][next_out] * next_layer.cost_weighted_input_derivatives[next_out]

      self.cost_weighted_input_derivatives[out] = gradient * self.activation_derivative(self.weighted_inputs[out])

      for inp in range(self.num_in):
        weight_gradient = self.cost_weighted_input_derivatives[out] * self.inputs[inp]
        self.weights[inp][out] -= learning_rate * weight_gradient

      self.bias[out] -= learning_rate * self.cost_weighted_input_derivatives[out]

def relu(n):
  return max(n, 0)

def relu_derivative(n):
  return 1 if n > 0 else 0

def sigmoid(n):
  return 1.0 / (1.0 + math.exp(-n))

def sigmoid_derivative(n):
  return sigmoid(n) * (1 - sigmoid(n))

inputs = Layer(2, 3, sigmoid, sigmoid_derivative)
hidden1 = Layer(3, 3, sigmoid, sigmoid_derivative)
hidden2 = Layer(3, 1, sigmoid, sigmoid_derivative)

def forward(a, b):
  results1 = inputs.forward([a, b])
  results2 = hidden1.forward(results1)
  results3 = hidden2.forward(results2)
  return results3[0]

learning_rate = 0.005
epochs = 10000

def normalize(side_length):
  return side_length / 20

def unnormalize(normalized):
  return normalized * 20

def cost(expected, actual):
  return (expected - actual) ** 2

def cost_derivative(expected, actual):
  return (expected - actual) * 2

for epoch in range(1, epochs + 1):
  total_loss = 0
  for entry in dataset:
    a = normalize(entry['x'])
    b = normalize(entry['y'])
    c = normalize(entry['hypot'])

    output = forward(a, b)
    loss = cost(c, output)
    total_loss += loss

    gradient = cost_derivative(output, c)
    
    hidden2.backward(gradient, learning_rate)
    hidden1.backward_from(learning_rate, hidden2)
    inputs.backward_from(learning_rate, hidden1)

  if epoch % 100 == 0:
    average_loss = total_loss / len(dataset)
    print(f"epoch {epoch}/{epochs}, loss {average_loss}")

while True:
  a = normalize(float(input()))
  b = normalize(float(input()))
  c = unnormalize(forward(a, b))
  print(c)
