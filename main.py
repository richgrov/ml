import random

dataset = []
for current in range(70, 80):
  for target in range(70, 80):
    dataset.append({'current': current, 'target': target, 'time': max(target - current, 0) * 5})
print(f'dataset contains {len(dataset)} entries')

def normalize_temp(temp):
  return (temp - 32) / (100 - 32)

def normalize_time(t):
  return t / 60

current_weight = random.uniform(-1, 1)
target_weight = random.uniform(-1, 1)
bias = random.uniform(-1, 1)

learning_rate = 0.01
num_epochs = 50000

def forward(current_temp, target_temp):
  output = (current_temp * current_weight) + (target_temp * target_weight) + bias
  return max(output, 0) # ReLU

for epoch in range(num_epochs):
  total_loss = 0

  for entry in dataset:
    current = normalize_temp(entry['current'])
    target = normalize_temp(entry['target'])
    duration = normalize_time(entry['time'])

    output = forward(current, target)
    loss = (output - duration) ** 2
    total_loss += loss

    gradient = (output - duration) * 2
    current_weight_gradient = gradient * current
    target_weight_gradient = gradient * target
    bias_gradient = gradient

    current_weight -= learning_rate * current_weight_gradient
    target_weight -= learning_rate * target_weight_gradient
    bias -= learning_rate * bias_gradient

  avg_loss = total_loss / len(dataset)
  if epoch % 100 == 0:
    #print(f"prediction: {output} ({round(output * 60)} mins)")
    print("mean squared loss: " + str(avg_loss))
    #print(f"gradients: {gradient} {current_weight_gradient} {target_weight_gradient} {bias_gradient}")

print(f"{current_weight} {target_weight} {bias}")

while True:
  input_current = (float(input()) - 32) / (100 - 32)
  input_target = (float(input()) - 32) / (100 - 32)
  output = forward(input_current, input_target)
  print(f"it will take {round(output * 60)} mins")
