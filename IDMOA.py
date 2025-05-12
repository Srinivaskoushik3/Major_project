import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Generate Simple Dataset
# -----------------------------
np.random.seed(42)
data = np.random.rand(100, 3)  # 100 samples, 3 features (CPU, Memory, Network)
X_train = data
y_train = data[:, -1]

# -----------------------------
# Step 2: Build BResRNN Model
# -----------------------------
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, activation='relu'), input_shape=(3, 1)),
    Dropout(0.2),
    Bidirectional(LSTM(32, return_sequences=False, activation='relu')),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Reshape input for LSTM
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

print("\n🎯 Training BResN-IDMO Model for Load Prediction...")
model.fit(X_train_reshaped, y_train, epochs=20, verbose=1)

# Prediction function
def predict_load(state):
    state_reshaped = np.array(state).reshape((1, 3, 1))
    return model.predict(state_reshaped, verbose=0)[0][0]

# -----------------------------
# Step 3: Improved Dollmaker Optimization (IDMOA)
# -----------------------------
population_size = 20
max_iterations = 25
dim = 5 * 3  # 5 containers, each has 3 features

population = np.random.rand(population_size, dim) * 0.8 + 0.1
velocity = np.random.uniform(-0.5, 0.5, size=(population_size, dim))

fitness_history = []
energy_history = []
migration_time_history = []
cost_history = []

def fitness(solution):
    predicted_load = predict_load(solution[:3])
    energy_consumption = np.sum(solution ** 2)
    migration_time = np.sum(solution) * 2
    cost = np.sum(solution) * 0.5
    total_score = (
        0.3 * predicted_load +
        0.3 * energy_consumption +
        0.2 * migration_time +
        0.2 * cost
    )
    return total_score, energy_consumption, migration_time, cost

print("\n🚀 Running Improved Dollmaker Optimization Algorithm...")

best_solution = None
best_fitness = float('inf')
best_energy = None
best_migration_time = None
best_cost = None

for iteration in range(max_iterations):
    for i in range(population_size):
        # Adaptive update
        velocity[i] = velocity[i] * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.05, 0.05, size=dim)
        population[i] += velocity[i]
        population[i] = np.clip(population[i], 0, 1)

        current_fitness, energy, migration_time, cost = fitness(population[i])

        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_solution = population[i]
            best_energy = energy
            best_migration_time = migration_time
            best_cost = cost

    fitness_history.append(best_fitness)
    energy_history.append(best_energy)
    migration_time_history.append(best_migration_time)
    cost_history.append(best_cost)

    print(f"Iteration {iteration + 1}/{max_iterations} - Best Fitness: {best_fitness:.4f}")

# -----------------------------
# Step 4: Plot Performance Graphs
# -----------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(fitness_history, label='Fitness', color='blue')
plt.title('Fitness Progression')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(energy_history, label='Energy Consumption', color='purple')
plt.title('Energy Consumption Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(migration_time_history, label='Migration Time', color='green')
plt.title('Migration Time Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Time')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(cost_history, label='Migration Cost', color='red')
plt.title('Migration Cost Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()

plt.tight_layout()
plt.savefig("bresn_idmo_performance.png")  # Save the plot as an image
# plt.show() # Uncomment if running locally

# -----------------------------
# Step 5: Print Final Results
# -----------------------------
print("\n✅ Best Solution Found:")
for i in range(5):
    print(f"Container {i + 1}: CPU={best_solution[i * 3]:.2f}, Memory={best_solution[i * 3 + 1]:.2f}, Network={best_solution[i * 3 + 2]:.2f}")

print(f"\nBest  Consumption: {best_energy:.4f}")
print(f"Best Migration Time: {best_migration_time:.4f}")
print(f"Best Migration Cost: {best_cost:.4f}")