import numpy as np
import matplotlib.pyplot as plt

# Define the system matrices
A = np.array([[1, 1], [0, 1]])  # state transition matrix
H = np.array([[1, 0]])  # measurement matrix
Q = np.array([[0.1, 0], [0, 0.1]])  # process noise covariance matrix
R = np.array([[1]])  # measurement noise covariance matrix

# Define the initial state and covariance
x0 = np.array([[0], [0]])  # initial state vector
P0 = np.array([[1, 0], [0, 1]])  # initial covariance matrix

# Generate some dummy data
N = 100
true_states = np.zeros((2, N))
measurements = np.zeros((1, N))
for i in range(N):
    true_states[:, i] = np.dot(A, true_states[:, i-1]) + np.random.multivariate_normal([0, 0], Q)
    measurements[:, i] = np.dot(H, true_states[:, i]) + np.random.normal(0, np.sqrt(R[0, 0]))

# Initialize the filter
x_hat = x0
P_hat = P0

# Run the filter
filtered_states = np.zeros((2, N))
for i in range(N):
    # Prediction step
    x_pred = np.dot(A, x_hat)
    P_pred = np.dot(np.dot(A, P_hat), A.T) + Q
    
    # Update step
    K = np.dot(np.dot(P_pred, H.T), np.linalg.inv(np.dot(np.dot(H, P_pred), H.T) + R))
    x_hat = x_pred + np.dot(K, measurements[:, i] - np.dot(H, x_pred))
    P_hat = np.dot(np.eye(2) - np.dot(K, H), P_pred)
    
    # Save the filtered state estimate
    filtered_states[:, i] = x_hat[:, 0]



# Plot the true states and measurements
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(true_states[0, :], label='True Position')
ax[0].plot(measurements[0, :], '.', label='Measurements')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Position')
ax[0].legend()
ax[1].plot(true_states[1, :], label='True Velocity')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Velocity')
ax[1].legend()
fig.suptitle('True States and Measurements')

# Plot the filtered states
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(filtered_states[0, :], label='Filtered Position')
ax[0].plot(true_states[0, :], label='True Position')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Position')
ax[0].legend()
ax[1].plot(filtered_states[1, :], label='Filtered Velocity')
ax[1].plot(true_states[1, :], label='True Velocity')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Velocity')
ax[1].legend()
fig.suptitle('Filtered States')
plt.show()
