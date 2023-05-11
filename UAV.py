import numpy as np
import matplotlib.pyplot as plt

# Define the system matrices
A = np.array([[1, 0.1], [0, 1]])  # state transition matrix
B = np.array([[0], [1]])  # input matrix
C = np.array([[1, 0]])  # measurement matrix
Q = np.array([[0.01, 0], [0, 0.1]])  # process noise covariance matrix
R = np.array([[1]])  # measurement noise covariance matrix

# Define the initial state and covariance
x0 = np.array([[0], [0]])  # initial state vector
P0 = np.array([[1, 0], [0, 1]])  # initial covariance matrix

# Define the control gains
Kp = 1.0  # proportional gain
Kd = 0.5  # derivative gain

# Generate some dummy data
N = 1000
true_states = np.zeros((2, N))
measurements = np.zeros((1, N))
velocity = np.zeros((1, N))
for i in range(N):
    true_states[:, i] = np.dot(A, true_states[:, i-1]) + np.random.multivariate_normal([0, 0], Q)
    measurements[:, i] = np.dot(C, true_states[:, i]) + np.random.normal(0, np.sqrt(R[0, 0]))
    velocity[:, i] = true_states[1, i]

# Initialize the filter and controller
x_hat = x0
P_hat = P0
u = np.zeros((2, N))

# Run the filter and controller
filtered_states = np.zeros((2, N))
estimated_cov = np.zeros((2,N))
P_hat_trace = np.zeros(N)
for i in range(N):
    # Prediction step
    x_pred = np.dot(A, x_hat)
    P_pred = np.dot(np.dot(A, P_hat), A.T) + Q
    
    # Update step
    K = np.dot(np.dot(P_pred, C.T), np.linalg.inv(np.dot(np.dot(C, P_pred), C.T) + R))
    x_hat = x_pred + np.dot(K, measurements[:, i] - np.dot(C, x_pred))
    P_hat = np.dot(np.eye(2) - np.dot(K, C), P_pred)
    
    # Control step
    error = x_hat[0, 0] - true_states[0, i]
    d_error = x_hat[1, 0]
    u[:, i] = -Kp*error - Kd*d_error
    
    # Save the filtered state estimate
    filtered_states[:, i] = x_hat[:, 0]
    estimated_cov[:,i] = P_hat[:,0]
    P_hat_trace[i] = np.trace(P_hat)
# Plot the results
fig, ax = plt.subplots(3, 1, figsize=(8, 12))
ax[0].plot(true_states[0, :], label='True Altitude')
ax[0].plot(measurements[0, :], '.', label='Measurements')
ax[0].plot(filtered_states[0, :], label='Filtered Altitude')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Altitude')
ax[0].legend()
ax[1].plot(u[0, :], label='Control Input')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Pitch Angle')
ax[1].legend()
ax[2].plot(velocity[0, :], label='True Velocity')
ax[2].plot(filtered_states[1, :], label='Estimated Velocity')
ax[2].set_xlabel('Time')
ax[2].set_ylabel('Velocity')
ax[2].legend()
# ax[2].plot(P_trace)
# ax[2].set_xlabel('Time')
# ax[2].set_ylabel('Trace of P')
plt.show()

