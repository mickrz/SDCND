#CarND-MPC-Project
Self-Driving Car Engineer Nanodegree Program

---
## Rubric points:

###The Model: 
*Student describes their model in detail. This includes the state, actuators and update equations.*
I used the kinematic model outlined in class. The model uses the states - car's x and y position, orientation, velocity, cross-track error and orientation error. The model also includes the control outputs - acceleration and steering angle. To calculate the state and actuation for the current timestep, the state and actuation from the previous timestep is used .

###Timestep Length and Elapsed Duration (N & dt):
*Student discusses the reasoning behind the chosen N (timestep length) and dt (elapsed duration between timesteps) values. Additionally the student details the previous values tried.*
I used the value of 10 and 0.1 for N and dt respectively. I also tried other combinations, but these values were suggested by Udacity. Some of the values I tried were (15, 0.1) = 1.5s in the horizon, (25, 0.5) = 12.5s in the horizon, (20, 0.5) = 10s in the horizon and so on. 

As noted in the lessons: "MPC attempts to approximate a continuous reference trajectory by means of discrete paths between actuations. Larger values of dt result in less frequent actuations, which makes it harder to accurately approximate a continuous reference trajectory."

In this case, one second is sufficient, but would likely need to be tweaked for other use cases.

###Polynomial Fitting and MPC Preprocessing:
*A polynomial is fitted to waypoints. If the student preprocesses waypoints, the vehicle state, and/or actuators prior to the MPC procedure it is described.*
Using polyfit in main.cpp, I passed in the waypoints ptsx and ptsy after transforming them to the vehicles perspective (lns 101-112).

###Model Predictive Control with Latency:
*The student implements Model Predictive Control that handles a 100 millisecond latency. Student provides details on how they deal with latency.*
Latency is a common obstacle when dealing with hardware or networks. There is a 100ms delay to simulate the propagation of commands through the system. Again the kinematic equations is used to handle the latency (lns 112-117). Also to help control the oscillation from left to right (and vice versa), I used multipliers to punish steering angle, cross-track error and orientation error which helps stabilize the car along the path (lns 52-67).
