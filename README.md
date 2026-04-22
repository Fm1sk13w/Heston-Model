# Heston-Model
## authors: Oliwier Arent, Filip Miśkiewicz, Natalia Przewdzięk
This project focuses on the numerical evaluation of the Heston stochastic volatility model for pricing financial derivatives using Python.

**General topic**
This project focuses on the numerical evaluation of the Heston stochastic volatility model for pricing financial derivatives. While standard models like Black-Scholes assume constant volatility, the Heston model treats variance as a stochastic process, resulting in a complex, two-dimensional parabolic partial differential equation (PDE) with a mixed cross-derivative term. The primary focus of our work will be on the numerical analysis and computational techniques required to solve this degenerate 2D PDE stably and efficiently. We will also check the model's accuracy based on the real data. The project will be implemented in Python.

**Objectives**
The core objectives of our project are divided into mathematical formulation, numerical implementation, and benchmarking:
PDE Formulation and Boundary Conditions: Define the 2D Heston PDE and properly formulate the boundary conditions.
Numerical Solver Implementation: Design and implement an Alternating Direction Implicit (ADI) finite difference scheme to efficiently solve the PDE while maintaining stability.
Semi-Analytical Benchmarking: Implement Heston's semi-analytical solution using characteristic functions and Fourier inversion to serve as a ground-truth benchmark for testing the convergence and accuracy of our numerical PDE solver.
Real-World Application: Extract real-world option pricing data to calibrate the PDE parameters and visualize the resulting implied volatility surface.
