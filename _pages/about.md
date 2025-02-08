---
permalink: /
title: "Physics-Informed Neural Networks (PINNs): Bridging Data and Physical Laws"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---
![Block Diagram](/images/PINN-diagram.png){: .align-right width = "50%"}


Physics-Informed Neural Networks (PINNs) are a powerful fusion of deep learning and fundamental physical laws. Unlike traditional neural networks that rely solely on data, PINNs incorporate partial differential equations (PDEs) to ensure that predictions remain consistent with the governing physics of a system.This unique approach helps PINNs to make accurate predictions even with limited data making them very effective for solving complex scientific problems. From fluid mechanics to quantum physics, PINNs are transforming the way researchers model and understand the physical world.

Problem formulation in PINNs
------
Physics-Informed Neural Networks (PINNs) are designed to solve partial differential equations (PDEs) by incorporating physical laws directly into the learning process. These physical laws are represented in the form of a PDE, where the goal is to find a solution that satisfies both the equation itself and the initial and boundary conditions.

A PDE typically describes how a function changes over both time and space. The primary goal of a PINN is to minimize a composite loss function that incorporates different aspects of the problem, ensuring that the network's predictions align with both the physical model and the given conditions.

The composite loss function
------
Training of a PINN invloves minimizing the composite loss function, which is a combination of three key components:

### Initial Condition Loss

 This ensures that the network's prediction at the start of the problem (i.e., at time t=0) matches the provided initial condition. The loss penalizes the network if its output deviates from the known initial values.

 $$
L_{ic}(\theta) = \frac{1}{N_{ic}} \sum_{i=1}^{N_{ic}} \left| u_{\theta}(0, x_c^i) - g(x_c^i) \right|^2
$$


### Boundary Condition Loss: 

This component enforces that the solution meets the boundary conditions at the spatial domain boundaries. If the network’s prediction at the boundaries doesn’t satisfy the required physical behavior, this loss term penalizes the network.

$$
L_{bc}(\theta) = \frac{1}{N_{bc}} \sum_{i=1}^{N_{bc}} \left| B[u_{\theta}](t_{bc}^i, x_{bc}^i) \right|^2
$$


### PDE Residual Loss: 
The core of the PINN approach is ensuring that the network's predictions satisfy the PDE itself. This loss term calculates how well the network's output adheres to the governing differential equation by minimizing the residual of the PDE.

$$
L_r(\theta) = \frac{1}{N_r} \sum_{i=1}^{N_r} \left| R_{\theta}(t_r^i, x_r^i) \right|^2
$$



The goal of minimizing this term is to ensure that the neural network’s learned solution respects the underlying physical equations governing the system.

### Final Composite Loss Function

By combining all three losses, we define the total loss function as:

$$ L(\theta) = L_{ic}(\theta) + L_{bc}(\theta) + L_{r}(\theta) $$


This function serves as the objective for optimization. The training process adjusts the network parameters 𝜃 to minimize this total loss, ensuring that the predictions satisfy the initial
conditions, boundary conditions, and PDE itself.


Training of a PINN
------
The training process of a Physics-Informed Neural Network (PINN) involves several key steps that ensure the network learns a solution that satisfies both the given initial and boundary conditions as well as the underlying partial differential equation (PDE). 

![PINN training](/images/PINNtr.png){: width = "50%"}

#### *Figure 1: Training of a conventional PINN*

### Neural Network Representation
At the core of the architecture is a fully connected neural network that takes spatial and temporal coordinates u(x,t) as inputs and produces an approximation of the solution function u(x,t). The network parameters (weights and biases) are updated iteratively to improve accuracy.

### Automatic Differentiation
PINNs leverage automatic differentiation to compute derivatives of the predicted solution u(x,t). These derivatives are essential for computing the PDE residual loss.

### Loss Function Computation
loss components are summed together to form the composite loss function, which acts as the optimization objective.

The neural network parameters are updated iteratively using an optimization algorithm (such as gradient descent) to minimize the total loss. The training process continues until the loss function converges below a predefined threshold ϵ, ensuring that the network has learned an accurate solution.


Challenges in Training PINNs
------
The training process of PINNs is often hindered by several critical pathologies. These challenges can degrade the accuracy, robustness, and physical reliability of the learned solution. Below, we explore some of the key issues faced during training.

### Spectral Bias
PINNs tend to struggle with learning high-frequency components of a solution. This is because standard neural networks typically learn smooth, low-frequency features, making it difficult to approximate sharp gradients.

### Causality Violation
Traditional training methods may fail to respect the inherent temporal dependencies in PDE solutions. This can lead to scenarios where later time steps are learned incorrectly because earlier ones were not sufficiently accurate.

### Unbalanced Loss Gradients
The composite loss function in PINNs consists of multiple terms, including initial condition loss, boundary condition loss, and PDE residual loss. If these terms propagate gradients unevenly, optimization can become unstable or biased, leading to poor generalization.

Proposed training pipeline
------
To overcome these challenges the following training pipeline has been proposed:

![Training pipeline](/images/trainingPipeline.png){: width = "50%"}

#### *Figure 2: Training pipeline of a PINN*
##### *source: Wang, Sifan, Shyam Sankaran, Hanwen Wang, and Paris Perdikaris. 2023. An Expert's Guide to Training Physics-informed Neural Networks.*

### PDE Non-Dimensionalization:

Non-dimensionalization is a technique used to simplify and analyze physical systems by scaling variables such that they become dimensionless and fall within a reasonable numerical range.The key benefits of non-dimensionalization include:

- Preventing vanishing gradients
- Avoiding numerical dominance of certain variables
- Improving convergence speed and stability


### Selecting a suitable Network Architecture:

Selecting a suitable architecture is also critical for ensuring that PINNs can efficiently learn complex PDE solutions. The following network design choices have been proposed:


- **Multi-Layer Perceptrons (MLPs):**  MLPs are universal approximatiors, mapping spatial and temporal coordinates to solution values. 

  - **Recommended depth and width:** 3–6 layers with 128–512 neurons per layer.
  - **Activation functions:** The Tanh function is preferred as it provides smooth, differentiable outputs. Functions like ReLU are avoided due to their zero second-order derivative, which negatively impacts PDE residual computation.
  - **Initialization:** The Glorot scheme is used to initialize weights, ensuring balanced gradient flow during training.

- **Random Fourier Feature Embeddings:** This technique is used to counteract spectral bias. RWF transforms input coordinates into a higher-dimensional representation using sinusoidal functions before passing them through the MLP.
  - **Why it works:** Fourier embeddings allow the network to represent high-frequency components more effectively, improving its ability to capture sharp transitions in PDE solutions.
- **Random Weight Factorization (RWF):** RWF is a method that improves model convergence and robustness by factorizing weight matrices into scaling factors and direction vectors. Instead of learning raw weights directly, this technique reformulates them as:    $$ W = \text{diag}(\exp(s)) \cdot V $$
 

  where **s** is a trainable scale factor and **V** represents the weight matrix. Advanteages of using this approch are: 

  - More stable optimization
  - Improved generalization
  - Faster training convergence


### Training: 

- **Loss Balancing:** PINNs struggle with multi-scale losses, making manual weight selection impractical. We use a self-adaptive loss balancing approach by:

  - Normalizing gradient norms across different loss terms.
  - Updating weights dynamically to prevent bias toward specific losses.
  - Exploring NTK-based weighting, though it is computationally heavier.


- **Respecting Temporal Causality**:

  Physics-Informed Neural Networks (PINNs) often violate temporal causality when solving time-dependent PDEs. To address this, the following can be done: 

  - Split the temporal domain into segments.
  - Assign exponential temporal weights to ensure solutions progress sequentially.
  - Use a carefully chosen causality parameter ϵ to balance optimization difficulty.

- **Curriculum Training**: 
  For complex problems like high Reynolds number fluid dynamics, standard PINN training fails. A curriculum approach helps by:

  - Breaking training into smaller, manageable time windows.
  - Using prior solutions as initialization for subsequent steps.
  - Gradually increasing difficulty, such as solving lower Reynolds numbers first.

Algorithm for training PINNs affectively
------

| Step                          | Description                                                                                       | Key Details                                                                          |
|-------------------------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| **Step 1: Preparation**       | Non-dimensionalize the PDE system                                                               | Ensures stability and consistency.                                                   |
| **Step 2: Network Design**    | Represent the solution using an MLP with Fourier Feature Embeddings and Random Weight Factorization. | Use **Fourier embeddings**, tanh activation, and Glorot initialization.            |
| **Step 3: Loss Function**     | Combine losses for initial conditions, boundary conditions, and physics laws.                   | $$ L(\theta) = \lambda_{ic} L_{ic}(\theta) + \lambda_{bc} L_{bc}(\theta) + \lambda_{r} L_{r}(\theta) $$|
| **Step 4: Weight Setup**      | Initialize all weights to 1.                                                                    | $$ \text{Global: } (\lambda_{ic}, \lambda_{bc}, \lambda_r); \text{ Temporal: } (w_i = 1) $$      |
| **Step 5: Training Loop**     | Train the network using gradient descent.                                                       | $$ \text{Adjust temporal, global weights, update parameters } \theta. $$                    |

Results
------

### Allen-Cahn equation:

This is a widely studied non-linear PDE used to model phase transitions in physical systems. The equation is as follows:

  $$ u_t - 0.0001 u_{xx} + 5u^3 - 5u = 0, \quad u(0,x) = x^2 \cos(\pi x), \quad t \in [0,1], x \in [-1,1] $$

- **Issues with conventional PINNs:** After analyzing the performance of a conventional PINN model applied to the 1D Allen-Cahn equation three main issues are observed, wich are evident from the analysis figure below.

![Allen cahn](/images/allenCahn.png)
#### *Figure 3: Training analysis of a conventional PINN model for 10,000 iterations. Top left: Histograms of back-propagated gradients for PDE residual and initial condition losses, showing gradient imbalance. Top right: Temporal PDE residual loss, indicating causality violation. Bottom: NTK eigenvaluesvrevealing spectral bias.*
##### *source: Wang, Sifan, Shyam Sankaran, Hanwen Wang, and Paris Perdikaris. 2023. An Expert's Guide to Training Physics-informed Neural Networks.*


  - **Gradient Imbalances:** Back-propagated gradients for the PDE residual loss dominate those of the initial condition loss, resulting in unbalanced updates during training.

  - **Spectral Bias:** The NTK eigenvalues show rapid decay, indicating the model’s struggle to capture high-frequency details in the solution, as shown in the figure.

  - **Causality Violation:** The model minimizes PDE residuals at later times first, violating the physical causality of the problem.

### Abalation Study:
Abalation study is the method which is used to study the imoact of different compnonents on the performance of the system. The models performance on the can be observed by disabling certain components of the algorithm as shown below,

![Allen cahn](/images/benchmark.png)

#### *Figure 4: Abalation study results for Allen-Cahn equation*
##### *source: Wang, Sifan, Shyam Sankaran, Hanwen Wang, and Paris Perdikaris. 2023. An Expert's Guide to Training Physics-informed Neural Networks.*

 As shown in figure 5,  the full algorithm yields the best performance with a relative L2 error of 5.84×10−4. Disabling any component results in a degradation of performance, with the most significant drop observed when the Fourier Feature embedding is removed (error 4.35×10 −1
 ). This indicates that addressing spectral bias is crucial for improving model accuracy. Run-times across configurations are consistent, with conventional PINNs showing a slightly shorter run-time of 12.93 minutes.

Conclusion
------
The 1D Allen-Cahn equation revealed critical challenges for conventional PINNs, including gradient imbalances, spectral bias, and causality violations. These issues were effectively addressed by the proposed training pipeline.

### Key Contributions:

Fourier feature embeddings and random weight factorization significantly enhanced model performance, mitigating spectral bias. Gradient normalization and causality weighting ensured balanced training and maintained correct temporal relationships. The full algorithm achieved state-of-the-art results, with a relative L2 error of 5.84×10 −4, demonstrating its robustness and effectiveness.

Suggested Improvements
------
- While gradient descent and its variants are widely used, further exploration into more advanced optimization algorithms (such as L-BFGS or adaptive optimizers) could potentially improve convergence speed and accuracy.

- Combining PINNs with other deep learning architectures such as recurrent neural networks (RNNs) for time-series problems or convolutional neural networks (CNNs) for spatially distributed solutions could be used in other domians as well.

- In many cases, limited data can reduce the accuracy of PINNs. Using synthetic data generation methods or incorporating physics-informed augmentation techniques could provide the network with richer input, helping to bridge the gap between sparse observational data and robust physical models.


References
------
Wang, Sifan, Shyam Sankaran, Hanwen Wang, and Paris Perdikaris. An Expert's Guide to Training Physics-Informed Neural Networks. 2023.

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations. Journal of Computational Physics, 378, 686-707.

Karniadakis, G. E., Lu, L., & Perdikaris, P. (2021). Physics-Informed Machine Learning. SIAM Review, 63(3), 495-523.

Sirignano, J., & Spiliopoulos, K. (2018). DGM: A Deep Learning Algorithm for Solving Partial Differential Equations. Journal of Computational Physics, 375, 1339-1364.
