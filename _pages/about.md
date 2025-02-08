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

![Initial Condition](/images/initialLoss.png){: width = "50px"}

  - Lic(θ) is the loss function enforcing the initial condition.
  - Nic is the number of training points used to enforce the initial condition.
  - uθ(0,xci) is the network's predicted solution at the initial time for a given spatial point.
  - g(xci) is the actual initial condition value at xci.
​

### Boundary Condition Loss: 

This component enforces that the solution meets the boundary conditions at the spatial domain boundaries. If the network’s prediction at the boundaries doesn’t satisfy the required physical behavior, this loss term penalizes the network.

![Boundary Condition](/images/Boundloss.png){: width = "50%"}

  - Lbc(θ) is the loss function enforcing boundary conditions.
  - Nbc is the number of training points used to enforce the boundary condition.
  - B[uθ] is the boundary condition operator applied to the predicted solution at time t and spatial location x.

### PDE Residual Loss: 
The core of the PINN approach is ensuring that the network's predictions satisfy the PDE itself. This loss term calculates how well the network's output adheres to the governing differential equation by minimizing the residual of the PDE.

![Residual Loss](/images/resloss.png){: width = "50%"}

  - Lr(θ) is the loss function enforcing the PDE constraints.
  - Nr is the number of training points sampled for checking PDE satisfaction.
  - Rθ(t,x) is the PDE residual, which measures how well the network's predictions satisfy the differential equation.

The goal of minimizing this term is to ensure that the neural network’s learned solution respects the underlying physical equations governing the system.

### Final Composite Loss Function

By combining all three losses, we define the total loss function as:

![Composite Loss](/images/comploss.png){: width = "50%"}

This function serves as the objective for optimization. The training process adjusts the network parameters 𝜃 to minimize this total loss, ensuring that the predictions satisfy the initial
conditions, boundary conditions, and PDE itself.


Training of a PINN
------
The training process of a Physics-Informed Neural Network (PINN) involves several key steps that ensure the network learns a solution that satisfies both the given initial and boundary conditions as well as the underlying partial differential equation (PDE). 

![PINN training](/images/trpinn.png){: width = "50%"}

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

- Spectral Bias
PINNs tend to struggle with learning high-frequency components of a solution. This is because standard neural networks typically learn smooth, low-frequency features, making it difficult to approximate sharp gradients.

- Causality Violation
Traditional training methods may fail to respect the inherent temporal dependencies in PDE solutions. This can lead to scenarios where later time steps are learned incorrectly because earlier ones were not sufficiently accurate.

- Unbalanced Loss Gradients
The composite loss function in PINNs consists of multiple terms, including initial condition loss, boundary condition loss, and PDE residual loss. If these terms propagate gradients unevenly, optimization can become unstable or biased, leading to poor generalization.

Proposed training pipeline
------
To overcome these challenges the following training pipeline has been proposed:
![Training pipeline](/images/pipeline.png){: width = "50%"}


For site content, there is one markdown file for each type of content, which are stored in directories like _publications, _talks, _posts, _teaching, or _pages. For example, each talk is a markdown file in the [_talks directory](https://github.com/academicpages/academicpages.github.io/tree/master/_talks). At the top of each markdown file is structured data in YAML about the talk, which the theme will parse to do lots of cool stuff. The same structured data about a talk is used to generate the list of talks on the [Talks page](https://academicpages.github.io/talks), each [individual page](https://academicpages.github.io/talks/2012-03-01-talk-1) for specific talks, the talks section for the [CV page](https://academicpages.github.io/cv), and the [map of places you've given a talk](https://academicpages.github.io/talkmap.html) (if you run this [python file](https://github.com/academicpages/academicpages.github.io/blob/master/talkmap.py) or [Jupyter notebook](https://github.com/academicpages/academicpages.github.io/blob/master/talkmap.ipynb), which creates the HTML for the map based on the contents of the _talks directory).

**Markdown generator**

The repository includes [a set of Jupyter notebooks](https://github.com/academicpages/academicpages.github.io/tree/master/markdown_generator
) that converts a CSV containing structured data about talks or presentations into individual markdown files that will be properly formatted for the Academic Pages template. The sample CSVs in that directory are the ones I used to create my own personal website at stuartgeiger.com. My usual workflow is that I keep a spreadsheet of my publications and talks, then run the code in these notebooks to generate the markdown files, then commit and push them to the GitHub repository.

How to edit your site's GitHub repository
------
Many people use a git client to create files on their local computer and then push them to GitHub's servers. If you are not familiar with git, you can directly edit these configuration and markdown files directly in the github.com interface. Navigate to a file (like [this one](https://github.com/academicpages/academicpages.github.io/blob/master/_talks/2012-03-01-talk-1.md) and click the pencil icon in the top right of the content preview (to the right of the "Raw | Blame | History" buttons). You can delete a file by clicking the trashcan icon to the right of the pencil icon. You can also create new files or upload files by navigating to a directory and clicking the "Create new file" or "Upload files" buttons. 

Example: editing a markdown file for a talk
![Editing a markdown file for a talk](/images/editing-talk.png)

For more info
------
More info about configuring Academic Pages can be found in [the guide](https://academicpages.github.io/markdown/), the [growing wiki](https://github.com/academicpages/academicpages.github.io/wiki), and you can always [ask a question on GitHub](https://github.com/academicpages/academicpages.github.io/discussions). The [guides for the Minimal Mistakes theme](https://mmistakes.github.io/minimal-mistakes/docs/configuration/) (which this theme was forked from) might also be helpful.
