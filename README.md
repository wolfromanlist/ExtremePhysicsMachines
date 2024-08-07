# Goal
Make a physics-informed ELM using domain decomposition for the Helmholtz equation using jax.

The Helmholtz equation is:
```math
\Delta u(x) + ku(x) = f(x) \quad  \forall x \in \Omega \newline
u(x) = u_\mathrm{b}(x) \quad \forall x \in \Omega
```

## Outline

Have an ELM for each subdomain, \
calculate a beta in each subdomain, \
calculate a (local/global?) residual and loss using the prediction from the sub-ELMs \
(later for Navier-Stokes: wrap the whole process in a LM optimization loop which iteratively optimizes the beta to minimize the loss/residuals)


### Detailed Plan

Domain Decomposition: 
- Choose a domain (x,y,t)
- For starters, place two RBFs in the domain, without any trainability
    - choose locations
    - make lambdas (exponential function, radius, center) covering the domain
- Initialize the sub-ELM for each lambda
    - use ELM() class that you already made
    - ...
