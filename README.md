# TreeEmbedding.jl - a Monte Carlo Decision Tree Search (MCDTS) for optimal embedding

![DynamicalSystems.jl logo: The Double Pendulum](https://i.imgur.com/nFQFdB0.gif)

| **Documentation**   |  **Tests**     |  Gitter |
|:--------:|:-------------------:|:-----:|
|[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaDynamics.github.io/DynamicalSystems.jl/dev) | [![CI](https://github.com/juliadynamics/TreeEmbedding.jl/workflows/CI/badge.svg)](https://github.com/JuliaDynamics/TreeEmbedding.jl/actions)  | [![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/JuliaDynamics/Lobby)


`TreeEmbedding.jl` implements the MCDTS algorithm outlined in [Kraemer, K.H., Gelbrecht, M. et al. 2022](https://link.springer.com/article/10.1007/s11071-022-07280-2) and is one of the packages composing **DynamicalSystems.jl**. It aims to provide an optimal time delay state space reconstruction from time series data with the help of decisions trees and suitable statistics that guide the decisions done during the rollout of these trees. For all details of the algorithm the reader is referred to the accompanying paper. Here we provide an implementation of all the variants described in the paper. 

For more details please see our detailed [documentation pages](https://juliadynamics.github.io/DynamicalSystems.jl/dev/contents/#TreeEmbedding).