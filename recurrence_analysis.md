# False-Nearest Neigbour based Embedding for Recurrence Analysis 

In the following, we will demonstrate how to perform an embedding of a 8-dimensional Lorenz 96 system (L96) for a recurrence analysis. We will determine recurrence quantification analysis (RQA) measures from the true, fully known system and then compare them to the RQA measures computed from an embedding that can only use the 2nd, 5th and 7th dimension of the L96. [^Kraemer2022] showed that a False-nearest neighbour approach works best in this case, so we will use TreeEmbedding with an FNN cost function and the continuity statistic to pre-select delay. For a comprehensive and more systematic evaluation of the different embedding algorithms for RQA, see [^Kraemer2022].

First we generate the data from L96 and compute the RQA for the fully known system. 



Now, we compute a Theiler window and setup TreeEmbedding with an FNN cost function and the continuity statistic as a delay preselection.
# References 

[^Kraemer2022]: Kraemer, K.H., Gelbrecht, M., Pavithran, I., Sujith, R. I. and Marwan, N. (2022). [Optimal state space reconstruction via Monte Carlo decision tree search. Nonlinear Dynamics](https://doi.org/10.1007/s11071-022-07280-2).

