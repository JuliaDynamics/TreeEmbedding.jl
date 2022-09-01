import Pkg 
Pkg.activate("../TreeEmbeddingTest/")

begin
    using TreeEmbedding
    using DynamicalSystems
    using DelayEmbeddings
    using RecurrenceAnalysis
    using Statistics
end

begin # generate data 
    N = 8 # number of oscillators
    dt = 0.1 # sampling time
    total = 5000  # time series length
    u0 = [0.590; 0.766; 0.566; 0.460; 0.794; 0.854; 0.200; 0.298] # initial conditions 
    lo96 = Systems.lorenz96(N, u0; F = 3.8)
    data = trajectory(lo96, total*dt;  Δt = dt, Ttr = 2500*dt)
end 

begin # reference RQA 
    ε = 0.05  # recurrence threshold
    lmin = 2   # minimum line length for RQA

    # theiler window estimation
    w1 = DelayEmbeddings.estimate_delay(data[:,2], "mi_min") # this is the time series we take as an example in the univariate case 
    theiler = w1 

    R_ref = RecurrenceMatrix(data, ε; fixedrate = true)
    RQA_ref = rqa(R_ref; theiler = theiler, lmin = lmin)
end 

begin # TDE RQA 
    dmax = 10   # maximum dimension for traditional tde
    
    data_sample = data[:,2] # has to be univariate 
    𝒟, τ_tde2, E = optimal_traditional_de(data_sample, "fnn"; dmax = dmax)

    R_tde = RecurrenceMatrix(𝒟, ε; fixedrate = true)
    RQA_tde = rqa(R_tde; theiler = theiler, lmin = lmin)
    RQA_tde = hcat(RQA_tde...)
end 

begin # TreeEmbedding RQA
    trials = 1 # trials for MCDTS
    taus = 0:100 # possible delays
    
    # pick one time series
    t_idx = [2,4,7]
    data_sample = data[:,t_idx]

    optim = TreeEmbedding.FNNOptim()
    tree = mcdts_embedding(Dataset(data_sample), optim, theiler, taus, trials; max_depth = 20)
    best_node = TreeEmbedding.best_embedding(tree)

    𝒟_tree = genembed(data_sample, best_node.τs, best_node.ts)
    R_tree = RecurrenceMatrix(𝒟_tree, ε; fixedrate = true)
    RQA_tree = rqa(R_tree; theiler = theiler, lmin = lmin)
    RQA_tree = hcat(RQA_tree...)
end
