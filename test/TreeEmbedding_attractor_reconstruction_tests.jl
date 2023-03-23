
delays = 0:50
runs = 10
runs2 = 10
T_steps = 50

println("\nTesting TreeEmbedding L & FNN on Lorenz univariate")
@time begin
@testset "TreeEmbedding L & FNN on Lorenz univariate" begin

    # L
    Random.seed!(1234)
    pecuzal = TreeEmbedding.PecuzalOptim()
    tree = mcdts_embedding(data[:,1], pecuzal, w1, delays, runs)
    best_node1 = TreeEmbedding.best_embedding(tree)
    @test best_node1.τs == [0, 9, 42, 20]
    @test L(best_node1) < -0.9

    # L with regularization
    reg_cost = TreeEmbedding.RegularizedCost(TreeEmbedding.LStatistic(), 1/5)
    pecuzal_reg = TreeEmbedding.MCDTSOptimGoal(reg_cost, TreeEmbedding.ContinuityFunction())
    tree = mcdts_embedding(data[:,1], pecuzal_reg, w1, delays, runs)
    best_node = TreeEmbedding.best_embedding(tree)
    @test length(best_node.τs) < length(best_node1.τs)
    @test L(best_node1) < L(best_node)

    # L with tws
    Random.seed!(1234)
    tws = 2:4:delays[end]
    optmodel2 = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.LStatistic(0,3,tws), TreeEmbedding.ContinuityFunction())
    tree2 = mcdts_embedding(data[:,1], optmodel2, w1, delays, runs)
    best_node2 = TreeEmbedding.best_embedding(tree2)
    @test best_node2.τs == best_node1.τs
    @test L(best_node2) > L(best_node1)

    # L with tws and less fiducials for computation
    Random.seed!(1234)
    tws = 2:2:delays[end]
    samplesize = 0.5
    optmodel3 = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.LStatistic(0,3,tws,samplesize), TreeEmbedding.ContinuityFunction())
    tree3 = mcdts_embedding(data[:,1], optmodel3, w1, delays, runs)
    best_node3 = TreeEmbedding.best_embedding(tree3)
    @test best_node3.τs == best_node1.τs
    @test L(best_node1) < L(best_node3)

    # L with tws and threshold
    Random.seed!(1234)
    optmodel4 = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.LStatistic(-0.5,3,tws), TreeEmbedding.ContinuityFunction())
    tree4 = mcdts_embedding(data[:,1], optmodel4, w1, delays, runs)
    best_node4 = TreeEmbedding.best_embedding(tree4)
    @test length(best_node4.τs) < length(best_node2.τs)
    @test best_node4.τs == [0, 9]
    @test L(best_node4) > L(best_node2)

    # FNN with threshold
    Random.seed!(1234)
    threshold = 0.05
    optmodel4 = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.FNNStatistic(threshold), TreeEmbedding.ContinuityFunction())
    tree = mcdts_embedding(data[:,1], optmodel4, w1, delays, runs)
    best_node = TreeEmbedding.best_embedding(tree)
    @test best_node.τs == [0, 27, 21, 6]
    @test L(best_node) > threshold

    # FNN with threshold and regularization
    Random.seed!(1234)
    reg_cost = TreeEmbedding.RegularizedCost(TreeEmbedding.FNNStatistic(threshold), 1/250)
    optmodel4_reg = TreeEmbedding.MCDTSOptimGoal(reg_cost, TreeEmbedding.ContinuityFunction())
    tree = mcdts_embedding(data[:,1], optmodel4_reg, w1, delays, runs)
    best_node2 = TreeEmbedding.best_embedding(tree)
    @test length(best_node2.τs) < length(best_node.τs)
    @test L(best_node2) > threshold

    # FNN regularization with 0 strength
    Random.seed!(1234)
    reg_cost = TreeEmbedding.RegularizedCost(TreeEmbedding.FNNStatistic(threshold), 0)
    optmodel4_reg = TreeEmbedding.MCDTSOptimGoal(reg_cost, TreeEmbedding.ContinuityFunction())
    tree = mcdts_embedding(data[:,1], optmodel4_reg, w1, delays, runs)
    best_node3 = TreeEmbedding.best_embedding(tree)
    @test best_node3.τs == best_node3.τs
    @test L(best_node3) == L(best_node)

    # FNN regularization overregulated
    Random.seed!(1234)
    reg_cost = TreeEmbedding.RegularizedCost(TreeEmbedding.FNNStatistic(threshold), 1/2)
    optmodel4_reg = TreeEmbedding.MCDTSOptimGoal(reg_cost, TreeEmbedding.ContinuityFunction())
    tree = mcdts_embedding(data[:,1], optmodel4_reg, w1, delays, runs)
    best_node4 = TreeEmbedding.best_embedding(tree)
    @test best_node4.τs == [0]
    @test L(best_node4) == 1

    # FNN with threshold and less fid-points
    Random.seed!(1234)
    optmodel4 = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.FNNStatistic(threshold,2,0.8), TreeEmbedding.ContinuityFunction())
    tree = mcdts_embedding(data[:,1], optmodel4, w1, delays, runs)
    best_node_less = TreeEmbedding.best_embedding(tree)
    @test best_node_less.τs == best_node.τs
    @test L(best_node_less) > threshold

    L_YY = TreeEmbedding.compute_delta_L(data[:,1], best_node.τs, delays[end];  w = w1)
    L_YY2 = TreeEmbedding.compute_delta_L(data[:,1], best_node_less.τs, delays[end];  w = w1)
    @test L_YY == L_YY2

end


println("\nTesting TreeEmbedding L & FNN on Lorenz multivariate")
@testset "TreeEmbedding L & FNN on Lorenz multivariate" begin

    Random.seed!(1234)
    tws = 2:4:delays[end]
    optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.LStatistic(-0.05,3,tws), TreeEmbedding.ContinuityFunction())
    tree = mcdts_embedding(data, optmodel, w, delays, runs2)
    best_node = TreeEmbedding.best_embedding(tree)
    
    Random.seed!(1234)
    optmodel2 = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.FNNStatistic(0.05), TreeEmbedding.ContinuityFunction())
    tree2 = mcdts_embedding(data, optmodel2, w, delays, runs2)
    best_node2 = TreeEmbedding.best_embedding(tree2)
    L_YY = TreeEmbedding.compute_delta_L(Dataset(data), best_node2.τs, best_node2.ts, delays[end];  w = w, tws = 2:4:delays[end])
    
    @test best_node.τs == [0, 9, 42, 20]
    @test best_node.ts == [1, 1, 1, 1]
    @test best_node2.τs == [0, 22, 16, 7]
    @test best_node2.ts == [1, 2, 2, 1]
    @test L(best_node) < -0.9
    @test L(best_node) < L_YY
    
    # less fid points
    Random.seed!(1234)
    optmodel2 = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.FNNStatistic(0.05,2,0.2), TreeEmbedding.ContinuityFunction())
    tree3 = mcdts_embedding(data, optmodel2, w, delays, runs2)
    best_node3 = TreeEmbedding.best_embedding(tree3)
    
    @test best_node3.τs == [0, 48, 39, 0, 2]
    @test best_node3.ts == [1, 3, 2, 3, 2]
    
    # less fid points regularized
    Random.seed!(1234)
    reg_cost = TreeEmbedding.RegularizedCost(TreeEmbedding.FNNStatistic(0.05,2,0.2), 1/25)
    optmodel3 = TreeEmbedding.MCDTSOptimGoal(reg_cost, TreeEmbedding.ContinuityFunction())
    tree4 = mcdts_embedding(data, optmodel3, w, delays, runs2)
    best_node4 = TreeEmbedding.best_embedding(tree4)
    
    @test best_node4.τs == [0, 14, 24, 19]
    @test best_node4.ts == [1, 3, 3, 3]
    @test L(best_node4) > L(best_node3)

end
end
true