println("\nTesting TreeEmbedding CCM on univariate Logistic map data:")
@time begin
@testset "TreeEmbedding CCM on univariate Logistic map data" begin

    Lval = 3500
    x = zeros(Lval)
    y = zeros(Lval)
    r = 3.8
    r2 = 3.5
    βxy = 0.02
    βyx = 0.1
    x[1]=0.4
    y[1]=0.2

    for i = 2:Lval
        x[i]=x[i-1]*(r-r*x[i-1]-βxy*y[i-1])
        y[i]=y[i-1]*(r2-r2*y[i-1]-βyx*x[i-1])
    end

    w1 = DelayEmbeddings.estimate_delay(x, "mi_min")
    w2 = DelayEmbeddings.estimate_delay(y, "mi_min")

    test1 = DelayEmbeddings.standardize(x)
    test2 = DelayEmbeddings.standardize(y)
    # try TreeEmbedding with CCM
    taus1 = 0:10 # the possible delay vals
    trials = 20 # the sampling of the tree

    Random.seed!(1234)
    optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.CCM_ρ(test2), TreeEmbedding.RangeFunction())
    tree = mcdts_embedding(Dataset(test1), optmodel, w1, taus1, trials)
    best_node = TreeEmbedding.best_embedding(tree)
    τ_mcdts = best_node.τs
    ts_mcdts = best_node.ts
    Lval = L(best_node)

    # less fid points
    Random.seed!(1234)
    optmodel2 = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.CCM_ρ(test2,1,0.5), TreeEmbedding.RangeFunction())
    tree2 = mcdts_embedding(Dataset(test1), optmodel2, w1, taus1, trials)
    best_node2 = TreeEmbedding.best_embedding(tree2)
    τ_mcdts2 = best_node2.τs
    ts_mcdts2 = best_node2.ts
    L2 = L(best_node2)

    @test Lval < -0.9
    @test L2 < Lval
    @test length(τ_mcdts) == 3 == length(τ_mcdts2)
    @test sort(τ_mcdts) == sort(τ_mcdts2) == [0, 1, 2]

    Random.seed!(1234)
    optmodel2 = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.CCM_ρ(test1), TreeEmbedding.RangeFunction())
    tree2 = mcdts_embedding(Dataset(test2), optmodel2, w2, taus1, trials)
    best_node = TreeEmbedding.best_embedding(tree2)
    τ_mcdts = best_node.τs
    ts_mcdts = best_node.ts
    Lval2 = L(best_node)

    @test Lval2 < Lval
    @test length(τ_mcdts) ==  3
    @test sort(τ_mcdts) == [0, 1, 2]

end
end
true