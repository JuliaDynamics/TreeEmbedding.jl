
println("\nTesting TreeEmbedding prediction of Lorenz")
@time begin
@testset "TreeEmbedding prediction of Lorenz" begin

    T_steps = 50

    max_depth = 15
    x1 = data[1:end-T_steps,1]
    x2 = data[end-T_steps+1:end,1]
    y1 = data[1:end-T_steps,2]
    y2 = data[end-T_steps+1:end,2]
    
    # Prediction range-function, zeroth predictor first comp-MSE
    delays = 0:5
    runs = 10

    Random.seed!(1234)
    Tw_in = 1 #prediction horizon insample
    Tw_out = 5 # prediction horizon out-of-sample
    KNN = 3 # nearest neighbors for pred method
    error_weight_insample = 1
    error_weight_oosample = 0 

    PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in)
    PredLoss = TreeEmbedding.PredictionLoss(1)
    PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType, 0, 1,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Range_function())

    tree = mcdts_embedding(x1, optmodel, w1, delays, runs; max_depth)
    best_node = TreeEmbedding.best_embedding(tree)
    τ_mcdts = best_node.τs
    L_mcdts = L(best_node)
    @test τ_mcdts == [0,2]
    @test 0.0457 < L_mcdts < 0.0458

    Random.seed!(1234)
    optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType, 0, 1,), TreeEmbedding.Range_function())
    tree = mcdts_embedding(x1, optmodel, w1, delays, runs; max_depth)
    best_node = TreeEmbedding.best_embedding(tree)
    τ_mcdts = best_node.τs
    L_mcdts = L(best_node)
    @test sort(τ_mcdts) == [0, 3, 5]
    @test 0.0944 < L_mcdts < 0.0945

    # Prediction range-function, linear predictor MSE
    Random.seed!(1234)
    Tw_in = 2 #prediction horizon insample
    Tw_out = 2 # prediction horizon out-of-sample
    KNN = 1 # nearest neighbors for pred method
    error_weight_insample = 0.5
    error_weight_oosample = 0.5

    PredMeth = TreeEmbedding.local_model("linear", KNN, Tw_out, Tw_in)
    PredLoss = TreeEmbedding.PredictionLoss(2)
    PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType, 0, 1,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Range_function())

    tree = mcdts_embedding(x1, optmodel, w1, delays, runs; max_depth)
    best_node = TreeEmbedding.best_embedding(tree)
    τ_mcdts2 = best_node.τs
    L_mcdts2 = L(best_node)
    @test sort(τ_mcdts2) == [0,1,2,3,4,5]
    @test 3.46e-5 < L_mcdts2 < 3.47e-5

    # Prediction range-function, zeroth predictor first-comp-KL
    Random.seed!(1234)
    KNN = 1 # nearest neighbors for pred method
    num_trials = 10 # number of trials for out of sample prediction error
    Tw_out = 5 # oos prediction horizon
    Tw_in = 1 # insample prediction horizon
    choose_func = (L)->(TreeEmbedding.minL(L))
    error_weight_insample = 0
    error_weight_oosample = 1

    PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in, num_trials)
    PredLoss = TreeEmbedding.PredictionLoss(3)
    PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType, 0, 1,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Range_function())

    tree = mcdts_embedding(x1, optmodel, w1, delays, runs; max_depth, choose_func)
    best_node = TreeEmbedding.best_embedding(tree)
    τ_mcdts3 = best_node.τs
    L_mcdts3 = L(best_node)
    @test sort(τ_mcdts3) == [0,2,3,4,5]
    @test 0.02772 < L_mcdts3 < 0.02773

    # Prediction continuity-function, linear predictor mean-KL
    Random.seed!(1234)
    delays = 0:50
    KNN = 6 # nearest neighbors for pred method
    num_trials = 5 # number of trials for out of sample prediction error
    Tw_out = 5 # oos prediction horizon
    Tw_in = 1 # insample prediction horizon
    choose_func = (L)->(TreeEmbedding.minL(L))

    PredMeth = TreeEmbedding.local_model("linear", KNN, Tw_out, Tw_in, num_trials)
    PredLoss = TreeEmbedding.PredictionLoss(4)
    PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType, 0, 1), TreeEmbedding.Continuity_function())

    tree = mcdts_embedding(x1, optmodel, w1, delays, runs; max_depth, choose_func)
    best_node = TreeEmbedding.best_embedding(tree)
    τ_mcdts4 = best_node.τs
    L_mcdts4 = L(best_node)
    @test sort(τ_mcdts4) == [0, 7, 15, 20, 26]
    @test 9.31e-5 < L_mcdts4 < 9.32e-5

    # multivariate prediction continuity-function, zeroth predictor first-comp-MSE
    data_sample = Dataset(hcat(x1,y1))

    Random.seed!(1234)
    Tw_in = 1 #prediction horizon insample
    Tw_out = 5 # prediction horizon out-of-sample
    num_trials = 5 # number of trials for out of sample prediction error
    KNN = 5 # nearest neighbors for pred method
    error_weight_insample = 1
    error_weight_oosample = 0.3

    PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in, num_trials)
    PredLoss = TreeEmbedding.PredictionLoss(1)
    PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType, 0, 1, [error_weight_insample; error_weight_oosample]), TreeEmbedding.Continuity_function())

    tree = mcdts_embedding(data_sample, optmodel, w1, delays, runs; max_depth)
    best_node = TreeEmbedding.best_embedding(tree)
    τ_mcdts5 = best_node.τs
    ts_mcdts5 = best_node.ts
    L_mcdts5 = L(best_node)
    @test τ_mcdts5 == [0, 4, 1, 2, 0]
    @test ts_mcdts5 == [1, 2, 1, 2, 2]
    @test 0.042< L_mcdts5 < 0.043


    # multivariate prediction range-function, zeroth predictor first-comp-MSE, less fiducials
    Random.seed!(1234)
    Tw_in = 1 #prediction horizon insample
    Tw_out = 1 # prediction horizon out-of-sample
    KNN = 3 # nearest neighbors for pred method
    error_weight_insample = 1
    error_weight_oosample = 0
    samplesize = 0.5

    PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in)
    PredLoss = TreeEmbedding.PredictionLoss(1)
    PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType,0,samplesize,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Continuity_function())

    tree = mcdts_embedding(data_sample, optmodel, w1, delays, runs; max_depth)
    best_node = TreeEmbedding.best_embedding(tree)
    τ_mcdts5 = best_node.τs
    ts_mcdts5 = best_node.ts
    L_mcdts5 = L(best_node)
    @test τ_mcdts5 == [0, 22, 29, 3, 8, 27]
    @test ts_mcdts5 == [1, 2, 2, 2, 2, 2]
    @test 0.637 < L_mcdts5 < 0.638

    # Prediction Continuity-function, zeroth predictor mean-MSE
    delays = 0:100
    Random.seed!(1234)
    Tw_in = 5 #prediction horizon insample
    Tw_out = 1 # prediction horizon out-of-sample
    KNN = 4 # nearest neighbors for pred method
    error_weight_insample = 1
    error_weight_oosample = 1

    PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in)
    PredLoss = TreeEmbedding.PredictionLoss(2)
    PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType,0,1,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Continuity_function())

    tree = mcdts_embedding(x1, optmodel, w1, delays, runs; max_depth)
    best_node = TreeEmbedding.best_embedding(tree)
    τ_mcdts = best_node.τs
    L_mcdts = L(best_node)
    @test τ_mcdts == [0, 9, 5]
    @test 0.0924 < L_mcdts < 0.0925

    # Prediction Continuity-function, zeroth predictor mean-MSE, large Tw
    delays = 0:100
    Random.seed!(1234)
    Tw_in = 1 #prediction horizon insample
    Tw_out = 20 # prediction horizon out-of-sample
    KNN = 4 # nearest neighbors for pred method
    error_weight_insample = 0
    error_weight_oosample = 1

    PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in)
    PredLoss = TreeEmbedding.PredictionLoss(2)
    PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType,0,1,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Continuity_function())

    tree = mcdts_embedding(x1, optmodel, w1, delays, runs; max_depth)
    best_node = TreeEmbedding.best_embedding(tree)
    τ_mcdts = best_node.τs
    L_mcdts = L(best_node)
    @test τ_mcdts == [0, 26, 19, 7]
    @test 0.1372 < L_mcdts < 0.1373

end
end

true