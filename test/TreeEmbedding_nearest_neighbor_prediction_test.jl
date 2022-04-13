delays = 0:50
runs = 10
runs2 = 10
T_steps = 50

println("\nTesting TreeEmbedding prediction of Lorenz")
@time begin
@testset "TreeEmbedding prediction of Lorenz" begin

    # max_depth = 15
    # x1 = data[1:end-T_steps,1]
    # x2 = data[end-T_steps+1:end,1]
    # y1 = data[1:end-T_steps,2]
    # y2 = data[end-T_steps+1:end,2]
    
    # # Prediction range-function, zeroth predictor first comp-MSE
    # delays = 0:5
    # runs = 10

    # Random.seed!(1234)
    # Tw_in = 1 #prediction horizon insample
    # Tw_out = 5 # prediction horizon out-of-sample
    # KNN = 3 # nearest neighbors for pred method
    # error_weight_insample = 1
    # error_weight_oosample = 0 

    # PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in)
    # PredLoss = TreeEmbedding.PredictionLoss(1)
    # PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    # optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType, 0, 1,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Range_function())

    # tree = mcdts_embedding(Dataset(x1), optmodel, w1, delays, runs; max_depth)
    # best_node = TreeEmbedding.best_embedding(tree)
    # τ_mcdts = best_node.τs
    # L_mcdts = L(best_node)
    # @test length(τ_mcdts) == 2
    # @test τ_mcdts[2] == 2
    # @test 0.045 < L_mcdts < 0.046

    # error_weight_insample = 0.5
    # error_weight_oosample = 0.5
    # optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType, 0, 1,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Range_function())
    # tree = mcdts_embedding(Dataset(x1), optmodel, w1, delays, runs; max_depth)
    # best_node = TreeEmbedding.best_embedding(tree)
    # τ_mcdts = best_node.τs
    # L_mcdts = L(best_node)
    # @test length(τ_mcdts) == 4
    # @test sort(τ_mcdts) == [0, 1, 3, 5]
    # @test 0.043 < L_mcdts < 0.045

    # # Prediction range-function, linear predictor first comp-MSE
    # delays = 0:5
    # runs = 5
    # Random.seed!(1234)
    # Tw_in = 2 #prediction horizon insample
    # Tw_out = 2 # prediction horizon out-of-sample
    # KNN = 1 # nearest neighbors for pred method
    # error_weight_insample = 0.5
    # error_weight_oosample = 0.5

    # PredMeth = TreeEmbedding.local_model("linear", KNN, Tw_out, Tw_in)
    # PredLoss = TreeEmbedding.PredictionLoss(1)
    # PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    # optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType, 0, 1,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Range_function())

    # tree = mcdts_embedding(Dataset(x1), optmodel, w1, delays, runs; max_depth)
    # best_node = TreeEmbedding.best_embedding(tree)
    # τ_mcdts2 = best_node.τs
    # L_mcdts2 = L(best_node)
    # @test sort(τ_mcdts2) == [0,1,2,3,4,5]
    # @test 0.0002 < L_mcdts2 < 0.0003

    # # Prediction range-function, zeroth predictor mean-KL
    # delays = 0:10
    # runs = 1

    # Random.seed!(1234)
    # KNN = 3 # nearest neighbors for pred method
    # num_trials = 10 # number of trials for out of sample prediction error
    # Tw_out = 5 # oos prediction horizon
    # Tw_in = 1 # insample prediction horizon
    # choose_func = (L)->(TreeEmbedding.minL(L))
    # error_weight_insample = 1
    # error_weight_oosample = 0
    
    # PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in, num_trials)
    # PredLoss = TreeEmbedding.PredictionLoss(4)
    # PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    # optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType, 0, 1,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Range_function())
    
    # tree = mcdts_embedding(x1, optmodel, w1, delays, runs; max_depth, choose_func)
    # best_node = TreeEmbedding.best_embedding(tree)
    # τ_mcdts3 = best_node.τs
    # L_mcdts3 = L(best_node)
    # @test sort(τ_mcdts3) == [0,3]
    # @test 0.0016 < L_mcdts3 < 0.0017

    # # Prediction range-function, linear predictor first-comp-KL
    # delays = 0:5
    # runs = 1
    # Random.seed!(1234)
    # KNN = 3 # nearest neighbors for pred method
    # num_trials = 10 # number of trials for out of sample prediction error
    # Tw_out = 5 # oos prediction horizon
    # Tw_in = 1 # insample prediction horizon
    # choose_func = (L)->(TreeEmbedding.minL(L))
    # error_weight_insample = 1
    # error_weight_oosample = 0
    
    # PredMeth = TreeEmbedding.local_model("linear", KNN, Tw_out, Tw_in, num_trials)
    # PredLoss = TreeEmbedding.PredictionLoss(3)
    # PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    # optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType, 0, 1,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Range_function())
    
    # tree = mcdts_embedding(x1, optmodel, w1, delays, runs; max_depth, choose_func)
    # best_node = TreeEmbedding.best_embedding(tree)
    # τ_mcdts4 = best_node.τs
    # L_mcdts4 = L(best_node)
    # @test sort(τ_mcdts4) == [0, 3]
    # @test 1.11e-5 < L_mcdts4 < 1.12e-5

    # # multivariate prediction range-function, zeroth predictor first-comp-MSE
    # delays = 0:5
    # runs = 5
    # data_sample = Dataset(hcat(x1,y1))

    # Random.seed!(1234)
    # Tw_in = 1 #prediction horizon insample
    # Tw_out = 1 # prediction horizon out-of-sample
    # KNN = 3 # nearest neighbors for pred method
    # error_weight_insample = 1
    # error_weight_oosample = 0

    # PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in)
    # PredLoss = TreeEmbedding.PredictionLoss(1)
    # PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    # optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType, 0, 1, [error_weight_insample; error_weight_oosample]), TreeEmbedding.Range_function())

    # tree = mcdts_embedding(data_sample, optmodel, w1, delays, runs; max_depth)
    # best_node = TreeEmbedding.best_embedding(tree)
    # τ_mcdts5 = best_node.τs
    # ts_mcdts5 = best_node.ts
    # L_mcdts5 = L(best_node)
    # @test τ_mcdts5 == [0, 2]
    # @test ts_mcdts5 == [1, 2]
    # @test 0.027 < L_mcdts5 < 0.028


    # # multivariate prediction range-function, zeroth predictor first-comp-MSE, less fiducials

    # Random.seed!(1234)
    # Tw_in = 1 #prediction horizon insample
    # Tw_out = 1 # prediction horizon out-of-sample
    # KNN = 3 # nearest neighbors for pred method
    # error_weight_insample = 1
    # error_weight_oosample = 0
    # samplesize = 0.5

    # PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in)
    # PredLoss = TreeEmbedding.PredictionLoss(1)
    # PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    # optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType,0,samplesize,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Range_function())

    # tree = mcdts_embedding(data_sample, optmodel, w1, delays, runs; max_depth)
    # best_node = TreeEmbedding.best_embedding(tree)
    # τ_mcdts5 = best_node.τs
    # ts_mcdts5 = best_node.ts
    # L_mcdts5 = L(best_node)
    # @test τ_mcdts5 == [0, 4, 2]
    # @test ts_mcdts5 == [1, 1, 1]
    # @test 0.616 < L_mcdts5 < 0.617

    # # Prediction Continuity-function, zeroth predictor first comp-MSE
    # delays = 0:100
    # runs = 10
    # Random.seed!(1234)
    # Tw_in = 1 #prediction horizon insample
    # Tw_out = 1 # prediction horizon out-of-sample
    # KNN = 3 # nearest neighbors for pred method
    # error_weight_insample = 1
    # error_weight_oosample = 0

    # PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in)
    # PredLoss = TreeEmbedding.PredictionLoss(1)
    # PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    # optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType,0,1,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Continuity_function())

    # tree = mcdts_embedding(Dataset(x1), optmodel, w1, delays, runs; max_depth)
    # best_node = TreeEmbedding.best_embedding(tree)
    # τ_mcdts = best_node.τs
    # L_mcdts = L(best_node)
    # @test τ_mcdts == [0, 9]
    # @test 0.05 < L_mcdts < 0.06

    # # Prediction Continuity-function, zeroth predictor first comp-MSE, more neighbors
    # Random.seed!(1234)
    # Tw_in = 1 #prediction horizon insample
    # Tw_out = 1 # prediction horizon out-of-sample
    # KNN = 8 # nearest neighbors for pred method
    # error_weight_insample = 1
    # error_weight_oosample = 0

    # PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in)
    # PredLoss = TreeEmbedding.PredictionLoss(1)
    # PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    # optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType,0,1,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Continuity_function())

    # tree = mcdts_embedding(Dataset(x1), optmodel, w1, delays, runs; max_depth)
    # best_node2 = TreeEmbedding.best_embedding(tree)
    # τ_mcdts2 = best_node2.τs
    # L_mcdts2 = L(best_node2)
    # @test τ_mcdts2 == τ_mcdts
    # @test 0.065 < L_mcdts2 < 0.066

    # Prediction Range-function, zeroth predictor first comp-MSE, Tw = 30
    delays = 0:5
    runs = 1
    Random.seed!(1234)
    num_trials = 10 # number of trials for out of sample prediction error
    Tw_out = 5 # oos prediction horizon
    Tw_in = 30 # insample prediction horizon
    KNN = 3 # nearest neighbors for pred method
    error_weight_insample = 1
    error_weight_oosample = 0
    choose_func = (L)->(TreeEmbedding.minL(L))
    
    PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in, num_trials)
    PredLoss = TreeEmbedding.PredictionLoss(1)
    PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType,0,1,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Range_function())
    
    tree = mcdts_embedding(x1, optmodel, w1, delays, runs; max_depth, choose_func, verbose=true)
    best_node = TreeEmbedding.best_embedding(tree)
    τ_mcdts = best_node.τs
    L_mcdts = L(best_node)
    @test sort(τ_mcdts) == [0, 1, 4]
    @test 0.99 < L_mcdts < 1.0

    # # Prediction Continuity-function, zeroth predictor first all-comp-MSE, Tw = 50, more neighbors
    # delays = 0:80
    # runs = 1
    # Random.seed!(1234)
    # KNN = 6 # nearest neighbors for pred method
    # num_trials = 5 # number of trials for out of sample prediction error
    # Tw_out = 5 # oos prediction horizon
    # Tw_in = 1 # insample prediction horizon
    # choose_func = (L)->(TreeEmbedding.minL(L))
    
    # PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in, num_trials)
    # PredLoss = TreeEmbedding.PredictionLoss(2)
    # PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    # optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType), TreeEmbedding.Continuity_function())
    
    # tree = mcdts_embedding(x1, optmodel, w1, delays, runs; max_depth, choose_func)
    # best_node = TreeEmbedding.best_embedding(tree)
    # τ_mcdts = best_node.τs
    # L_mcdts = L(best_node)
    # @test sort(τ_mcdts) == [0, 5, 9]
    # @test 0.06 < L_mcdts < 0.1

    # # Prediction Continuity-function, linear predictor first all-comp-MSE, Tw = 50, more neighbors
    # delays = 0:80
    # runs = 10
    # Random.seed!(1234)
    # Tw_in = 5 #prediction horizon insample
    # Tw_out = 1 # prediction horizon out-of-sample
    # KNN = 6 # nearest neighbors for pred method
    # error_weight_insample = 1
    # error_weight_oosample = 0

    # PredMeth = TreeEmbedding.local_model("linear", KNN, Tw_out, Tw_in)
    # PredLoss = TreeEmbedding.PredictionLoss(2)
    # PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    # optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType,0,1,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Continuity_function())

    # tree = mcdts_embedding(Dataset(x1), optmodel, w1, delays, runs; max_depth)
    # best_node = TreeEmbedding.best_embedding(tree)
    # τ_mcdts = best_node.τs
    # L_mcdts = L(best_node)
    # @test τ_mcdts == [0, 9, 5]
    # @test 0.033 < L_mcdts < 0.034

    # # Prediction Continuity-function, linear predictor first all-comp-MSE, Tw = 50, more neighbors, less fiducials
    # delays = 0:80
    # runs = 10
    # Random.seed!(1234)
    # Tw_in = 5 #prediction horizon insample
    # Tw_out = 1 # prediction horizon out-of-sample
    # KNN = 6 # nearest neighbors for pred method
    # error_weight_insample = 1
    # error_weight_oosample = 0
    # samplesize = 0.5

    # PredMeth = TreeEmbedding.local_model("linear", KNN, Tw_out, Tw_in)
    # PredLoss = TreeEmbedding.PredictionLoss(2)
    # PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    # optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType,0,samplesize,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Continuity_function())

    # tree = mcdts_embedding(Dataset(x1), optmodel, w1, delays, runs; max_depth)
    # best_node = TreeEmbedding.best_embedding(tree)
    # τ_mcdts = best_node.τs
    # L_mcdts = L(best_node)
    # @test τ_mcdts == [0, 9, 5]
    # @test 0.164 < L_mcdts < 0.165

end
end

true