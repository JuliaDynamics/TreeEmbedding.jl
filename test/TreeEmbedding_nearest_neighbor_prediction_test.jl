
println("\nTesting TreeEmbedding prediction of Lorenz")
@time begin
@testset "TreeEmbedding prediction of Lorenz" begin

    T_steps = 50

    max_depth = 15
    x1 = data[1:end-T_steps,1]
    x2 = data[end-T_steps+1:end,1]
    y1 = data[1:end-T_steps,2]
    y2 = data[end-T_steps+1:end,2]

    #####

    # Test NN-forecasting methods independently from embedding optimizing
    Y_trial = embed(x1, 3, w)

    Tw_in = 1 #prediction horizon insample
    Tw_out = 5 # prediction horizon out-of-sample
    KNN = 3 # nearest neighbors for pred method
    samplesize = 1

    PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in)
    prediction_insample, ns, temp = TreeEmbedding.insample_prediction(PredMeth, Y_trial; samplesize, w)

    @test Y_trial[1] == [-10.261485829931198, -8.23625170009478, -6.5709009676592265]
    @test prediction_insample[1] == [-10.401107082410284, -7.803902271408204, -6.442797064840611]
    @test prediction_insample[100] == [-8.991069467388533, -6.258722439494736, -7.459626377255716]

    PredictionLoss = TreeEmbedding.PredictionLoss(1)
    costs_insample = TreeEmbedding.compute_costs_from_prediction(PredictionLoss, prediction_insample, Y_trial, PredMeth.Tw_in, ns)
    @test costs_insample == 0.37204693617062456
    
    PredictionLoss = TreeEmbedding.PredictionLoss(2)
    costs_insample = TreeEmbedding.compute_costs_from_prediction(PredictionLoss, prediction_insample, Y_trial, PredMeth.Tw_in, ns)
    @test costs_insample == 0.4634731978522113
    
    PredictionLoss = TreeEmbedding.PredictionLoss(3)
    costs_insample = TreeEmbedding.compute_costs_from_prediction(PredictionLoss, prediction_insample, Y_trial, PredMeth.Tw_in, ns)
    @test costs_insample == 0.013229431563550997
    
    PredictionLoss = TreeEmbedding.PredictionLoss(4)
    costs_insample = TreeEmbedding.compute_costs_from_prediction(PredictionLoss, prediction_insample, Y_trial, PredMeth.Tw_in, ns)
    @test costs_insample == 0.014020691329639021

    KNN = 7
    PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in)
    prediction_insample, ns, temp = TreeEmbedding.insample_prediction(PredMeth, Y_trial; samplesize, w)

    @test prediction_insample[1] == [-10.475824990237367, -7.862249129934621, -6.34814615223535]
    @test prediction_insample[100] == [-9.082393020151338, -6.278012992500267, -7.379658787516312]

    Tw_in = 4
    PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in)
    prediction_insample, ns, temp = TreeEmbedding.insample_prediction(PredMeth, Y_trial; samplesize, w)

    @test prediction_insample[1] == [ -9.917781189190842, -6.834366360783968, -6.81994342218101]
    @test prediction_insample[100] == [-7.705390632465702, -6.222726841729036, -8.624366124067231]

    Tw_in = 1
    PredMeth = TreeEmbedding.local_model("linear", KNN, Tw_out, Tw_in)
    prediction_insample, ns, temp = TreeEmbedding.insample_prediction(PredMeth, Y_trial; samplesize, w)

    @test prediction_insample[1] == [-10.250767157116556, -7.8655765196596015, -6.619399745711982]
    @test prediction_insample[100] == [-9.0588381328615, -6.215913402473714, -7.379275386507503]

    Tw_in = 3
    KNN = 3
    PredMeth = TreeEmbedding.local_model("linear", KNN, Tw_out, Tw_in)
    prediction_insample, ns, temp = TreeEmbedding.insample_prediction(PredMeth, Y_trial; samplesize, w)

    @test prediction_insample[1] == [ -9.959820444468269, -7.227110425809523, -6.887937241321643]
    @test prediction_insample[100] == [ -8.102474744015897, -6.121537468270782, -8.201839821259416]


    # out-of-sample:
    Tw_in = 1 #prediction horizon insample
    Tw_out = 1 # prediction horizon out-of-sample
    KNN = 3 # nearest neighbors for pred method
    samplesize = 1
    trialss = 1

    PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in, trialss)
    PredictionLoss = TreeEmbedding.PredictionLoss(1)

    Random.seed!(123)
    costs_out_of_sample = TreeEmbedding.out_of_sample_prediction(PredMeth, PredictionLoss, Y_trial; w)

    @test costs_out_of_sample == 0.6174185479597902

    PredictionLoss = TreeEmbedding.PredictionLoss(2)
    Random.seed!(123)
    costs_out_of_sample = TreeEmbedding.out_of_sample_prediction(PredMeth, PredictionLoss, Y_trial; w)

    @test costs_out_of_sample == 0.3319391558758847

    PredictionLoss = TreeEmbedding.PredictionLoss(3)
    Random.seed!(123)
    costs_out_of_sample = TreeEmbedding.out_of_sample_prediction(PredMeth, PredictionLoss, Y_trial; w)

    @test costs_out_of_sample == 0

    Tw_out = 3 # prediction horizon out-of-sample
    KNN = 8 # nearest neighbors for pred method
    samplesize = 1
    trialss = 5

    PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in, trialss)
    PredictionLoss = TreeEmbedding.PredictionLoss(1)

    Random.seed!(123)
    costs_out_of_sample = TreeEmbedding.out_of_sample_prediction(PredMeth, PredictionLoss, Y_trial; w)

    @test costs_out_of_sample == 0.5421776166018837

    PredMeth = TreeEmbedding.local_model("linear", KNN, Tw_out, Tw_in, trialss)
    PredictionLoss = TreeEmbedding.PredictionLoss(1)

    Random.seed!(123)
    costs_out_of_sample = TreeEmbedding.out_of_sample_prediction(PredMeth, PredictionLoss, Y_trial; w)

    @test costs_out_of_sample == 0.024738135630960313



    #####
    
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

    # tree = mcdts_embedding(x1, optmodel, w1, delays, runs; max_depth)
    # best_node = TreeEmbedding.best_embedding(tree)
    # τ_mcdts = best_node.τs
    # L_mcdts = L(best_node)
    # @test τ_mcdts == [0,2]
    # @test 0.0457 < L_mcdts < 0.0458

    # Random.seed!(1234)
    # optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType, 0, 1,), TreeEmbedding.Range_function())
    # tree = mcdts_embedding(x1, optmodel, w1, delays, runs; max_depth)
    # best_node = TreeEmbedding.best_embedding(tree)
    # τ_mcdts = best_node.τs
    # L_mcdts = L(best_node)
    # @test sort(τ_mcdts) == [0, 3, 5]
    # @test 0.0944 < L_mcdts < 0.0945

    # # Prediction range-function, linear predictor MSE
    # Random.seed!(1234)
    # Tw_in = 2 #prediction horizon insample
    # Tw_out = 2 # prediction horizon out-of-sample
    # KNN = 1 # nearest neighbors for pred method
    # error_weight_insample = 0.5
    # error_weight_oosample = 0.5

    # PredMeth = TreeEmbedding.local_model("linear", KNN, Tw_out, Tw_in)
    # PredLoss = TreeEmbedding.PredictionLoss(2)
    # PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    # optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType, 0, 1,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Range_function())

    # tree = mcdts_embedding(x1, optmodel, w1, delays, runs; max_depth)
    # best_node = TreeEmbedding.best_embedding(tree)
    # τ_mcdts2 = best_node.τs
    # L_mcdts2 = L(best_node)
    # @test sort(τ_mcdts2) == [0,1,2,3,4,5]
    # @test 3.46e-5 < L_mcdts2 < 3.47e-5

    # # Prediction range-function, zeroth predictor first-comp-KL
    # Random.seed!(1234)
    # KNN = 1 # nearest neighbors for pred method
    # num_trials = 10 # number of trials for out of sample prediction error
    # Tw_out = 5 # oos prediction horizon
    # Tw_in = 1 # insample prediction horizon
    # choose_func = (L)->(TreeEmbedding.minL(L))
    # error_weight_insample = 0
    # error_weight_oosample = 1

    # PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in, num_trials)
    # PredLoss = TreeEmbedding.PredictionLoss(3)
    # PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    # optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType, 0, 1,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Range_function())

    # tree = mcdts_embedding(x1, optmodel, w1, delays, runs; max_depth, choose_func)
    # best_node = TreeEmbedding.best_embedding(tree)
    # τ_mcdts3 = best_node.τs
    # L_mcdts3 = L(best_node)
    # @test sort(τ_mcdts3) == [0,2,3,4,5]
    # @test 0.02772 < L_mcdts3 < 0.02773

    # # Prediction continuity-function, linear predictor mean-KL
    # Random.seed!(1234)
    # delays = 0:50
    # KNN = 6 # nearest neighbors for pred method
    # num_trials = 5 # number of trials for out of sample prediction error
    # Tw_out = 5 # oos prediction horizon
    # Tw_in = 1 # insample prediction horizon
    # choose_func = (L)->(TreeEmbedding.minL(L))

    # PredMeth = TreeEmbedding.local_model("linear", KNN, Tw_out, Tw_in, num_trials)
    # PredLoss = TreeEmbedding.PredictionLoss(4)
    # PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    # optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType, 0, 1), TreeEmbedding.Continuity_function())

    # tree = mcdts_embedding(x1, optmodel, w1, delays, runs; max_depth, choose_func)
    # best_node = TreeEmbedding.best_embedding(tree)
    # τ_mcdts4 = best_node.τs
    # L_mcdts4 = L(best_node)
    # @test sort(τ_mcdts4) == [0, 7, 15, 20, 26]
    # @test 9.31e-5 < L_mcdts4 < 9.32e-5

    # # multivariate prediction continuity-function, zeroth predictor first-comp-MSE
    # data_sample = Dataset(hcat(x1,y1))

    # Random.seed!(1234)
    # Tw_in = 1 #prediction horizon insample
    # Tw_out = 5 # prediction horizon out-of-sample
    # num_trials = 5 # number of trials for out of sample prediction error
    # KNN = 5 # nearest neighbors for pred method
    # error_weight_insample = 1
    # error_weight_oosample = 0.3

    # PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in, num_trials)
    # PredLoss = TreeEmbedding.PredictionLoss(1)
    # PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    # optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType, 0, 1, [error_weight_insample; error_weight_oosample]), TreeEmbedding.Continuity_function())

    # tree = mcdts_embedding(data_sample, optmodel, w1, delays, runs; max_depth)
    # best_node = TreeEmbedding.best_embedding(tree)
    # τ_mcdts5 = best_node.τs
    # ts_mcdts5 = best_node.ts
    # L_mcdts5 = L(best_node)
    # @test τ_mcdts5 == [0, 4, 1, 2, 0]
    # @test ts_mcdts5 == [1, 2, 1, 2, 2]
    # @test 0.042< L_mcdts5 < 0.043


    # # multivariate prediction continuity-function, zeroth predictor first-comp-MSE, less fiducials
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
    # optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType,0,samplesize,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Continuity_function())

    # tree = mcdts_embedding(data_sample, optmodel, w1, delays, runs; max_depth)
    # best_node = TreeEmbedding.best_embedding(tree)
    # τ_mcdts5 = best_node.τs
    # ts_mcdts5 = best_node.ts
    # L_mcdts5 = L(best_node)
    # @test τ_mcdts5 == [0, 22, 29, 3, 8, 27]
    # @test ts_mcdts5 == [1, 2, 2, 2, 2, 2]
    # @test 0.637 < L_mcdts5 < 0.638

    # # Prediction Continuity-function, zeroth predictor mean-MSE
    # delays = 0:100
    # Random.seed!(1234)
    # Tw_in = 5 #prediction horizon insample
    # Tw_out = 1 # prediction horizon out-of-sample
    # KNN = 4 # nearest neighbors for pred method
    # error_weight_insample = 1
    # error_weight_oosample = 1

    # PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in)
    # PredLoss = TreeEmbedding.PredictionLoss(2)
    # PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    # optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType,0,1,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Continuity_function())

    # tree = mcdts_embedding(x1, optmodel, w1, delays, runs; max_depth)
    # best_node = TreeEmbedding.best_embedding(tree)
    # τ_mcdts = best_node.τs
    # L_mcdts = L(best_node)
    # @test τ_mcdts == [0, 9, 5]
    # @test 0.0924 < L_mcdts < 0.0925

    # # Prediction Continuity-function, zeroth predictor mean-MSE, large Tw
    # delays = 0:100
    # Random.seed!(1234)
    # Tw_in = 1 #prediction horizon insample
    # Tw_out = 20 # prediction horizon out-of-sample
    # KNN = 4 # nearest neighbors for pred method
    # error_weight_insample = 0
    # error_weight_oosample = 1

    # PredMeth = TreeEmbedding.local_model("zeroth", KNN, Tw_out, Tw_in)
    # PredLoss = TreeEmbedding.PredictionLoss(2)
    # PredType = TreeEmbedding.MCDTSpredictionType(PredLoss, PredMeth)
    # optmodel = TreeEmbedding.MCDTSOptimGoal(TreeEmbedding.Prediction_error(PredType,0,1,[error_weight_insample; error_weight_oosample]), TreeEmbedding.Continuity_function())

    # tree = mcdts_embedding(x1, optmodel, w1, delays, runs; max_depth)
    # best_node = TreeEmbedding.best_embedding(tree)
    # τ_mcdts = best_node.τs
    # L_mcdts = L(best_node)
    # @test τ_mcdts == [0, 26, 19, 7]
    # @test 0.1372 < L_mcdts < 0.1373

end
end

true