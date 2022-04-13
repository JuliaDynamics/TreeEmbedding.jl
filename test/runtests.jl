using TreeEmbedding
using Test, DelimitedFiles
using Random
using DynamicalSystemsBase
using DelayEmbeddings
import TreeEmbedding.L
import Downloads

# Check Lorenz System
# Download some test timeseries
tsfolder = joinpath(@__DIR__, "timeseries")
todownload = ["test_time_series_Lorenz_N_875_multivariate.csv"]
repo = "https://raw.githubusercontent.com/JuliaDynamics/JuliaDynamics/master/timeseries"
mkpath(tsfolder)
for a in todownload
    Downloads.download(repo*"/"*a, joinpath(tsfolder, a))
end
data = Dataset(readdlm(joinpath(tsfolder, "test_time_series_Lorenz_N_875_multivariate.csv")))

w1 = DelayEmbeddings.estimate_delay(data[:,1],"mi_min")
w2 = DelayEmbeddings.estimate_delay(data[:,2],"mi_min")
w3 = DelayEmbeddings.estimate_delay(data[:,3],"mi_min")

w = maximum(hcat(w1,w2,w3))

@testset "TreeEmbedding embedding tests" begin

    #include("TreeEmbedding_attractor_reconstruction_tests.jl")
    #include("TreeEmbedding_convergent_cross_mapping_tests.jl")
    include("TreeEmbedding_nearest_neighbor_prediction_test.jl")

end
