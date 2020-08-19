using Test, SafeTestsets

@safetestset "InpParser Tests" begin include("InpParser/parser.jl") end
@safetestset "MMA Tests" begin include("MMA/mma.jl") end
@safetestset "TopOptProblems Tests" begin
    include("TopOptProblems/problems.jl")
    #include("TopOptProblems/metadata.jl")
end
@safetestset "CSIMP Tests" begin include("csimp.jl") end
@safetestset "AugLag Tests" begin
    include("AugLag/auglag.jl")
    include("AugLag/compliance.jl")
end
@safetestset "Global Stress Tests" begin include("stress.jl") end
@safetestset "Makie Visualization" begin include("Visualization/makie.jl") end
