abstract type ConstitutiveLaw end

struct NeoHookean{T} <: ConstitutiveLaw
    μ::T
    κ::T
end
struct MooneyRivlin{T} <: ConstitutiveLaw
    C₁₀::T
    C₀₁::T
    κ::T
end
struct Yeoh2{T} <: ConstitutiveLaw
    C₁₀::T 
    C₂₀::T  
end
struct Yeoh3{T} <: ConstitutiveLaw
    C₁₀::T 
    C₂₀::T  
    C₃₀::T
end

function init_material(type_constructor, ξ::Dict{Symbol,<:Real})
    params = fieldnames(type_constructor)
    values = [ξ[param] for param in params]
    _T = promote_type(map(typeof, values)...)
    T = _T <: Integer ? Float64 : _T
    return type_constructor(map(T, values)...)
end

NeoHookean(ξ::Dict{Symbol,<:Real}) = init_material(NeoHookean, ξ)
MooneyRivlin(ξ::Dict{Symbol,<:Real}) = init_material(MooneyRivlin, ξ)
Yeoh2(ξ::Dict{Symbol,<:Real}) = init_material(Yeoh2, ξ)
Yeoh3(ξ::Dict{Symbol,<:Real}) = init_material(Yeoh3, ξ)

function Ψ(mp::M) where M <: ConstitutiveLaw
    return Ψ(nothing,mp)
end

function Ψ(C, mp::NeoHookean)
    if isnothing(C)
        @variables Ī₁, bulk
    else
        J = sqrt(det(C))
        bulk = (J-1)^2
        Ī₁ = tr(C)*J^(-2/3)
    end
    return mp.μ/2*(Ī₁-3) + mp.κ/2*bulk
end
function Ψ(C, mp::MooneyRivlin)
    if isnothing(C)
        @variables Ī₁, Ī₂, bulk
    else
        J = sqrt(det(C))
        bulk = (J-1)^2
        Ī₁ = tr(C)*J^(-2/3)
        Ī₂ = 0.5*(mp.Ī₁^2-tr(C*C)*J^(-4/3))
    end
    return mp.C₁₀*(Ī₁-3) + mp.C₀₁*(Ī₂-3) + mp.κ/2*bulk
end
function Ψ(C, mp::Yeoh2)
    if isnothing(C)
        @variables Ī₁, bulk
    else
        J = sqrt(det(C))
        bulk = (J-1)^2
        Ī₁ = tr(C)*J^(-2/3)
    end
    return C₁₀*(Ī₁-3) + C₂₀*(Ī₁-3)^2 + (κ/2)*bulk
end
function Ψ(C, mp::Yeoh3)
    if isnothing(C)
        @variables Ī₁, bulk
    else
        J = sqrt(det(C))
        bulk = (J-1)^2
        Ī₁ = tr(C)*J^(-2/3)
    end
    return C₁₀*(Ī₁-3) + C₂₀*(Ī₁-3)^2 + C₃₀*(Ī₁-3)^3 + (κ/2)*bulk
end

# TODO combine Ψ functions into one combined function??
# probably will do when all written in terms of F --> entails messing with matricies_and_vectors stuff
# note: useful for optionally calcuating without assigning Ī₂ = mp isa MooneyRivlin ? sum(substitute(λᵅ[i], Dict(α => -2)) for i in 1:3) : Ī₂