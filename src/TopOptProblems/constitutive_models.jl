import InteractiveUtils: subtypes

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
    κ::T
end
struct Yeoh3{T} <: ConstitutiveLaw
    C₁₀::T
    C₂₀::T
    C₃₀::T
    κ::T
end
struct ArrudaBoyce{T} <: ConstitutiveLaw
    μ::T
    λₘ::T
    κ::T
end

Base.length(M::ConstitutiveLaw) = fieldcount(typeof(M))
function Base.show(io::IO, ::MIME"text/plain", T::Type{<:ConstitutiveLaw})
    names = join(fieldnames(T), ", ")
    println(io, "$(T) model parameters: $names")
end

function init_material(type_constructor, ξ::Dict{Symbol,<:Real})
    params = fieldnames(type_constructor)
    values = [ξ[param] for param in params]
    _T = promote_type(map(typeof, values)...)
    T = _T <: Integer ? Float64 : _T
    return type_constructor(map(T, values)...)
end
for T in subtypes(ConstitutiveLaw)
    name = nameof(T)
    @eval $(Symbol(name))(ξ::Dict{Symbol,<:Real}) = init_material($(Symbol(name)), ξ)
end

function Ψ(mp::M) where M <: ConstitutiveLaw
    return Ψ(nothing,mp)
end
function Ψ(C, mp::NeoHookean)
    if isnothing(C)
        @variables Ī₁, bulk, μ, κ
    else
        J = sqrt(det(C))
        bulk = (J-1)^2
        Ī₁ = tr(C)*J^(-2/3)
        μ = mp.μ; κ = mp.κ
    end
    return μ/2*(Ī₁-3) + κ/2*bulk
end
function Ψ(C, mp::MooneyRivlin)
    if isnothing(C)
        @variables Ī₁, Ī₂, bulk, C₁₀, C₀₁, κ
    else
        J = sqrt(det(C))
        bulk = (J-1)^2
        Ī₁ = tr(C)*J^(-2/3)
        Ī₂ = 0.5*(mp.Ī₁^2-tr(C*C)*J^(-4/3))
        C₁₀ = mp.C₁₀; C₀₁ = mp.C₀₁; κ = mp.κ
    end
    return C₁₀*(Ī₁-3) + C₀₁*(Ī₂-3) + κ/2*bulk
end
function Ψ(C, mp::Yeoh2)
    if isnothing(C)
        @variables Ī₁, bulk, C₁₀, C₂₀, κ
    else
        J = sqrt(det(C))
        bulk = (J-1)^2
        Ī₁ = tr(C)*J^(-2/3)
        C₁₀ = mp.C₁₀; C₂₀ = mp.C₂₀; κ = mp.κ
    end
    return C₁₀*(Ī₁-3) + C₂₀*(Ī₁-3)^2 + (κ/2)*bulk
end
function Ψ(C, mp::Yeoh3)
    if isnothing(C)
        @variables Ī₁, bulk, C₁₀, C₂₀, C₃₀, κ
    else
        J = sqrt(det(C))
        bulk = (J-1)^2
        Ī₁ = tr(C)*J^(-2/3)
        C₁₀ = mp.C₁₀; C₂₀ =  mp.C₂₀; C₃₀ = mp.C₃₀; κ= mp.κ
    end
    return C₁₀*(Ī₁-3) + C₂₀*(Ī₁-3)^2 + C₃₀*(Ī₁-3)^3 + (κ/2)*bulk
end
function Ψ(C, mp::ArrudaBoyce)
    if isnothing(C)
        @variables Ī₁, bulk, μ, λₘ, κ
    else
        J = sqrt(det(C))
        bulk = (J-1)^2
        Ī₁ = tr(C)*J^(-2/3)
        μ = mp.μ; λₘ = mp.λₘ; κ = mp.κ
    end
    expansion_coeffs = [1/2 1/20 11/1050 19/7000 519/673750]
    return sum(i -> μ*(expansion_coeffs[i]/λₘ^(2i - 2))*(Ī₁^i - 3^i), 1:5) + (κ/2)*bulk
end