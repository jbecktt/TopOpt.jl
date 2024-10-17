abstract type ConstitutiveLaw end

@params struct NeoHooke{T} <: ConstitutiveLaw
    μ::T
    λ::T
end
@params struct MooneyRivlin{T} <: ConstitutiveLaw
    C₁₀::T
    C₀₁::T
    κ::T
end

function NeoHooke(ξ::Dict{Symbol,<:Real})
    μ = ξ[:μ]
    λ = ξ[:λ]
    _T = promote_type(typeof(μ), typeof(λ))
    T = _T <: Integer ? Float64 : _T
    return NeoHooke(T(μ),T(λ))
end
function MooneyRivlin(ξ::Dict{Symbol,AbstractFloat})
    C₁₀ = ξ[:C₁₀]
    C₀₁ = ξ[:C₀₁]
    κ = ξ[:κ]
    _T = promote_type(typeof(C₁₀), typeof(C₀₁), typeof(κ))
    T = _T <: Integer ? Float64 : _T
    return MooneyRivlin(T(C₁₀),T(C₀₁),T(κ))
end