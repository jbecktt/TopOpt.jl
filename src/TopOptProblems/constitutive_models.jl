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

function NeoHookean(ξ::Dict{Symbol,<:Real})
    μ = ξ[:μ]
    κ = ξ[:κ]
    _T = promote_type(typeof(μ), typeof(κ))
    T = _T <: Integer ? Float64 : _T
    return NeoHookean(T(μ),T(κ))
end
function MooneyRivlin(ξ::Dict{Symbol,<:Real})
    C₁₀ = ξ[:C₁₀]
    C₀₁ = ξ[:C₀₁]
    κ = ξ[:κ]
    _T = promote_type(typeof(C₁₀), typeof(C₀₁), typeof(κ))
    T = _T <: Integer ? Float64 : _T
    return MooneyRivlin(T(C₁₀),T(C₀₁),T(κ))
end