"""
    apply_boundary_with_zerodiag!(Kσ, ch)

Apply boundary condition to a matrix. Zero-out the corresponding [i,:] and [:,j] with
i, j ∈ ch.prescribed_dofs.

This function is typically used with the stress stiffness matrix `Kσ`. More info about this can be found at: 
https://github.com/JuliaTopOpt/TopOpt.jl/wiki/Applying-boundary-conditions-to-the-stress-stiffness-matrix
"""
function apply_boundary_with_zerodiag!(Kσ, ch)
    T = eltype(Kσ)
    Ferrite.apply!(Kσ, T[], ch, true)
    for i in 1:length(ch.values)
        d = ch.prescribed_dofs[i]
        Kσ[d, d] = zero(T)
    end
    return Kσ
end

function ChainRulesCore.rrule(::typeof(apply_boundary_with_zerodiag!), Kσ, ch)
    project_to = ChainRulesCore.ProjectTo(Kσ)
    function pullback_fn(Δ)
        return NoTangent(), apply_boundary_with_zerodiag!(project_to(Δ), ch), NoTangent()
    end
    return apply_boundary_with_zerodiag!(Kσ, ch), pullback_fn
end

#=
Derivations for `rrule` of `apply_boundary_with_zerodiag!`

g(F(K)), F: K1 -> K2

dg/dK1_ij = dg/dK2_i'j' * dK2_i'j'/dK1_ij
          = Delta[i',j'] * dK2_i'j'/dK1_ij

dK2_i'j'/dK1_ij = 0, if i' or j' in ch.prescribed_dofs
                = 1, otherwise

dg/dK1_ij = 0, if i or j in ch.prescribed_dofs
          = Delta[i,j], otherwise
=#

########################################

"""
    apply_boundary_with_meandiag!(K, ch)

Apply boundary condition to a matrix. Zero-out the corresponding [i,:] and [:,j] with
i, j ∈ ch.prescribed_dofs, then fill in K[i,i] for i ∈ ch.prescribed_dofs with the
mean diagonal of the original matrix.
"""
function apply_boundary_with_meandiag!(
    K::Union{SparseMatrixCSC,Symmetric}, ch::ConstraintHandler, f::AbstractVector = eltype(K)[], applyzero::Bool = false
)
    if isempty(f)
        @assert f isa AbstractVector
    end
    Ferrite.apply!(K, f, ch, applyzero)
    return K, f
end

function ChainRulesCore.rrule(::typeof(apply_boundary_with_meandiag!), K1, ch, f1, applyzero)
    fisempty = isempty(f1)
    project_to_K1 = ChainRulesCore.ProjectTo(K1)
    project_to_f1 = fisempty ? nothing : ChainRulesCore.ProjectTo(f1)
    diagK1 = diag(K1)
    jac_meandiag = sign.(diagK1) / length(diagK1)
    function pullback_fn(Δ)
        ΔK2, Δf2 = Δ; 
        ΔK1=SparseMatrixCSC(zeros(size(K1)))
        Δf1 = zeros(eltype(f1), length(f1))
        if ΔK2!=ZeroTangent()
            ΔK1 = project_to_K1(deepcopy(ΔK2))
        end
        if Δf2!=ZeroTangent()
            Δf1 = Array(project_to_f1(Δf2))
        end
        ΔK2_ch_diagsum = zero(eltype(K1))
        for i in 1:length(ch.values)
            d = ch.prescribed_dofs[i]
            ΔK2_ch_diagsum += ΔK2[d,d]
        end
        apply_boundary_with_zerodiag!(ΔK1, ch)
        ΔK1=Matrix(ΔK1)
        for i in 1:size(K1, 1)
            ΔK1[i, i] += ΔK2_ch_diagsum * jac_meandiag[i]
            if applyzero == false && !fisempty
                if i ∉ ch.prescribed_dofs 
                    for j in 1:length(ch.prescribed_dofs) 
                        d = ch.prescribed_dofs[j]
                        ΔK1[i,d] +=  Δf2[i] * -ch.values[j]
                    end
                end
                ΔK1[i, i] += sum(Δf2[ch.prescribed_dofs] .* ch.values .* jac_meandiag[i])
            end
        end
        if fisempty
            return NoTangent(), ΔK1, NoTangent(), NoTangent(), NoTangent()
        else
            Δf1[ch.prescribed_dofs] .= 0 # Add correction for inhomogenous DBCs
            return NoTangent(), ΔK1, NoTangent(), Δf1, NoTangent()
        end
    end
    Ferrite.apply!(K1,f1,ch,applyzero)
    return (K1,f1), pullback_fn
end

#=
Derivations for `rrule` of `apply_boundary_with_meandiag!`
    g(F(K,f)), F(K1,f1) -> K2(K1) and f2(K1,f1) 
    The outputs (K2 and f2) do rely on ch and applyzero however these derivatives are not considered 
    because they are not functions of pd whereas K1 and f1 both functions of pseudodensities.

[1] ∂g/∂K1_ij = (sum_i'j' ∂g/∂K2_i'j' * ∂K2_i'j'/∂K1_ij) + (sum_i' ∂g/∂f2_i' * ∂f2_i'/∂K1_ij)
              = (sum_i'j' ΔK2_i'j' * ∂K2_i'j'/∂K1_ij) + (sum_i' Δf2_i' * ∂f2_i'/∂K1_ij)

[1.1] If applyzero == true
Derive ∂K2_i'j'/∂K1_ij:
If i' != j' and (i' or j' ∈ ch.prescribed_dofs)
    ∂K2_i'j'/∂K1_ij = 0
If i' != j' and (i' or j' ∉ ch.prescribed_dofs)
    If i' == i and j' == j
        ∂K2_i'j'/∂K1_ij = 1
    If i' != i or j' != j
        ∂K2_i'j'/∂K1_ij = 0
If i' == j' and !(i' or j' ∈ ch.prescribed_dofs)
    If i' == i and j' == j
        ∂K2_i'j'/∂K1_ij = 1
    If i' != i or j' != j
        ∂K2_i'j'/∂K1_ij = 0
If i' == j' and (i' or j' ∈ ch.prescribed_dofs)
    If i == j
        ∂K2_i'j'/∂K1_ij = ∂(meandiag(K1))/∂K1_ii
                        = sign(K1_ii) / size(K1, 1)
    If i != j
        ∂K2_i'j'/∂K1_ij = 0
Derive ∂f2_i'/∂K1_ij:
∂f2_i'/∂K1_ij = 0
Derive ∂g/∂K1_ij:
If i != j and (i or j ∈ ch.prescribed_dofs)
    ∂g/∂K1_ij = 0
If i != j and (i or j ∉ ch.prescribed_dofs)
    ∂g/∂K1_ij = ΔK2_ij
If i == j and i ∈ prescribed_dofs
    dg/dK1_ii = sum_{i' in prescribed_dofs} ΔK2_i'i' * d(meandiag(K1))/dK1_ii
              = sum_{i' in prescribed_dofs} ΔK2_i'i' * sign(K1_ii) / size(K1, 1)
If i == j and !(i in prescribed_dofs)
    dg/dK1_ii = ΔK2_ii + sum_{i' in prescribed_dofs} ΔK2_i'i' * d(meandiag(K1))/dK1_ii
              = ΔK2_ii + sum_{i' in prescribed_dofs} ΔK2_i'i' * sign(K1_ii) / size(K1, 1)

[1.2] If applyzero == false and isempty(f)
Derive ∂K2_i'j'/∂K1_ij:
The derivation for ∂K2_i'j'/∂K1_ij is independent of applyzero. See [1.1] for details.
Derive ∂f2_i'/∂K1_ij:
∂f2_i'/∂K1_ij = f1 (i.e., empty set because f is empty and unmodified)
Derive ∂g/∂K1_ij:
The derivation for ∂g/∂K1_ij is identical to [1.1] because the second term does not contribute. 

[1.3] If applyzero == false and ~isempty(f)
Derive ∂K2_i'j'/∂K1_ij:
The derivation for ∂K2_i'j'/∂K1_ij is independent of applyzero. See [1.1] for details.
Derive ∂f2_i'/∂K1_ij:
If i' == i and j ∈ ch.prescribed_dofs and i ∉ ch.prescribed_dofs
    ∂f2_i'/∂K1_ij = -ch.values[j]
If i' ∈ ch.prescribed_dofs and i == j
    ∂f2_i'/∂K1_ij = ∂(meandiag(K1))/∂K1_ii * ch.values[i']
                  = sign(K1_ii) / size(K1, 1) * ch.values[i']
Else 
    ∂f2_i'/∂K1_ij = 0 
Derive ∂g/∂K1_ij:
If i != j
    If j ∈ ch.prescribed_dofs
        ∂g/∂K1_ij = Δf2[i] * -ch.values[j]
    If i ∈ ch.prescribed_dofs
        ∂g/∂K1_ij = 0
If i != j and (i or j ∉ ch.prescribed_dofs)
    ∂g/∂K1_ij = ΔK2_ij
If i == j and i ∈ prescribed_dofs
    dg/dK1_ii = sum_{i' in prescribed_dofs} (ΔK2_i'i' * sign(K1_ii) / size(K1, 1) + Δf2_i' * sign(K1_ii) / size(K1, 1) * ch.values[i'])
              = sum_{i' in prescribed_dofs} ((ΔK2_i'i' + Δf2_i' * ch.values[i']) * sign(K1_ii) / size(K1, 1))
If i == j and !(i in prescribed_dofs)
    dg/dK1_ii = ΔK2_ii + sum_{i' in prescribed_dofs} (ΔK2_i'i' * sign(K1_ii) / size(K1, 1) + Δf2_i' * sign(K1_ii) / size(K1, 1) * ch.values[i'])
              = ΔK2_ii + sum_{i' in prescribed_dofs} ((ΔK2_i'i' + Δf2_i' * ch.values[i']) * sign(K1_ii) / size(K1, 1))

[2] ∂g/∂f1_i = sum_j ∂g/∂f2_j * ∂f2_j/∂f1_i + ∂g/∂K2_i'j' * ∂K2_i'j'/∂f1_ij
             = sum_j Δf2_j * ∂f2_j/∂f1_i + ∂g/∂K2_i'j' * 0 
             = sum_j Δf2_j * ∂f2_j/∂f1_i

[2.1] If isempty(f): 
∂g/∂f1 = f1 (i.e., empty set because f is empty and unmodified)

[2.2] If ~isempty(f): 
Derive ∂f2_j/∂f1_i:
If i == j and i ∉ ch.prescribed_dofs
    ∂f2_j/∂f1_i = 1
Else 
    ∂f2_j/∂f1_i = 0
Derive ∂g/∂f1_i:
If i ∉ ch.prescribed_dofs
    ∂g/∂f1_i = Δf2_i
Else
    ∂g/∂f1_i = 0
=#