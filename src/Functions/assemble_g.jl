@TopOpt.params mutable struct AssembleG{T} <: AbstractFunction{T}
    solver::TopOpt.FEA.HyperelasticDisplacementSolver
    assembled_g::AbstractVector{T}
end

function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::AssembleG)
    return println("TopOpt global hyperelastic force residual assembly function")
end

function AssembleG(solver)
    assembled_g=zeros(ndofs(solver.problem.ch.dh))
    return  AssembleG(solver,assembled_g)
end

function (assemble_g::AssembleG)(element_g::AbstractVector{<:AbstractVector{<:Number}})
    @unpack solver,assembled_g=assemble_g
    assembled_g=zeros(ndofs(solver.problem.ch.dh))
    # celliterator = CellIterator(solver.problem.ch.dh)
    penalty=TopOpt.getpenalty(solver)
    celldofs_matrix=solver.elementinfo.metadata.cell_dofs
    for i in 1:length(element_g)
        dofs=celldofs_matrix[:,i]
        ge = element_g[i]
        if solver.problem.black[i]
            if assemble_f
                assembled_g=Assemble_G_i!(assembled_g, Int64.(copy(dofs)),ge)
            else
            end
        elseif solver.problem.white[i]
            if PENALTY_BEFORE_INTERPOLATION
                px = xmin
            else
                px = penalty(xmin)
            end
            ge = px .* Float64.(ge)
            assembled_g=Assemble_G_i!(assembled_g, Int64.(copy(dofs)),ge)
        else
            if PENALTY_BEFORE_INTERPOLATION
                px = TopOpt.density(penalty(solver.vars[solver.problem.varind[i]]), solver.xmin)
            else
                px = penalty(TopOpt.density(solver.vars[solver.problem.varind[i]], solver.xmin))
            end

            ge = px .* Float64.(ge)
            assembled_g=Assemble_G_i!(assembled_g, Int64.(copy(dofs)),ge)
        end
    end
    return assembled_g
end

function Assemble_G_i!(g,dofs,ge)
    g_=Float64.(g)
    Ferrite.assemble!(g_,dofs,Float64.(ge))
    return g_
end

function ChainRulesCore.rrule(::typeof(Assemble_G_i!), g,dofs,elem_g)
    function pullback_fn(Δ)
        δz_δge= Δ[dofs]
        δz_δg=Δ
        return NoTangent(), δz_δg, NoTangent() , δz_δge
    end
    return Assemble_G_i!(g,dofs,elem_g), pullback_fn
end