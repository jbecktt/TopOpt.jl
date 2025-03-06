@params mutable struct ElementG{T} <: AbstractFunction{T}
    solver::TopOpt.FEA.HyperelasticDisplacementSolver
    element_g::AbstractVector{T}
    Kesize::Number
    body_force::Vector{Float64}
    cellvalues::CellScalarValues
    cellvaluesV::CellVectorValues
end

function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::ElementG)
    return println("TopOpt element hyperelastic force residual function")
end

function ElementG(solver)
    quad_order=FEA.default_quad_order(solver.problem)
    dims = TopOptProblems.getdim(solver.problem)
    geom_order = TopOptProblems.getgeomorder(solver.problem)
    refshape = Ferrite.getrefshape(solver.problem.ch.dh.field_interpolations[1])
    interpolation_space = Ferrite.Lagrange{dims,refshape,geom_order}()
    quadrature_rule = Ferrite.QuadratureRule{dims,refshape}(quad_order) 
    nel = getncells(solver.problem.ch.dh.grid)
    T_ = TopOptProblems.floattype(solver.problem)
    Kes_solver = solver.elementinfo.Kes 
    _Ke1 = rawmatrix(Kes_solver[1])
    mat_type = _Ke1 isa Symmetric ? typeof(_Ke1.data) : typeof(_Ke1)
    cellvalues = CellScalarValues(quadrature_rule, interpolation_space)
    cellvaluesV = Ferrite.CellVectorValues(quadrature_rule, interpolation_space) 
    n_basefuncs = getnbasefunctions(cellvalues)
    Kesize = dims * n_basefuncs
    ρ = TopOptProblems.getdensity(solver.problem)
    MatrixType, VectorType = TopOptProblems.gettypes(T_, Val{mat_type}, Val{Kesize};hyperelastic=true)
    g = [0.0, 9.81, 0.0]
    body_force = ρ .* g
    ges=[zeros(StaticArraysCore.SVector{ndofs_per_cell(solver.problem.ch.dh), Float64}) for i in 1:length(solver.problem.varind)]
    return  ElementG(solver, ges, Kesize, body_force, cellvalues, cellvaluesV) 
end

function (eg::ElementG)(ue, cellvaluesV, cellvalues)
    @unpack solver, Kesize, body_force, cellvalues, cellvaluesV=eg    

    ge=zeros(Number,(ndofs_per_cell(solver.problem.ch.dh)))
    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        ∇u = function_gradient(cellvaluesV, q_point, ue) # JGB add (NEEDS TO BE CHECKED!!)
        F = one(∇u) + ∇u # JGB add 
        C = tdot(F) # JGB add 
        S, ∂S∂C = TopOptProblems.constitutive_driver(C, solver.mp) # JGB add 
        P = F ⋅ S # JGB add 
        for b in 1:Kesize
            ∇ϕb = shape_gradient(cellvaluesV, q_point, b) # JGB: like ∇δui
            ϕb = shape_value(cellvaluesV, q_point, b)
            ge[b] += ( ∇ϕb ⊡ P - ϕb ⋅ body_force ) * dΩ
        end
    end
    return ge
end

function (eg::ElementG)(u::TopOpt.Functions.DisplacementResult{T, N, V}) where {T, N, V}
@unpack solver, element_g, cellvalues, cellvaluesV=eg    
    element_g=Vector{Vector{Float64}}()
    celliterator = CellIterator(solver.problem.ch.dh)
    for (ci, cell) in enumerate(celliterator)
        dofs=celldofs(cell)
        _reinit!(cellvalues, cell)
        _reinit!(cellvaluesV, cell)
        cell_g=eg(u.u[dofs], cellvaluesV, cellvalues)
        element_g=vcat(element_g,[cell_g])
    end
    return element_g
end

function (eg::ElementG)(u::AbstractArray)
    @unpack solver=eg 
    @assert length(u)==ndofs(solver.problem.ch.dh)
return eg(DisplacementResult(u)) ;end


function _reinit!(cellvalues::Ferrite.CellScalarValues,cell::CellIterator)
    reinit!(cellvalues, cell)
    return cellvalues
end

function _reinit!(cellvaluesV::Ferrite.CellVectorValues,cell::CellIterator)
    reinit!(cellvaluesV, cell)
    return cellvaluesV
end

function ChainRulesCore.rrule(::typeof(_reinit!), cellvalues::Ferrite.CellScalarValues, cell::CellIterator)
    function pullback_fn(Δ)
        return NoTangent(), NoTangent(),NoTangent()
    end
    return reinit!(cellvalues, cell), pullback_fn
end

function ChainRulesCore.rrule(::typeof(_reinit!), cellvaluesV::Ferrite.CellVectorValues, cell::CellIterator)
    function pullback_fn(Δ)
        return NoTangent(), NoTangent(),NoTangent()
    end
    return reinit!(cellvaluesV, cell), pullback_fn
end

function ChainRulesCore.rrule(::typeof(⊡), A::SecondOrderTensor, B::SecondOrderTensor)
    project_to_A = ChainRulesCore.ProjectTo(A)
    project_to_B = ChainRulesCore.ProjectTo(B)
    function pullback_fn(Δ)
        return NoTangent(), @thunk(project_to_A(Δ * B)), @thunk(project_to_B(Δ  * A))
    end
    return A ⊡ B, pullback_fn
end

function ChainRulesCore.rrule(eg::ElementG, u::DisplacementResult)
    @unpack solver, cellvalues, cellvaluesV=eg    
    ges = eg(u)
    function pullback_fn(Δ)
        celliterator = CellIterator(solver.problem.ch.dh)
        Δu = zeros(length(u))
        Δu_threaded = [zeros(length(u.u)) for _ in 1:Threads.nthreads()]
        for (ci, cell) in enumerate(celliterator)  
            dofs=celldofs(cell)
            _reinit!(cellvalues, cell)
            _reinit!(cellvaluesV, cell)
            ges_cell_fn_ue = ue -> vec(eg(ue, cellvaluesV, cellvalues))
            jacobian_options=ForwardDiff.JacobianConfig(ges_cell_fn_ue,u.u[dofs],ForwardDiff.Chunk{Int64(round(length(dofs)))}())
            jac_cell_ue = ForwardDiff.jacobian(ges_cell_fn_ue, (u.u[dofs]),jacobian_options)
            ThreadsX.foreach(1:ndofs_per_cell(solver.problem.ch.dh)) do i
                tid = Threads.threadid()
                Δu_threaded[tid][dofs[i]] += dot(jac_cell_ue[:, i], vec(Δ[ci]))
            end
        end
        Δu = ThreadsX.mapreduce(identity, +, Δu_threaded)
        return Tangent{typeof(eg)}(;
        solver=NoTangent(),
        element_g=NoTangent()
    ),
        Δu
    end
    return ges, pullback_fn
end 