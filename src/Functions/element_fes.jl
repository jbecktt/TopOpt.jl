@params mutable struct ElementF{T} <: AbstractFunction{T}
    solver::TopOpt.FEA.HyperelasticDisplacementSolver
    element_f::AbstractVector{T}
    quad_order::Integer
    cellvalues::CellScalarValues
    cellvaluesV::CellVectorValues
    Kesize::Number
    body_force::Vector{Float64}
end

function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::ElementF)
    return println("TopOpt element-wise forcing vector function")
end

function ElementF(solver)
    quad_order=FEA.default_quad_order(solver.problem)
    dims = TopOptProblems.getdim(solver.problem)
    geom_order = TopOptProblems.getgeomorder(solver.problem)
    refshape = Ferrite.getrefshape(solver.problem.ch.dh.field_interpolations[1])
    interpolation_space = Ferrite.Lagrange{dims,refshape,geom_order}()
    quadrature_rule = Ferrite.QuadratureRule{dims,refshape}(quad_order) 
    nel = getncells(solver.problem.ch.dh.grid)
    fes=[zeros(StaticArraysCore.SVector{ndofs_per_cell(solver.problem.ch.dh), Float64}) for i in 1:length(solver.problem.varind)]
    cellvalues = CellScalarValues(quadrature_rule, interpolation_space)
    cellvaluesV = Ferrite.CellVectorValues(quadrature_rule, interpolation_space) 
    T_ = TopOptProblems.floattype(solver.problem)
    Kes_solver = solver.elementinfo.Kes 
    _Ke1 = rawmatrix(Kes_solver[1])
    mat_type = _Ke1 isa Symmetric ? typeof(_Ke1.data) : typeof(_Ke1)
    n_basefuncs = getnbasefunctions(cellvalues)
        
    Kesize = dims * n_basefuncs 
    g = [0.0, 9.81, 0.0]
    ρ = TopOptProblems.getdensity(solver.problem)
    body_force = ρ .* g
    MatrixType, VectorType = TopOptProblems.gettypes(T_, Val{mat_type}, Val{Kesize};hyperelastic=true)
    return  ElementF(solver,fes, quad_order, cellvalues, cellvaluesV, Kesize, body_force) 
end

function (ef::ElementF)(ue, cellvaluesV, cellvalues)
    @unpack solver,cellvalues, cellvaluesV, body_force, Kesize =ef
    fe=zeros(Number,(ndofs_per_cell(solver.problem.ch.dh)))
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
            fe[b] += ϕb ⋅ body_force * dΩ
        end
    end
    return fe
end

function (ef::ElementF)(u::TopOpt.Functions.DisplacementResult{T, N, V}) where {T, N, V}
    @unpack solver,element_f, cellvalues, cellvaluesV=ef
    
    element_f=Vector{Vector{Float64}}()
    celliterator = CellIterator(solver.problem.ch.dh)
    for (ci, cell) in enumerate(celliterator)
        dofs=celldofs(cell)
        _reinit!(cellvalues, cell)
        _reinit!(cellvaluesV, cell)
        cell_f=ef(u.u[dofs], cellvaluesV, cellvalues)
        element_f=vcat(element_f,[cell_f])
    end
    return element_f
end

function (ef::ElementF)(u::AbstractArray)
    @unpack solver=ef 
    @assert length(u)==ndofs(solver.problem.ch.dh)
return ef(DisplacementResult(u)) ;end

function ChainRulesCore.rrule(ef::ElementF, u::DisplacementResult)
    @unpack solver,cellvalues, cellvaluesV =ef
    fes = ef(u)
    function pullback_fn(Δ)
        celliterator = CellIterator(solver.problem.ch.dh)
        Δu = zeros(length(u))
        Δu_threaded = [zeros(length(u.u)) for _ in 1:Threads.nthreads()]
        for (ci, cell) in enumerate(celliterator)  
            dofs=celldofs(cell)
            _reinit!(cellvalues, cell)
            _reinit!(cellvaluesV, cell)
            fes_cell_fn_ue = ue -> vec(ef(ue, cellvaluesV, cellvalues))
            jacobian_options=ForwardDiff.JacobianConfig(fes_cell_fn_ue,u.u[dofs],ForwardDiff.Chunk{Int64(round(length(dofs)))}())
            jac_cell_ue = ForwardDiff.jacobian(fes_cell_fn_ue, (u.u[dofs]),jacobian_options)
            ThreadsX.foreach(1:ndofs_per_cell(solver.problem.ch.dh)) do i 
                tid = Threads.threadid()
                Δu_threaded[tid][dofs[i]] +=  jac_cell_ue[:,i]' * vec(Δ[ci])
            end
        end
        Δu = ThreadsX.mapreduce(identity, +, Δu_threaded)
        return Tangent{typeof(ef)}(;
        solver=NoTangent(),
        element_f=NoTangent()
    ),
        Δu
    end
    return fes, pullback_fn
end 