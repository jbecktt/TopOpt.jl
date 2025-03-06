struct ElementStiffnessMatrix{T<:Real,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    Ke::M
end
Base.length(x::ElementStiffnessMatrix) = length(x.x)
Base.size(x::ElementStiffnessMatrix, i...) = size(x.x, i...)
Base.getindex(x::ElementStiffnessMatrix, i...) = x.x[i...]
Base.:*(x::ElementStiffnessMatrix, y) = ElementStiffnessMatrix(x.x * y)

@params mutable struct ElementK{T} <: AbstractFunction{T}
    solver::AbstractFEASolver
    Kes::AbstractVector{<:AbstractMatrix{T}}
    Kes_0::AbstractVector{<:AbstractMatrix{T}} # un-interpolated
    penalty::AbstractPenalty{T}
    xmin::T
    Kesize::Integer
    cellvalues::CellScalarValues
    cellvaluesV::CellVectorValues
end

function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::ElementK)
    return println("TopOpt element stiffness matrix construction function")
end

function ElementK(solver::AbstractDisplacementSolver)
    @unpack elementinfo = solver
    dh = solver.problem.ch.dh
    penalty = getpenalty(solver)
    xmin = solver.xmin
    solver.vars = ones(getncells(dh.grid))
    # trigger Ke construction
    solver()
    Kes_solver = solver.elementinfo.Kes

    _Ke1 = rawmatrix(Kes_solver[1])
    mat_type = _Ke1 isa Symmetric ? typeof(_Ke1.data) : typeof(_Ke1)
    Kes = mat_type[]
    Kes_0 = mat_type[]
    for (cellidx, _) in enumerate(CellIterator(dh))
        _Ke = rawmatrix(Kes_solver[cellidx])
        Ke0 = _Ke isa Symmetric ? _Ke.data : _Ke
        Ke = similar(Ke0)
        push!(Kes_0, Ke0)
        push!(Kes, Ke)
    end
    # Extra stuff added to make linear and hyperelastic solver share same class (may be unnecessary) -- JGB
    quad_order=FEA.default_quad_order(solver.problem)
    dims = TopOptProblems.getdim(solver.problem)
    geom_order = TopOptProblems.getgeomorder(solver.problem)
    refshape = Ferrite.getrefshape(solver.problem.ch.dh.field_interpolations[1])
    interpolation_space = Ferrite.Lagrange{dims,refshape,geom_order}()
    quadrature_rule = Ferrite.QuadratureRule{dims,refshape}(quad_order) 
    cellvalues = CellScalarValues(quadrature_rule, interpolation_space)
    cellvaluesV = Ferrite.CellVectorValues(quadrature_rule, interpolation_space)
    n_basefuncs = getnbasefunctions(cellvalues)
    Kesize = dims * n_basefuncs
    return ElementK(solver, Kes, Kes_0, penalty, xmin, Kesize, cellvalues,cellvaluesV)
end

function (ek::ElementK)(xe::Number, ci::Integer)
    @unpack xmin, Kes_0, penalty = ek
    if PENALTY_BEFORE_INTERPOLATION
        px = density(penalty(xe), xmin)
    else
        px = penalty(density(xe), xmin)
    end
    return px * Kes_0[ci]
end

function (ek::ElementK{T})(x::PseudoDensities) where {T}
    @unpack solver, Kes = ek
    @assert getncells(solver.problem.ch.dh.grid) == length(x)
    for ci in 1:length(x)
        Kes[ci] = ek(x.x[ci], ci)
    end
    return copy(Kes)
end

function ChainRulesCore.rrule(ek::ElementK, x::PseudoDensities)
    @unpack solver, Kes = ek
    @assert getncells(solver.problem.ch.dh.grid) == length(x.x)
    Kes = ek(x)

    """
    g(F(x)), where F = ElementK

    Want: dg/dK_e_ij -> dg/dx_e

    dg/dx_e = sum_i'j' dg/dK_e_i'j' * dK_e_i'j'/dx_e
            = Delta[e][i,j] * dK_e_i'j'/dx_e
    """
    function pullback_fn(Δ)
        Δx = similar(x.x)
        for ci in 1:length(x.x)
            ek_cell_fn = xe -> vec(ek(xe, ci))
            jac_cell = ForwardDiff.derivative(ek_cell_fn, x.x[ci])
            Δx[ci] = jac_cell' * vec(Δ[ci])
        end
        return Tangent{typeof(ek)}(;
            solver=NoTangent(),
            Kes=Δ,
            Kes_0=NoTangent(),
            penalty=NoTangent(),
            xmin=NoTangent(),
        ),
        Tangent{typeof(x)}(; x=Δx)
    end
    return Kes, pullback_fn
end 

function ElementK(solver::AbstractHyperelasticSolver)
    @unpack elementinfo = solver
    dh = solver.problem.ch.dh 
    penalty = getpenalty(solver)
    xmin = solver.xmin
    solver.vars = ones(getncells(dh.grid))
    # trigger Ke construction
    Kes_solver = solver.elementinfo.Kes

    _Ke1 = rawmatrix(Kes_solver[1])
    mat_type = _Ke1 isa Symmetric ? typeof(_Ke1.data) : typeof(_Ke1)
    Kes = mat_type[]
    Kes_0 = mat_type[]
    for (cellidx, _) in enumerate(CellIterator(dh))
        _Ke = rawmatrix(Kes_solver[cellidx])
        Ke0 = _Ke isa Symmetric ? _Ke.data : _Ke
        Ke = similar(Ke0)
        push!(Kes_0, Ke0)
        push!(Kes, Ke)
    end
    quad_order=FEA.default_quad_order(solver.problem)
    dims = TopOptProblems.getdim(solver.problem)
    geom_order = TopOptProblems.getgeomorder(solver.problem)
    refshape = Ferrite.getrefshape(solver.problem.ch.dh.field_interpolations[1])
    interpolation_space = Ferrite.Lagrange{dims,refshape,geom_order}()
    quadrature_rule = Ferrite.QuadratureRule{dims,refshape}(quad_order) 
    cellvalues = CellScalarValues(quadrature_rule, interpolation_space)
    cellvaluesV = Ferrite.CellVectorValues(quadrature_rule, interpolation_space) # JGB: change from CellScalarValues 
    n_basefuncs = getnbasefunctions(cellvalues)
    Kesize = dims * n_basefuncs
    return ElementK(solver, Kes, Kes_0, penalty, xmin, Kesize, cellvalues,cellvaluesV)
end

function (ek::ElementK)(xe, ue,ci)
    @unpack xmin, penalty, Kesize, cellvalues,cellvaluesV  = ek
    Fe = zeros(3,3)
    collected_celliterator = collect(CellIterator(ek.solver.problem.ch.dh))
    
    reinit!(cellvalues, collected_celliterator[ci])
    reinit!(cellvaluesV, collected_celliterator[ci])
 
    Ke_0 = Matrix{Number}(undef, Kesize, Kesize)
    Ke_0 .= 0
    for q_point in 1:getnquadpoints(cellvalues)
        dΩ = getdetJdV(cellvalues, q_point)
        ∇u = function_gradient(cellvaluesV, q_point, ue) # JGB add (NEEDS TO BE CHECKED!!)
        F = one(∇u) + ∇u # JGB add 
        Fe += F*dΩ
        C = tdot(F) # JGB add 
        S, ∂S∂C = TopOptProblems.constitutive_driver(C, ek.solver.mp) # JGB add 
        I = one(S) # JGB add 
        ∂P∂F =  otimesu(I, S) + 2 * otimesu(F, I) ⊡ ∂S∂C ⊡ otimesu(F', I) # JGB add (neither P or F is symmetric so ∂P∂F will not either)
        for b in 1:Kesize
            ∇ϕb = shape_gradient(cellvaluesV, q_point, b) # JGB: like ∇δui
            ∇ϕb∂P∂F = ∇ϕb ⊡ ∂P∂F # Hoisted computation # JGB add
            for a in 1:Kesize
                ∇ϕa = shape_gradient(cellvaluesV, q_point, a) # JGB: like ∇δuj
                Ke_0[a,b] += (∇ϕb∂P∂F ⊡ ∇ϕa) * dΩ
            end
        end
    end

    if PENALTY_BEFORE_INTERPOLATION
        px = density(penalty(xe), xmin)
    else
        px = penalty(density(xe), xmin)
    end
    return px * Ke_0
end


function (ek::ElementK{T})(x::PseudoDensities, u::DisplacementResult) where {T}
    @unpack solver, Kes = ek
    @assert getncells(solver.problem.ch.dh.grid) == length(x)
    celliterator = CellIterator(solver.problem.ch.dh)
    for (ci, cell) in enumerate(celliterator)
        Kes[ci] = ek(x.x[ci], u[celldofs(cell)],ci)
    end
    element_Kes = TopOpt.convert(
        Vector{<:ElementMatrix},
        Kes;
    
        bc_dofs=solver.problem.ch.prescribed_dofs,
        dof_cells=solver.problem.metadata.dof_cells,
    )
    return copy(element_Kes)
end

function ChainRulesCore.rrule(ek::ElementK, x::PseudoDensities, u::DisplacementResult)
    @unpack solver, Kes = ek
    Kes = ek(x, u)

    """
    g(F(x,u)), where F = ElementK

    Want: dg/dK_e_ij -> dg/dx_e, dg/du_e

    dg/dx_e = sum_i'j' dg/dK_e_i'j' * dK_e_i'j'/dx_e
            = Delta[e][i',j'] * dK_e_i'j'/dx_e

    dg/du_e= sum_i'j' dg/dK_e_i'j' * dK_e_i'j'/du_e
           = Delta[e][i',j'] * dK_e_i'j'/du_e
    """
    function pullback_fn(Δ)
        Δx = similar(x.x)
        Δu_threaded = [zeros(length(u.u)) for _ in 1:Threads.nthreads()]
        Δu = zeros(length(u.u)) 
        cellarray = collect(enumerate(CellIterator(solver.problem.ch.dh)))
        dofs_matrix=solver.problem.metadata.cell_dofs
        ThreadsX.foreach(cellarray) do (ci, cell_)
            dofs=dofs_matrix[:,ci]
            ek_cell_fn_xe = xe -> reshape(ek(xe, u[dofs],ci),:)
            jac_cell_xe = ForwardDiff.derivative(ek_cell_fn_xe, x.x[ci])
            Δx[ci] = jac_cell_xe' * reshape(Δ[ci],:)



            ek_cell_fn_ue = ue -> reshape(ek(x.x[ci], ue,ci),:)
            jacobian_options=ForwardDiff.JacobianConfig(
                ek_cell_fn_ue,u.u[dofs],
                ForwardDiff.Chunk{length(dofs)}()
            )
            jac_cell_ue = ForwardDiff.jacobian(
                ek_cell_fn_ue, 
                u.u[dofs],
                jacobian_options
            )
            tid = Threads.threadid()
            ThreadsX.foreach(1:ndofs_per_cell(solver.problem.ch.dh)) do i
                Δu_threaded[tid][dofs[i]] += jac_cell_ue[:, i]' * reshape(Δ[ci],:)
            end

        end                                                                                                                                                                                                                                                                                                 
        Δu = ThreadsX.mapreduce(identity, +, Δu_threaded)
        return Tangent{typeof(ek)}(;
            solver=NoTangent(),
            Kes=Δ,
            Kes_0=NoTangent(),
            penalty=NoTangent(),
            xmin=NoTangent(),
        ),
        Δx, Δu
    end
    return Kes, pullback_fn
end