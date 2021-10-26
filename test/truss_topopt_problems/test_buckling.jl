using Test
using LinearAlgebra
using StaticArrays
using Ferrite

using TopOpt
using TopOpt.TrussTopOptProblems.TrussTopOptProblems: buckling, get_truss_Kσs, compute_local_axes
using TopOpt.TopOptProblems: getE, getdh
using TopOpt.TrussTopOptProblems: getA
using Arpack
# using IterativeSolvers

fea_ins_dir = joinpath(@__DIR__, "instances", "fea_examples");
gm_ins_dir = joinpath(@__DIR__, "instances", "ground_meshes");

# @testset "Tim buckling $problem_dim" for problem_dim in ["2d", "3d"]
#     file_name = "tim_$(problem_dim).json"
#     problem_file = joinpath(gm_ins_dir, file_name)

#     mats = TrussFEAMaterial(1.0, 0.3);
#     crossecs = TrussFEACrossSec(800.0);

#     node_points, elements, _, _ , fixities, load_cases = load_truss_json(problem_file)
#     ndim, nnodes, ncells = length(node_points[1]), length(node_points), length(elements)
#     loads = load_cases["0"]

#     problem = TrussProblem(Val{:Linear}, node_points, elements, loads, fixities, mats, crossecs);

#     # * Before optimization, check ground mesh stability
#     solver = FEASolver(Direct, problem);
#     solver()
#     K, G = buckling(problem, solver.globalinfo, solver.elementinfo; u=solver.u);
#     @test isfinite(logdet(cholesky(K+G)))

#     # API: A v = λ B v, B needs to be PD for numberical stability
#     # K + λ G = 0 <=> K v = -λ G v
#     # * 1/λ K + G = 0 <=> -G v = 1/λ K v
#     # dense_eigvals, buckmodes = eigen(Array(-G),Array(K));
#     # @show dense_smallest_pos_eigval = minimum(1 ./ dense_eigvals[findall(dense_eigvals.>0)]);

#     sparse_eigvals, buckmodes = eigs(-G,K, nev=1, which=:LR)
#     smallest_pos_eigval = 1/sparse_eigvals[1]

#     # @test smallest_pos_eigval == smallest_pos_eigval
#     @test smallest_pos_eigval >= 1.0

#     # * Run optimization, the optimized result should be unstable
#     xmin = 0.0001 # minimum density
#     x0 = fill(1.0, ncells) # initial design
#     p = 4.0 # penalty
#     V = 0.5 # maximum volume fraction

#     comp = TopOpt.Compliance(problem, solver)
#     function obj(x)
#         # minimize compliance
#         return comp(x)
#     end
#     function constr(x)
#         # volume fraction constraint
#         return sum(x) / length(x) - V
#     end

#     m = Model(obj)
#     addvar!(m, zeros(length(x0)), ones(length(x0)))
#     Nonconvex.add_ineq_constraint!(m, constr)

#     options = MMAOptions(
#         maxiter=1000, tol = Tolerance(kkt = 1e-4, f = 1e-4),
#     )
#     TopOpt.setpenalty!(solver, p)
#     r = Nonconvex.optimize(
#         m, MMA87(dualoptimizer = ConjugateGradient()),
#         x0, options = options,
#     );

#     solver = FEASolver(Direct, problem; xmin=xmin);
#     solver.vars = r.minimizer;
#     solver()
#     K, G = buckling(problem, solver.globalinfo, solver.elementinfo; u=solver.u);
#     # check optimizing compliance without buckling constraint will lead to instable structures
#     @test_throws PosDefException logdet(cholesky(K+G))

#     # using Makie
#     # using TopOpt.TrussTopOptProblems.TrussVisualization: visualize
#     # fig = visualize(problem; topology=r.minimizer)
# end

@testset "large displacement simulation MGZ-ex 9.1" begin
    problem_file = joinpath(fea_ins_dir, "mgz_geom_stiff_ex9.1.json")

    node_points, elements, mats, crosssecs, fixities, load_cases = load_truss_json(problem_file);

    P_val = 400 # kN
    P = Dict(2 => SVector{2}(P_val * [0.05, -1.0]))
    # P = load_cases["0"]

    maxit = 100
    utol = 1e-5
    x0 = fill(1.0, length(elements)) # initial design

    # * initialize displacements as zero
    nnode = length(node_points)
    u0 = zeros(nnode*2)

    # * additional fields
    u1 = deepcopy(u0)
    rhs = deepcopy(u0)

    step = 0
    load_parameters = 0.0:0.05:0.5;
    # tracking iteration history for plotting
    tip_displacement = fill(0.0, length(load_parameters), 2);

    # TODO Newton-Raphson Iterations
    # https://people.duke.edu/~hpgavin/cee421/truss-finite-def.pdf
    # https://kristofferc.github.io/JuAFEM.jl/dev/examples/hyperelasticity/
    for load_parameter in load_parameters
        # guess
        u1[:] = u0[:]
        P = Dict(2 => SVector{2}(load_parameter * P_val * [0.05, -1.0]))

        println("Load: $load_parameter")
        iter = 1;
        while true
            # internal force based on previous configuration
            # assemble K based on the conf (orig + u1)
            updated_nodes = Dict(nid => pt + u1[2*nid-1:2*nid] for (nid, pt) in node_points)
            problem = TrussProblem(Val{:Linear}, updated_nodes, elements, P, fixities, mats, crosssecs);
            solver = FEASolver(Direct, problem)
            # trigger assembly
            solver()

            assemble_k = TopOpt.AssembleK(problem)
            element_k = ElementK(solver)
            truss_element_kσ = TrussElementKσ(problem, solver)

            # reaction force in global coordinate
            Fr = solver.globalinfo.K * u1
            # external force
            F = solver.globalinfo.f
            @. rhs = F - Fr;

            # # ! check geometric stiffness matirx
            # * u_e, x_e -> Ksigma_e
            Kσs_1 = truss_element_kσ(u1, x0)
            Kσs_2 = get_truss_Kσs(problem, u1, solver.elementinfo.cellvalues)

            As = getA(problem)
            Es = getE(problem)
            for cell in CellIterator(getdh(problem))
                cellidx = cellid(cell)
                global_dofs = celldofs(cell)

                coords = getcoordinates(cell)
                L = norm(coords[1] - coords[2])
                A = As[cellidx]
                E = Es[cellidx]

                R2 = compute_local_axes(cell.coords[1], cell.coords[2])
                R4 = zeros(4,4)
                R4[1:2,1:2] = R2
                R4[3:4,3:4] = R2

                γ = vcat(-R2[:,1], R2[:,1])
                u_cell = @view u1[global_dofs]
                q_cell = E*A*(γ'*u_cell/L)

                # ? why MGZ formula different?
                # Kg_m = R4'*(q_cell[2]/L)*[1 0 -1 0; 
                #                            0 1 0 -1; 
                #                            -1 0 1 0; 
                #                            0 -1 0 1]*R4
                Kg_m = R4*(q_cell/L)* [0 0 0 0; 
                                        0 1 0 -1; 
                                        0 0 0 0; 
                                        0 -1 0 1]*R4'
                @test Kg_m ≈ Kσs_1[cellidx]
                @test Kg_m ≈ Kσs_2[cellidx] 
            end

            # solve for the new deformation
            Ke, Kg = buckling(problem, solver.globalinfo, solver.elementinfo; u=u1)
            K = Ke + Kg
            dchi = K\rhs
            u1[:] += dchi[:]

            print("$iter: ||du||=$(maximum(abs.(dchi[:])))\n")
            if maximum(abs.(dchi[:])) < utol # convergence check
                break;
            end
            if (iter > maxit)# bailout for failed convergence
                error("Possible failed convergence");
            end
            iter += 1;
        end
        u0[:] = u1[:];       # update the displacement
        
        step = step + 1
        # save displacement record for plotting
        tip_displacement[step, :] .= u1[2*2-1:2*2]
    end

    # using Makie
    # fig, ax, p = scatter(tip_displacement[:,1], load_parameters.*P_val,
    #     axis = (xlabel = "u_b (m)", ylabel = "P (kN)"))
    # lines!(tip_displacement[:,1], load_parameters.*P_val, color = :blue, linewidth = 3)
    # fig

    # using TopOpt.TrussTopOptProblems.TrussVisualization: visualize
    # fig = visualize(problem, u1)
end # end test set