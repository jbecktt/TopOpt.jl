struct Iteration
    i::Int
end

MatrixOf(::Type{Vector{T}}) where T = Matrix{T}
MatrixOf(::Type{CuVector{T}}) where T = CuMatrix{T}
zerosof(::Type{TM}, n...) where TM = (TM(undef, n...) .= 0)
onesof(::Type{TM}, n...) where TM = (TM(undef, n...) .= 1)
infsof(::Type{TM}, n...) where TM = (TM(undef, n...) .= Inf)
ninfsof(::Type{TM}, n...) where TM = (TM(undef, n...) .= -Inf)
nansof(::Type{TM}, n...) where TM = (TM(undef, n...) .= NaN)

@inline minus_plus(a, b) = a - b, a + b

@inline or(a,b) = a || b

macro matdot(v, A, j)
    r = gensym()
    T = gensym()
    esc(quote
        $T = promote_type(eltype($v), eltype($A))
        $r = zero($T)
        for i in 1:length($v)
            $r += $v[i] * $A[$j, i]
        end
        $r
    end)
end

function check_error(m, x0)
    if length(x0) != dim(m)
        throw(ArgumentError("initial variable must have same length as number of design variables"))
    end

    Threads.@threads for j in 1:length(x0)
        # x is not in box
        if !(min(m, j) <= x0[j] <= max(m,j))
            throw(ArgumentError("initial variable at index $j outside box constraint"))
        end
    end
end

# From Optim.jl
function assess_convergence(x::AbstractArray{T},
                            x_previous::AbstractArray,
                            f_x::Real,
                            f_x_previous::Real,
                            gr::AbstractArray,
                            xtol::Real,
                            ftol::Real,
                            grtol::Real) where {T}
    
    x_converged, f_converged, gr_converged, f_increased = false, false, false, false

    if x isa CuArray
        x_residual = mapreduce((x1, x2) -> abs(x1 - x2), max, x, x_previous, init=zero(T))
    else
        x_residual = maxdiff(x, x_previous)
    end
    f_residual = abs(f_x - f_x_previous)
    gr_residual = maximum(abs, gr)

    if x_residual < xtol
        x_converged = true
    end

    # Absolute Tolerance
    # if abs(f_x - f_x_previous) < ftol
    # Relative Tolerance
    if f_residual / (abs(f_x) + ftol) < ftol
        f_converged = true
    end

    if f_x > f_x_previous
        f_increased = true
    end

    if gr_residual < grtol
        gr_converged = true
    end

    converged = x_converged || f_converged || gr_converged

    return ConvergenceState(    x_converged, 
                                f_converged, 
                                gr_converged, 
                                x_residual, 
                                f_residual, 
                                gr_residual, 
                                f_increased, 
                                converged
                            )
end
