import GaussQuadrature,HaltonSequences

struct QuadRule{D}
    xq :: Matrix{Float64}
    wq :: Vector{Float64}
    xbuf :: Matrix{Float64} # To be used as buffer in computations
end


function nnodes(qr :: QuadRule{D}) where D
    length(qr.wq)
end

function setmv(qr :: QuadRule{D},m,C) where D
    QuadRule{D}(cholesky(C).L*qr.xq  .+ m,qr.wq)
end

iter_pow(it::It, ::Val{n}) where {It <: Any, n} =
    Iterators.product(
        ntuple(
            let it = it
                i -> it
            end,
            Val(n))...)


function phiinv(x)
    quantile(Normal(0,1),x)
end

function QuadRule{D}(xq :: Matrix{Float64},wq :: Vector{Float64}) where D
    QuadRule{D}(xq,wq,similar(xq))
end


function QuadRule(d :: Int, n :: Int, method = :mc)
    if (method ∈ [:mc,:MonteCarlo])
        xq = randn(d,n)
        wq = fill(1/n,n)
        QuadRule{d}(xq,wq)
    elseif (method ∈ [:hs,:Halton])
        xq=phiinv.(reduce(hcat,HaltonSequences.HaltonPoint(d)[1:n]))
        wq = fill(1/n,n)
        QuadRule{d}(xq,wq)
    elseif (method ∈ [:gh,:GaussHermite])
        nd1=Int(floor(n^(1/d))) #number of points of unidimensional rule
        x,w = GaussQuadrature.hermite(nd1)
        x *= sqrt(2)
        w /= sqrt(π)
        xq = reduce(hcat,map(collect,iter_pow(x,Val(d))))
        wq = [prod(collect(v)) for v in iter_pow(w,Val(d))][:]
        QuadRule{d}(xq,wq)
    else
        error("Method not supported. Supported methods are Monte Carlo (:mc), Halton (:hs), and tensorial Gauss-Hermite (:gh).")
    end
end

function QuadRule(nrm :: Normal,n,method=:mc)
    d = 1
    qr = QuadRule(d,n,method)
    for i in 1:nnodes(qr)
        qr.xq[i] = nrm.σ*qr.xq[i] + nrm.μ
    end
    qr
end


function QuadRule(mv :: MvNormal,n,method=:mc)
    d = length(mv)
    qr = QuadRule(d,n,method)
    for i in 1:nnodes(qr)
         qr.xq[:,i] = unwhiten(mv.Σ,qr.xq[:,i]) + mv.μ
    end
    qr
end




# function mgf(f,qr::QuadRule{D},t :: AbstractVector) where D
# #    z = 0.0
#     n = nnodes(qr)
#     s = [f(qr.xq[:,i])*qr.wq[i] for i in 1:n]
#     h = map((v)->exp(dot(t,v)), eachcol(qr.xq))
#     dot(s,h)
# end

# function cgf(f,qr::QuadRule{D},t :: AbstractVector) where D
#     log(mgf(f,qr,t))
# end





# function moments(f,qr :: QuadRule{D},R :: AbstractMatrix,b :: AbstractVector) where D
#     z = 0.0
#     m = zeros(D)
#     C = zeros(D,D)
#     n = nnodes(qr)
#     x = zeros(D)
#     for i in 1:length(qr.wq)
#         x .= R*qr.xq[:,i] + b
#         s = f(x)*qr.wq[i]
#         z += s
#         m .+= s*x
#         C .+= s*x*x'
#     end
#     m = m/z
#     (z=z,m= m,C= (C/z-m*m'))
# end



#Moments of density f(A*x)q(x) where q(x) is N(0,I)
# function moments(f,A :: AbstractMatrix,qr :: QuadRule{D}) where D
#     ms = moments(f,qr,cholesky(A*A').L,zeros(D))
#     @assert size(A,1) == D
#     G = A'/(A*A')
#     (z=ms.z,m=G*ms.m,C=I-G*A + G*ms.C*G')
# end

#Compute the moments of the following tilted distribution:
#q(x)*f(Ax)
#where q(x) is N(m,C)
#and A is a d × length(x) matrix
# function moments(f,A :: AbstractMatrix,qr :: QuadRule{D},m,C) where D
#     L = cholesky(C).L
#     c = A*m
#     ms = moments(v->f(c+v),A*L,qr)
#     (z=ms.z,m=m+L*ms.m,C=L*ms.C*L')
# end




