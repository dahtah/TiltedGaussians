function cgf(mv :: MvNormal,qr::QuadRule{D},f,t :: AbstractVector) where D
    n = nnodes(qr)
    wt = 0.0
    w2t = 0.0
    for i in 1:n
        s = unwhiten(mv.Σ,qr.xq[:,i])+mv.μ
        w = qr.wq[i]*f(s)
        wt += w
        w2t += w*exp(dot(s,t))
    end
    log(w2t) - log(wt)
end

function cgf(mvn :: MvNormal,qr::QuadRule{D},A ::AbstractMatrix,f,t :: AbstractVector) where D
    n = nnodes(qr)
    wt  = w2t = 0.0

    #G = A'/(A*mvn.Σ*A')
    Kaa = (A*mvn.Σ*A')
    M = cholesky(Kaa)

    Kba = mvn.Σ*A'
    Kab = Kba'
    mc = A*mvn.μ
    mvnc = MvNormal(mc,Kaa)
    cgf(mvnc,qr,f,M\( Kab*t )) + dot(t, mvn.μ - Kba*(M \ mc)) + .5*dot(t, (mvn.Σ - Kba*(M\Kab))*t)
end

function cgf(mvn :: MvNormal,t :: AbstractVector)
    dot(t,mvn.μ) + .5*dot(t,mvn.Σ*t)
end
#moments of tilted density q(x)f(x)
function moments(f,qr :: QuadRule{D}) where D
    z = 0.0
    m = zeros(D)
    C = zeros(D,D)
    n = nnodes(qr)
    for i in 1:length(qr.wq)
        x = @view qr.xq[:,i]
        s = f(x)*qr.wq[i]
        z += s
        m .+= s*x
        C .+= s*x*x'
    end
    (z=z,m= m/z,C= (C-m*m')/z)
end



#moments of tilted Gaussian
function moments(mv :: MvNormal,qr :: QuadRule{D},f) where D
    z = 0.0
    m = zeros(D)
    C = zeros(D,D)
    n = nnodes(qr)
    x = zeros(D)
    unwhiten!(qr.xbuf,mv.Σ,qr.xq)
    for i in 1:nnodes(qr)
        x .= qr.xbuf[:,i]+mv.μ
        s = f(x)*qr.wq[i]
        z += s
        m .+= s*x
        C .+= s*x*x'
    end
    m = m/z
    C = C/z - m*m'
    (z=z,m=m,C=C)
end

#Compute an expectation against a mv. normal distribution using n points and various approximate methods
function expectation(mvn :: MvNormal,f,n,method = :gh)
    qr = QuadRule(mvn,n,method);
    sum(qr.wq[i]*f(qr.xq[:,i]) for i in 1:nnodes(qr))
end


function tilted_moments(mvn :: MvNormal,qr :: QuadRule{D},A::AbstractMatrix,f) where D
    @assert D == size(A,1)
    @assert length(mvn) == size(A,2)
    mvc = A*mvn
    Kaa = mvc.Σ
    Kba = mvn.Σ*A'
    Kab = Kba'
    G=Kaa\Kab
    mc = mvc.μ
    mm=moments(mvc,qr,f)
    z=mm.z
    m=mvn.μ+G'*(mm.m - mc)
    C=G'*(mm.C-Kaa)*G + mvn.Σ
    (z=z,m=m,C=C)
end

#Computes the contribution of f to the precision and linear shift
#Mostly of interest within EP
function contributions_ep(mvn :: MvNormal,qr :: QuadRule{D},A::AbstractMatrix,f) where D
    @assert D == size(A,1)
    @assert length(mvn) == size(A,2)
    Σm = A*mvn.Σ*A'
    mvc = MvNormal(A*mvn.μ,Σm)
    Qc = inv(Σm)
    rc = Qc*mvc.μ
    mm=moments(mvc,qr,f)
    Qh = inv(mm.C)
    δQ = Qh-Qc
    rh = Qh*mm.m
    δr = rh - rc
    δz = log_partition(Qh,rh)-log_partition(Qc,rc)
    (δz=δz,δr=δr,δQ=δQ)
end


function log_partition(Q :: AbstractMatrix,r :: AbstractVector)
    @assert size(Q,1) == length(r)
    n = length(r)
    C=cholesky(Symmetric(Q))
    .5*(n*log(2π)-logdet(C)+r'*(C\r))
end

