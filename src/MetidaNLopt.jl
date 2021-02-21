module MetidaNLopt

    using NLopt, ForwardDiff, LinearAlgebra, Reexport
    @reexport using Metida
    import Metida: LMM, AbstractLMMDataBlocks, LMMDataViews, initvar, thetalength,
    varlinkrvecapply!, varlinkvecapply,
    lmmlog!, LMMLogMsg, fit_nlopt!, rmat_base_inc!, zgz_base_inc!

    reml_sweep_β_cuda(args...) = error("MetidaCu not found. \n - Run `using MetidaCu` before.")
    cudata(args...) = error("MetidaCu not found. \n - Run `using MetidaCu` before.")

    #=
    struct LMMDataBlocks{T1, T2} <: AbstractLMMDataBlocks
        # Fixed effect matrix views
        xv::T1
        # Responce vector views
        yv::T2
        function LMMDataBlocks(xv::Matrix, yv::Vector, vcovblock::Vector)
            x = Vector{typeof(xv)}(undef, length(vcovblock))
            y = Vector{typeof(yv)}(undef, length(vcovblock))
            for i = 1:length(vcovblock)
                x[i] = Matrix(view(xv, vcovblock[i],:))
                y[i] = Vector(view(yv, vcovblock[i]))
            end
            new{typeof(x), typeof(y)}(x, y)
        end
        function LMMDataBlocks(lmm)
            return LMMDataBlocks(lmm.data.xv, lmm.data.yv, lmm.covstr.vcovblock)
        end
    end
    =#
    function Metida.fit_nlopt!(lmm::LMM{T};
        solver = :nlopt,
        verbose = :auto,
        varlinkf = :exp,
        rholinkf = :sigm,
        aifirst = false,
        g_tol = 1e-12,
        x_tol = 1e-12,
        f_tol = 1e-12,
        hes::Bool = false,
        init = nothing,
        io = stdout) where T
        # Optimization function
        if solver == :nlopt
            optfunc = reml_sweep_β_nlopt
            data    = LMMDataViews(lmm)
        elseif solver == :cuda
            optfunc = reml_sweep_β_cuda
            data    = cudata(lmm)
        else
            error("Unknown solver!")
        end
        if verbose == :auto
            verbose = 1
        end
        ############################################################################
        #Initial variance
        θ  = zeros(T, lmm.covstr.tl)
        if isa(init, Vector{T}) && length(θ) == length(init)
            copyto!(θ, init)
        else
            initθ = sqrt(initvar(lmm.data.yv, lmm.mm.m)[1]/4)
            θ                      .= initθ
            for i = 1:length(θ)
                if lmm.covstr.ct[i] == :rho
                    θ[i] = 0.0
                end
            end
            lmmlog!(lmm, verbose, LMMLogMsg(:INFO, "Initial θ: "*string(θ)))
        end
        ############################################################################
        varlinkrvecapply!(θ, lmm.covstr.ct)
        ############################################################################
        #COBYLA
        opt = NLopt.Opt(:LN_BOBYQA,  thetalength(lmm))
        NLopt.ftol_rel!(opt, 0.0)
        NLopt.ftol_abs!(opt, f_tol)
        NLopt.xtol_rel!(opt, 0.0)
        NLopt.xtol_abs!(opt, x_tol)
        #opt.lower_bounds = lb::Union{AbstractVector,Real}
        #opt.upper_bounds = ub::Union{AbstractVector,Real}
        #-----------------------------------------------------------------------
        obj = (x,y) -> optfunc(lmm, data, varlinkvecapply(x, lmm.covstr.ct; rholinkf = rholinkf))[1]
        NLopt.min_objective!(opt, obj)
        #Optimization object
        lmm.result.optim = NLopt.optimize!(opt, θ)
        #Theta (θ) vector
        lmm.result.theta  = varlinkvecapply(lmm.result.optim[2], lmm.covstr.ct)
        lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Resulting θ: "*string(lmm.result.theta)))

        #-2 LogREML, β, iC
        lmm.result.reml, lmm.result.beta, iC, θ₃ = optfunc(lmm, data, lmm.result.theta)
        if !isnan(lmm.result.reml) && !isinf(lmm.result.reml)
            #Variance-vovariance matrix of β
            lmm.result.c            = pinv(iC)
            #SE
            lmm.result.se           = sqrt.(diag(lmm.result.c))
            lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Model fitted."))
            lmm.result.fit      = true
        else
            lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Model NOT fitted."))
            lmm.result.fit      = false
        end
        if hes && lmm.result.fit
                #Hessian
            lmm.result.h      = hessian(lmm, lmm.result.theta)
                #H positive definite check
            if !isposdef(Symmetric(lmm.result.h))
                lmmlog!(io, lmm, verbose, LMMLogMsg(:WARN, "Hessian is not positive definite."))
            end
            qrd = qr(lmm.result.h, Val(true))
            for i = 1:length(lmm.result.theta)
                if abs(qrd.R[i,i]) < 1E-8
                    if lmm.covstr.ct[qrd.jpvt[i]] == :var
                        lmmlog!(io, lmm, verbose, LMMLogMsg(:WARN, "Hessian parameter (variation) QR.R diagonal value ($(qrd.jpvt[i])) is less than 1e-10."))
                    elseif lmm.covstr.ct[qrd.jpvt[i]] == :rho
                        lmmlog!(io, lmm, verbose, LMMLogMsg(:WARN, "Hessian parameter (rho) QR.R diagonal value ($(qrd.jpvt[i])) is less than 1e-10."))
                    end
                end
            end
        end
        lmm
    end

    function reml_sweep_β_nlopt(lmm, data::AbstractLMMDataBlocks, θ::Vector{T}) where T
        n             = length(lmm.covstr.vcovblock)
        N             = length(lmm.data.yv)
        c             = (N - lmm.rankx)*log(2π)
        #---------------------------------------------------------------------------
        θ₁            = zero(T)
        θ₂            = zeros(T, lmm.rankx, lmm.rankx)
        θ₂tc          = zeros(T, lmm.rankx, lmm.rankx)
        θ₃            = zero(T)
        #βm            = zeros(T, lmm.rankx)
        βtc           = zeros(T, lmm.rankx)
        β             = Vector{T}(undef, lmm.rankx)
        A             = Vector{Matrix{T}}(undef, n)
        #X             = Vector{Matrix}(undef, n)
        #y             = Vector{Vector}(undef, n)
        q             = zero(Int)
        qswm          = zero(Int)
        logdetθ₂      = zero(T)
        @inbounds for i = 1:n
            q    = length(lmm.covstr.vcovblock[i])
            qswm = q + lmm.rankx
            V   = zeros(T, q, q)
            zgz_base_inc!(V, θ, lmm.covstr, lmm.covstr.vcovblock[i], lmm.covstr.sblock[i])
            rmat_base_inc!(V, θ[lmm.covstr.tr[end]], lmm.covstr, lmm.covstr.vcovblock[i], lmm.covstr.sblock[i])
            #-------------------------------------------------------------------
            #X[i] = view(lmm.data.xv,  lmm.covstr.vcovblock[i], :)
            #y[i] = view(lmm.data.yv, lmm.covstr.vcovblock[i])
            #-------------------------------------------------------------------
            #Cholesky
            A[i] = LinearAlgebra.LAPACK.potrf!('L', V)[1]
            try
                θ₁  += logdet(Cholesky(A[i], 'L', 0))
                #θ₁  += sum(log.(diag(A[i])))*2
            catch
                lmmlog!(lmm, LMMLogMsg(:ERROR, "θ₁ not estimated during REML calculation, V isn't positive definite or |V| less zero."))
                return (1e100, nothing, nothing, 1e100)
            end
            vX   = LinearAlgebra.LAPACK.potrs!('L', A[i], copy(data.xv[i]))
            vy   = LinearAlgebra.LAPACK.potrs!('L', A[i], copy(data.yv[i]))
            #LinearAlgebra.BLAS.gemm!('T', 'N', one(T), data.xv[i], vX, one(T), θ₂tc)
            mul!(θ₂tc, data.xv[i]', vX, one(T), one(T))
            #LinearAlgebra.BLAS.gemv!('T', one(T), data.xv[i], vy, one(T), βtc)
            mul!(βtc, data.xv[i]', vy, one(T), one(T))
            #-------------------------------------------------------------------
        end
        #Beta calculation
        copyto!(θ₂, θ₂tc)
        LinearAlgebra.LAPACK.potrf!('L', θ₂tc)
        copyto!(β, LinearAlgebra.LAPACK.potrs!('L', θ₂tc, βtc))
        # θ₃ calculation
        @simd for i = 1:n
            #r    = LinearAlgebra.BLAS.gemv!('N', -one(T), data.xv[i], βtc, one(T), copy(data.yv[i]))
            r    = mul!(copy(data.yv[i]), data.xv[i], βtc, -one(T), one(T))
            vr   = LinearAlgebra.LAPACK.potrs!('L', A[i], copy(r))
            θ₃  += r'*vr
            #θ₃  += BLAS.dot(length(r), r, 1, vr, 1)
        end
        logdetθ₂ = logdet(θ₂)
        return   θ₁ + logdetθ₂ + θ₃ + c, β, θ₂, θ₃
    end
end # module
