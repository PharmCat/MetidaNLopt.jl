module MetidaNLopt

    using NLopt, ForwardDiff, LinearAlgebra, Reexport
    @reexport using Metida
    import Metida: LMM, AbstractLMMDataBlocks, LMMDataViews, initvar, thetalength,
    varlinkrvecapply!, varlinkvecapply,
    lmmlog!, LMMLogMsg, fit_nlopt!, rmat_base_inc!, zgz_base_inc!, logerror!, reml_sweep_β

    reml_sweep_β_cuda(args...) = error("MetidaCu not found. \n - Run `using MetidaCu` before.")
    cudata(args...) = error("MetidaCu not found. \n - Run `using MetidaCu` before.")

    function Metida.fit_nlopt!(lmm::LMM{T};
        solver = :nlopt,
        verbose = :auto,
        varlinkf = :exp,
        rholinkf = :sigm,
        aifirst = false,
        g_tol = 1e-16,
        x_tol = 1e-16,
        x_rtol = -Inf,
        f_tol = 1e-16,
        f_rtol = -Inf,
        hes::Bool = false,
        init = nothing,
        io = stdout) where T

        if lmm.result.fit lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Refit model...")) end
        lmm.result.fit = false

        # Optimization function
        if solver == :nlopt
            optfunc = reml_sweep_β_nlopt
            data    = LMMDataViews(lmm)
        elseif solver == :cuda
            optfunc = reml_sweep_β_cuda
            data    = cudata(lmm)
        elseif solver == :nloptsw
            optfunc = reml_sweep_β
            data    = LMMDataViews(lmm)
        else
            error("Unknown solver!")
        end
        if verbose == :auto
            verbose = 1
        end
        ############################################################################
        # Initial variance
        θ  = zeros(T, lmm.covstr.tl)
        lb = similar(θ)
        ub = similar(θ)
        lb .= eps() * 1e4
        ub .= Inf
        if isa(init, Vector{T})
            if length(θ) == length(init)
                copyto!(θ, init)
                lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Using provided θ: "*string(θ)))
            else
                error("init length $(length(init)) != θ length $(length(θ))")
            end
        else
            initθ = sqrt(initvar(lmm.data.yv, lmm.mm.m)[1])/(length(lmm.covstr.random)+1)
            θ                      .= initθ
            for i = 1:length(θ)
                if lmm.covstr.ct[i] == :rho
                    θ[i] = 0.0
                    lb[i] = -1.0 + eps() * 1e4
                    ub[i] =  1.0 - eps() * 1e4
                end
            end
            lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Initial θ: "*string(θ)))
        end
        ############################################################################
        #varlinkrvecapply!(θ, lmm.covstr.ct; rholinkf = rholinkf)
        ############################################################################
        # COBYLA BOBYQA
        opt = NLopt.Opt(:LN_BOBYQA,  thetalength(lmm))
        NLopt.ftol_rel!(opt, f_rtol)
        NLopt.ftol_abs!(opt, f_tol)
        NLopt.xtol_rel!(opt, x_rtol)
        NLopt.xtol_abs!(opt, x_tol)
        opt.lower_bounds = lb
        opt.upper_bounds = ub
        #-----------------------------------------------------------------------
        #obj = (x,y) -> optfunc(lmm, data, varlinkvecapply(x, lmm.covstr.ct; rholinkf = rholinkf))[1]
        obj = (x,y) -> optfunc(lmm, data, x)[1]
        NLopt.min_objective!(opt, obj)
        # Optimization object
        lmm.result.optim = NLopt.optimize!(opt, θ)
        # Theta (θ) vector
        #lmm.result.theta  = varlinkvecapply(lmm.result.optim[2], lmm.covstr.ct)
        lmm.result.theta  = lmm.result.optim[2]
        lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Resulting θ: "*string(lmm.result.theta)))
        # -2 LogREML, β, iC
        lmm.result.reml, lmm.result.beta, iC, θ₃, noerrors = optfunc(lmm, data, lmm.result.theta)
        if !isnan(lmm.result.reml) && !isinf(lmm.result.reml) && noerrors
            # Variance-vovariance matrix of β
            lmm.result.c            = inv(iC)
            # SE
            if  !any(x-> x < 0.0, diag(lmm.result.c))
                lmm.result.se           = sqrt.(diag(lmm.result.c)) #ERROR: DomainError with -1.9121111845919027e-54
                if any(x-> x < 1e-8, lmm.result.se) && minimum(lmm.result.se)/maximum(lmm.result.se) < 1e-8 lmmlog!(io, lmm, verbose, LMMLogMsg(:WARN, "Some of the SE parameters is suspicious.")) end
                lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Model fitted."))
                lmm.result.fit      = true
            end
        end
        # Check G
        if lmm.covstr.random[1].covtype.s != :ZERO
            for i = 1:length(lmm.covstr.random)
                dg = det(gmatrix(lmm, i))
                if dg < 1e-8 lmmlog!(io, lmm, verbose, LMMLogMsg(:WARN, "det(G) of random effect $(i) is less 1e-08.")) end
            end
        end
        # Check Hessian
        if hes && lmm.result.fit
                # Hessian
            lmm.result.h      = hessian(lmm, lmm.result.theta)
                # H positive definite check
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
        #
        if !lmm.result.fit
            lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Model NOT fitted."))
        end
        lmm
    end



    function reml_sweep_β_nlopt(lmm, θ::Vector{T}) where T
        data    = LMMDataViews(lmm)
        reml_sweep_β_nlopt(lmm, data, θ)
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
        βtc           = zeros(T, lmm.rankx)
        β             = Vector{T}(undef, lmm.rankx)
        A             = Vector{Matrix{T}}(undef, n)
        logdetθ₂      = zero(T)
        try
            @inbounds @simd for i = 1:n
                q    = length(lmm.covstr.vcovblock[i])
                qswm = q + lmm.rankx
                V   = zeros(T, q, q)
                zgz_base_inc!(V, θ, lmm.covstr, lmm.covstr.vcovblock[i], lmm.covstr.sblock[i])
                rmat_base_inc!(V, θ[lmm.covstr.tr[end]], lmm.covstr, lmm.covstr.vcovblock[i], lmm.covstr.sblock[i])
            #-------------------------------------------------------------------
            # Cholesky
                A[i] = LinearAlgebra.LAPACK.potrf!('U', V)[1]

                θ₁  += logdet(Cholesky(A[i], 'U', 0))
                #θ₁  += sum(log.(diag(A[i])))*2
                vX   = LinearAlgebra.LAPACK.potrs!('U', A[i], copy(data.xv[i]))
                vy   = LinearAlgebra.LAPACK.potrs!('U', A[i], copy(data.yv[i]))
                mul!(θ₂tc, data.xv[i]', vX, one(T), one(T))
                mul!(βtc, data.xv[i]', vy, one(T), one(T))
            #-------------------------------------------------------------------
            end
        # Beta calculation
            copyto!(θ₂, θ₂tc)
            LinearAlgebra.LAPACK.potrf!('U', θ₂tc)
            copyto!(β, LinearAlgebra.LAPACK.potrs!('U', θ₂tc, βtc))
        # θ₃ calculation
        @inbounds @simd for i = 1:n
            #r    = LinearAlgebra.BLAS.gemv!('N', -one(T), data.xv[i], βtc, one(T), copy(data.yv[i]))
            r    = mul!(copy(data.yv[i]), data.xv[i], βtc, -one(T), one(T))
            vr   = LinearAlgebra.LAPACK.potrs!('U', A[i], copy(r))
            θ₃  += r'*vr
            #θ₃  += BLAS.dot(length(r), r, 1, vr, 1)
        end
        ldθ₂ = LinearAlgebra.LAPACK.potrf!('U', copy(θ₂))[1]
        logdetθ₂ = logdet(Cholesky(ldθ₂, 'U', 0))
        catch e
            logerror!(e, lmm)
            return (Inf, nothing, nothing, nothing, false)
        end
        return   θ₁ + logdetθ₂ + θ₃ + c, β, θ₂, θ₃, true
    end

end # module
