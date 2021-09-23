module MetidaNLopt

    using NLopt, LinearAlgebra, Reexport
    @reexport using Metida
    import Metida: ForwardDiff, LMM, AbstractLMMDataBlocks, LMMDataViews, initvar, thetalength,
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
        refitinit = false,
        io = stdout) where T

        if lmm.result.fit
            if length(lmm.log) > 0 deleteat!(lmm.log, 1:length(lmm.log)) end
            lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Refit model..."))
            if refitinit
                lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Using previous initial parameters."))
                init = lmm.result.theta
            end
        end
        lmm.result.fit = false
        qrdrlim = 1e-8

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
        obj = (x,y) -> optfunc(lmm, data, x)[1]
        NLopt.min_objective!(opt, obj)
        # Optimization object
        lmm.result.optim = NLopt.optimize!(opt, θ)
        # Theta (θ) vector
        lmm.result.theta  = lmm.result.optim[2]
        lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Resulting θ: "*string(lmm.result.theta)))
        # -2 LogREML, β, iC
        lmm.result.reml, lmm.result.beta, iC, θ₃, noerrors = optfunc(lmm, data, lmm.result.theta)
        if !noerrors LMMLogMsg(:ERROR, "The last optimization step wasn't accurate. Results can be wrong!") end
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
                if abs(qrd.R[i,i]) < qrdrlim
                    if lmm.covstr.ct[qrd.jpvt[i]] == :var
                        lmmlog!(io, lmm, verbose, LMMLogMsg(:WARN, "Hessian parameter (variation) QR.R diagonal value ($(qrd.jpvt[i])) is less than $qrdrlim."))
                    elseif lmm.covstr.ct[qrd.jpvt[i]] == :rho
                        lmmlog!(io, lmm, verbose, LMMLogMsg(:WARN, "Hessian parameter (rho) QR.R diagonal value ($(qrd.jpvt[i])) is less than $qrdrlim."))
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

    function logdet_(C::Cholesky)
        dd = zero(real(eltype(C)))
        noerror = true
        @inbounds for i in 1:size(C.factors,1)
            v = real(C.factors[i,i])
            if v > 0
                dd += log(v)
            else
                C.factors[i,i] *= -1e-8
                dd += log(real(C.factors[i,i]+4eps()))
                noerror = false
            end
        end
        dd + dd, noerror
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
        noerror       = true
            l = Base.Threads.SpinLock()
            #l = Base.Threads.ReentrantLock()
            @inbounds Base.Threads.@threads for i = 1:n
                q    = length(lmm.covstr.vcovblock[i])
                qswm = q + lmm.rankx
                V   = zeros(T, q, q)
                zgz_base_inc!(V, θ, lmm.covstr, lmm.covstr.vcovblock[i], lmm.covstr.sblock[i])
                rmat_base_inc!(V, θ[lmm.covstr.tr[end]], lmm.covstr, lmm.covstr.vcovblock[i], lmm.covstr.sblock[i])
            #-------------------------------------------------------------------
            # Cholesky
                A[i], info = LinearAlgebra.LAPACK.potrf!('U', V)
                vX   = LinearAlgebra.LAPACK.potrs!('U', A[i], copy(data.xv[i]))
                vy   = LinearAlgebra.LAPACK.potrs!('U', A[i], copy(data.yv[i]))
                # Check matrix and make it avialible for logdet computation
                if info == 0
                    θ₁ld = logdet(Cholesky(A[i], 'U', 0))
                    ne = true
                else
                    θ₁ld, ne = logdet_(Cholesky(A[i], 'U', 0))
                end
                lock(l) do
                    if ne == false noerror = false end
                    θ₁  += θ₁ld
                    mul!(θ₂tc, data.xv[i]', vX, one(T), one(T))
                    mul!(βtc, data.xv[i]', vy, one(T), one(T))
                end
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
            logdetθ₂, ne = logdet_(Cholesky(ldθ₂, 'U', 0))
            if ne == false noerror = false end
        return   θ₁ + logdetθ₂ + θ₃ + c, β, θ₂, θ₃, noerror
    end

end # module
