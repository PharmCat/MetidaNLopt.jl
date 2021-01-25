module MetidaNLopt

    using Metida, NLopt, ForwardDiff, LinearAlgebra
    import Metida: LMM, initvar, thetalength,
    varlinkrvecapply!, varlinkvecapply,
    lmmlog!, LMMLogMsg, reml_sweep_β, fit_nlopt!, rmat_base_inc!, zgz_base_inc!

    reml_sweep_β_cuda(args...) = error("MetidaCu not found. \n - Run `using MetidaCu` before.")

    function Metida.fit_nlopt!(lmm::LMM{T};
        solver = :nlopt,
        verbose = :auto,
        varlinkf = :exp,
        rholinkf = :sigm,
        aifirst = false,
        g_tol = 1e-12,
        x_tol = 1e-12,
        f_tol = 1e-12,
        hcalck::Bool = false,
        init = nothing) where T
        # Optimization function
        if solver == :nlopt
            optfunc = reml_sweep_β_nlopt
        elseif solver == :cuda
            optfunc = reml_sweep_β_cuda
        else
            error("Unknown solver!")
        end
        ############################################################################
        #Initial variance
        θ  = zeros(T, lmm.covstr.tl)
        if isa(init, Vector{T}) && length(θ) == length(init)
            copyto!(θ, init)
        else
            initθ = sqrt(initvar(lmm.mf.data[lmm.mf.f.lhs.sym], lmm.mm.m)[1]/4)
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
        #NLopt.ftol_rel!(opt, 1.0e-14)
        NLopt.ftol_abs!(opt, f_tol)
        #NLopt.xtol_rel!(opt, 1.0e-14)
        NLopt.xtol_abs!(opt, x_tol)
        #opt.lower_bounds = lb::Union{AbstractVector,Real}
        #opt.upper_bounds = ub::Union{AbstractVector,Real}
        #-----------------------------------------------------------------------
        obj = (x,y) -> optfunc(lmm, varlinkvecapply(x, lmm.covstr.ct))[1]
        NLopt.min_objective!(opt, obj)
        #Optimization object
        lmm.result.optim = NLopt.optimize!(opt, θ)
        #Theta (θ) vector
        lmm.result.theta  = varlinkvecapply(lmm.result.optim[2], lmm.covstr.ct)
        try
            #Hessian
            if hcalck
                lmm.result.h      = ForwardDiff.hessian(x -> reml_sweep_β(lmm, x)[1], lmm.result.theta)
                qrd = qr(lmm.result.h, Val(true))
                for i = 1:length(lmm.result.theta)
                    if abs(qrd.R[i,i]) < 1E-10
                        if lmm.covstr.ct[qrd.jpvt[i]] == :var
                            lmmlog!(lmm, verbose, LMMLogMsg(:WARN, "Variation QR.R diagonal value ($(qrd.jpvt[i])) is less than 1e-10."))
                        elseif lmm.covstr.ct[qrd.jpvt[i]] == :rho
                            lmmlog!(lmm, verbose, LMMLogMsg(:WARN, "Rho QR.R diagonal value ($(qrd.jpvt[i])) is less than 1e-10."))
                        end
                    end
                end
            end
            #-2 LogREML, β, iC
            lmm.result.reml, lmm.result.beta, iC, θ₃ = optfunc(lmm, lmm.result.theta)
            #Variance-vovariance matrix of β
            lmm.result.c            = pinv(iC)
            #SE
            lmm.result.se           = sqrt.(diag(lmm.result.c))
            #Fit true
            lmm.result.fit          = true
        catch
            #-2 LogREML, β, iC
            lmm.result.reml, lmm.result.beta, iC, θ₃ = optfunc(lmm, lmm.result.theta)
            #Fit false
            lmm.result.fit          = false
        end
        lmm
    end

    function reml_sweep_β_nlopt(lmm, θ::Vector{T}) where T
        n             = length(lmm.data.block)
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
        A             = Vector{Matrix}(undef, n)
        X             = Vector{Matrix}(undef, n)
        y             = Vector{Vector}(undef, n)
        q             = zero(Int)
        qswm          = zero(Int)
        logdetθ₂      = zero(T)
        @inbounds for i = 1:n
            q    = length(lmm.data.block[i])
            qswm = q + lmm.rankx
            V   = zeros(T, q, q)
            zgz_base_inc!(V, θ, lmm.covstr, lmm.data.block[i], lmm.covstr.sblock[i])
            rmat_base_inc!(V, θ[lmm.covstr.tr[end]], lmm.covstr, lmm.data.block[i], lmm.covstr.sblock[i])
            #-------------------------------------------------------------------
            X[i] = view(lmm.data.xv,  lmm.data.block[i], :)
            y[i] = view(lmm.data.yv, lmm.data.block[i])
            #-------------------------------------------------------------------
            #Cholesky
            A[i] = LinearAlgebra.LAPACK.potrf!('L', V)[1]
            θ₁  += logdet(Cholesky(A[i], 'L', 0))
            vX   = LinearAlgebra.LAPACK.potrs!('L', A[i], copy(X[i]))
            vy   = LinearAlgebra.LAPACK.potrs!('L', A[i], copy(y[i]))
            LinearAlgebra.BLAS.gemm!('T', 'N', one(T), X[i], vX, one(T), θ₂tc)
            LinearAlgebra.BLAS.gemv!('T', one(T), X[i], vy, one(T), βtc)
            #-------------------------------------------------------------------
        end
        #Beta calculation
        copyto!(θ₂, θ₂tc)
        LinearAlgebra.LAPACK.potrf!('L', θ₂tc)
        copyto!(β, LinearAlgebra.LAPACK.potrs!('L', θ₂tc, βtc))
        # θ₃ calculation
        @simd for i = 1:n
            r = LinearAlgebra.BLAS.gemv!('N', -one(T), X[i], βtc,
            one(T), y[i])
            vr   = LinearAlgebra.LAPACK.potrs!('L', A[i], copy(r))
            θ₃  += r'*vr
        end
        logdetθ₂ = logdet(θ₂)
        return   θ₁ + logdetθ₂ + θ₃ + c, β, θ₂, θ₃
    end
end # module
