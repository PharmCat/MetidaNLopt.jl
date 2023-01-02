module MetidaNLopt

    using NLopt, LinearAlgebra, Reexport
    @reexport using Metida
    import Metida: ForwardDiff, LMM, AbstractLMMDataBlocks, LMMDataViews, initvar, thetalength,
    varlinkrvecapply!, varlinkvecapply, num_cores, varlinkvecapply!, diag!,
    lmmlog!, LMMLogMsg, fit_nlopt!, reml_sweep_β, reml_sweep_β_nlopt, vmatrix!, gmatvec

    reml_sweep_β_cuda(args...) = error("MetidaCu not found. \n - Run `using MetidaCu` before.")
    cudata(args...) = error("MetidaCu not found. \n - Run `using MetidaCu` before.")

    function Metida.fit_nlopt!(lmm::LMM{T}; kwargs...) where T

        kwkeys = keys(kwargs)

        :solver ∈ kwkeys ? solver = kwargs[:solver] : solver = :nlopt
        :verbose ∈ kwkeys ? verbose = kwargs[:verbose] : verbose = :auto
        :varlinkf ∈ kwkeys ? varlinkf = kwargs[:varlinkf] : varlinkf = :exp
        :rholinkf ∈ kwkeys ? rholinkf = kwargs[:rholinkf] : rholinkf = :sigm
        :aifirst ∈ kwkeys ? aifirst = kwargs[:aifirst] : aifirst = false
        :aifmax ∈ kwkeys ? aifmax = kwargs[:aifmax] : aifmax = 10
        :g_tol ∈ kwkeys ? g_tol = kwargs[:g_tol] : g_tol = 1e-10
        :x_tol ∈ kwkeys ? x_tol = kwargs[:x_tol] : x_tol = 1e-10
        :f_tol ∈ kwkeys ? f_tol = kwargs[:f_tol] : f_tol = 1e-10
        :x_rtol ∈ kwkeys ? x_rtol = kwargs[:x_rtol] : x_rtol = 0.0
        :f_rtol ∈ kwkeys ? f_rtol = kwargs[:f_rtol] : f_rtol = 0.0
        :hes ∈ kwkeys ? hes = kwargs[:hes] : hes = true
        :init ∈ kwkeys ? init = kwargs[:init] : init = :nothing
        :io ∈ kwkeys ? io = kwargs[:io] : io = stdout
        :time_limit ∈ kwkeys ? time_limit = kwargs[:time_limit] : time_limit = 0
        #:iterations ∈ kwkeys ? iterations = kwargs[:iterations] : iterations = 300
        :refitinit ∈ kwkeys ? refitinit = kwargs[:refitinit] : refitinit = false
        :optmethod ∈ kwkeys ? optmethod = kwargs[:optmethod] : optmethod = :LN_BOBYQA
        :singtol ∈ kwkeys ? singtol = kwargs[:singtol] : singtol = 1e-8
        :maxthreads ∈ kwkeys ? maxthreads = kwargs[:maxthreads] : maxthreads = num_cores()
        :dopt ∈ kwkeys ? dopt = kwargs[:dopt] : dopt = :LN_BOBYQA
        :istepm ∈ kwkeys ? istepm = kwargs[:istepm] : istepm = 0.001
        :sstepm ∈ kwkeys ? sstepm = kwargs[:sstepm] : sstepm = 0.00001

        if lmm.result.fit
            if length(lmm.log) > 0 deleteat!(lmm.log, 1:length(lmm.log)) end
            lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Refit model..."))
            if refitinit
                lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Using previous initial parameters."))
                init = lmm.result.theta
            end
        end
        lmm.result.fit = false
        # Optimization function
        if solver == :nlopt
            optfunc = reml_sweep_β_nlopt
            data    = lmm.dv
        elseif solver == :cuda
            optfunc = reml_sweep_β_cuda
            data    = cudata(lmm)
        elseif solver == :nloptsw
            optfunc = reml_sweep_β
            data    = lmm.dv
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
        lb .= eps()^2
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
            θ .= initθ
            for i = 1:length(θ)
                if lmm.covstr.ct[i] == :rho
                    θ[i]  = 0.1
                elseif lmm.covstr.ct[i] == :theta
                    θ[i]  = 1.0
                    lb[i] = -Inf
                end
            end
            lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Initial θ: "*string(θ)))
        end
        for i = 1:length(θ)     
            if lmm.covstr.ct[i] == :rho
                lb[i]    = -Inf#-1.0 + eps()
                ub[i]    =  Inf#1.0 - eps()     
            end
        end
        ############################################################################
        ############################################################################
        # :LN_COBYLA :LN_NELDERMEAD :LN_SBPLX  :LD_MMA  :LD_CCSAQ :LD_SLSQP :LD_LBFGS
        # :LN_PRAXIS :LD_VAR1 :AUGLAG
        # :LN_BOBYQA :LN_NEWUOA 
        opt = NLopt.Opt(optmethod, thetalength(lmm))
        NLopt.ftol_rel!(opt, f_rtol)
        NLopt.ftol_abs!(opt, f_tol)
        NLopt.xtol_rel!(opt, x_rtol)
        NLopt.xtol_abs!(opt, x_tol)
        opt.lower_bounds = lb
        opt.upper_bounds = ub
        opt.maxtime      = time_limit
        istep = similar(θ)
        varlinkrvecapply!(θ, lmm.covstr.ct; varlinkf = varlinkf, rholinkf = rholinkf)
        for i = 1:length(θ)     
            if lmm.covstr.ct[i] == :var
                istep[i]    = istepm*θ[i]
            elseif lmm.covstr.ct[i] == :rho
                istep[i]    = istepm*θ[i]
            elseif lmm.covstr.ct[i] == :theta
                istep[i]    = istepm*θ[i]
            end
        end
        opt.initial_step = istep
        counter = Dict{Symbol, Int}(:iter => 0)
        #-----------------------------------------------------------------------
        obj(x, ::Any) = begin
            if length(g) > 0
                error("Gradient not defined!")
            end
            val = optfunc(lmm, data, varlinkvecapply!(x, lmm.covstr.ct; varlinkf = varlinkf, rholinkf = rholinkf); maxthreads = maxthreads)[1] 
            if val == Inf 
                lmmlog!(io, lmm, verbose, LMMLogMsg(:WARN, "Unstable values during optimization. Check results."))
            end
            counter[:iter] += 1
            val
        end
        NLopt.min_objective!(opt, obj)
        # Optimization object
        lmm.result.optim = NLopt.optimize!(opt, θ)
        # Second step
        if isa(dopt, Symbol)
            copyto!(θ, lmm.result.optim[2])
            opt = NLopt.Opt(dopt,  thetalength(lmm))
            NLopt.ftol_rel!(opt, f_rtol)
            NLopt.ftol_abs!(opt, f_tol)
            NLopt.xtol_rel!(opt, x_rtol)
            NLopt.xtol_abs!(opt, x_tol)
            opt.lower_bounds = lb
            opt.upper_bounds = ub
            opt.maxtime      = time_limit
            opt.initial_step = lmm.result.optim[2] .* sstepm
            NLopt.min_objective!(opt, obj)
            lmm.result.optim = NLopt.optimize!(opt, θ)
        end
        varlinkvecapply!(lmm.result.optim[2], lmm.covstr.ct; varlinkf = varlinkf, rholinkf = rholinkf)
        # Theta (θ) vector
        lmm.result.theta  = lmm.result.optim[2]
        lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Resulting θ: "*string(lmm.result.theta)*", Iterations:$(counter[:iter])"))
        # -2 LogREML, β, iC
        lmm.result.reml, lmm.result.beta, iC, θ₃, noerrors = optfunc(lmm, data, lmm.result.theta)
        if !noerrors LMMLogMsg(:ERROR, "The last optimization step wasn't accurate. Results can be wrong!") end
        # Fit true
        if !isnan(lmm.result.reml) && !isinf(lmm.result.reml) && rank(iC) == size(iC, 1)
            # Variance-vovariance matrix of β
            copyto!(lmm.result.c, inv(iC))
            # SE
            if  !any(x -> x < 0.0, diag(lmm.result.c))
                diag!(sqrt, lmm.result.se, lmm.result.c)
                if any(x-> x < singtol, lmm.result.se) && minimum(lmm.result.se)/maximum(lmm.result.se) < singtol lmmlog!(io, lmm, verbose, LMMLogMsg(:WARN, "Some of the SE parameters is suspicious.")) end
                lmmlog!(io, lmm, verbose, LMMLogMsg(:INFO, "Model fitted."))
                lmm.result.fit      = true
            else
                lmmlog!(io, lmm, verbose,LMMLogMsg(:ERROR, "Some variance less zero: $(diag(lmm.result.c))."))
            end
        else
            lmmlog!(io, lmm, verbose, LMMLogMsg(:ERROR, "REML not estimated or final iteration completed with errors."))
        end

        # Check G
        lmm.result.ipd = true
        if !isa(lmm.covstr.random[1].covtype.s, ZERO)
            for i = 1:length(lmm.covstr.random)
                if isposdef(gmatrix(lmm, i)) == false
                    lmm.result.ipd =  false
                    lmmlog!(io, lmm, verbose, LMMLogMsg(:WARN, "Variance-covariance matrix (G) of random effect $(i) is not positive definite."))
                end
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
            qrd = qr(lmm.result.h)
            for i = 1:length(lmm.result.theta)
                if abs(qrd.R[i,i]) < singtol
                    lmmlog!(io, lmm, verbose, LMMLogMsg(:WARN, "Hessian parameter ($(lmm.covstr.ct[i])) QR.R diagonal value ($(i)) is less than $(singtol)."))
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
end # module
