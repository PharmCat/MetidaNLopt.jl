# MetidaNLopt
# Copyright © 2019-2020 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>
using Metida
using MetidaNLopt
using  Test, CSV, DataFrames, StatsModels, CategoricalArrays

df0 = CSV.File(joinpath(dirname(pathof(Metida)),"..","test","csv","df0.csv"); types = [String, String, String, String, Float64, Float64]) |> DataFrame
transform!(df0, :subject => categorical, renamecols=false)
transform!(df0, :period => categorical, renamecols=false)
transform!(df0, :sequence => categorical, renamecols=false)
transform!(df0, :formulation=> categorical, renamecols=false)

ftdf         = CSV.File(joinpath(dirname(pathof(Metida)),"..","test","csv","1fptime.csv"); types = [String, String, Float64, Float64]) |> DataFrame

ftdf2        = CSV.File(joinpath(dirname(pathof(Metida)),"..","test","csv","1freparma.csv"); types = [String, String, Float64, Float64]) |> DataFrame

ftdf3        = CSV.File(joinpath(dirname(pathof(Metida)),"..","test","csv","2f2rand.csv"); types =
[String,  Float64, Float64, String, String, String, String, String]) |> DataFrame

@testset "  Basic test                                               " begin
    lmm = LMM(@formula(var~sequence+period+formulation), df0;
    random = VarEffect(@covstr(formulation|subject), CSH),
    repeated = VarEffect(@covstr(formulation|subject), DIAG),
    )
    fit!(lmm; solver = :nlopt, f_tol=1e-16, x_tol=1e-16)
    #Metida.fit_nlopt!(lmm; solver = :nlopt, rholinkf = :sigm, f_tol=0.0, x_tol = 0.0, f_rtol =0.0, x_rtol =1e-18)
    #Metida.m2logreml(lmm) ≈ 10.065239006121315
    #10.065239006121315
    #10.065456008797781
    @test Metida.m2logreml(lmm) ≈ 10.065238620486195 atol=1E-6


    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    random = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.CSH),
    repeated = Metida.VarEffect(Metida.@covstr(formulation|subject), Metida.CS),
    )
    Metida.fit!(lmm, solver = :nlopt, f_tol=1e-16, x_tol=1e-16)
    @test Metida.m2logreml(lmm) ≈ 10.3039977509049 atol=1E-6


    lmm = Metida.LMM(@formula(var~sequence+period+formulation), df0;
    repeated = Metida.VarEffect(Metida.@covstr(period|subject), Metida.CSH),
    )
    Metida.fit!(lmm; solver = :nlopt, f_tol=1e-16, x_tol=1e-16,
    )
    @test Metida.m2logreml(lmm) ≈ 8.740095378772942 atol=1E-6


    lmm = Metida.LMM(@formula(response ~1 + factor*time), ftdf;
    random = Metida.VarEffect(Metida.@covstr(1 + time|subject&factor), Metida.CSH),
    )
    Metida.fit!(lmm; solver = :nlopt, f_tol=1e-16, x_tol=1e-16)
    @test Metida.m2logreml(lmm) ≈ 1300.1807598168923 atol=1E-6

    lmm = Metida.LMM(@formula(response ~ 1 + factor*time), ftdf2;
    repeated = Metida.VarEffect(Metida.@covstr(time|subject&factor), Metida.ARMA),
    )
    Metida.fit!(lmm; solver = :nlopt, f_tol=1e-16, x_tol=1e-16)
    @test Metida.m2logreml(lmm) ≈ 715.4528559688382 atol = 1E-6

    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = [Metida.VarEffect(Metida.@covstr(r1|subject), Metida.CS), Metida.VarEffect(Metida.@covstr(r2|subject), Metida.CS)],
    )
    Metida.fit!(lmm; solver = :nlopt, f_tol=1e-16, x_tol=1e-16)
    @test Metida.m2logreml(lmm)  ≈ 710.4250214813896 atol=1E-6

    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = Metida.VarEffect(Metida.@covstr(r1|subject), Metida.AR),
    repeated = Metida.VarEffect(Metida.@covstr(p|subject), Metida.DIAG),
    )
    Metida.fit!(lmm; solver = :nlopt, f_tol=1e-10, x_tol=1e-10)
    @test Metida.m2logreml(lmm)  ≈ 698.8792511057682 atol=1E-6

    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = Metida.VarEffect(Metida.@covstr(p|r1&r2), Metida.ARMA),
    )
    Metida.fit!(lmm, solver = :nlopt)
    @test Metida.m2logreml(lmm)  ≈ 913.9176298311813 atol=1E-6

    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(1 + r1 + r2|subject), Metida.TOEPHP(3); coding = Dict(:r1 => DummyCoding(), :r2 => DummyCoding())),
    )
    Metida.fit!(lmm, solver = :nlopt, f_tol=1e-16, x_tol=1e-16)
    @test Metida.m2logreml(lmm)  ≈ 705.9946274598822 atol=1E-6


    lmm = Metida.LMM(@formula(response ~ 1 + factor), ftdf3;
    random = Metida.VarEffect(Metida.@covstr(r2|subject), Metida.DIAG),
    repeated = Metida.VarEffect(Metida.@covstr(p|subject), Metida.TOEPP(3)),
    )
    Metida.fit!(lmm, solver = :nlopt, f_tol=1e-16, x_tol=1e-16)
    @test Metida.m2logreml(lmm)  ≈ 773.9575538254085 atol=1E-6
end
