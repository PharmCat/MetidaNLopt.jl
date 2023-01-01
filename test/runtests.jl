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

ftdf3        = CSV.File(joinpath(dirname(pathof(Metida)),"..","test","csv","ftdf3.csv"); types =
[String,  Float64, Float64, String, String, String, String, String, Float64]) |> DataFrame

@testset "  Basic test                                               " begin
    lmm = LMM(@formula(var~sequence+period+formulation), df0;
    random = VarEffect(@covstr(formulation|subject), CSH),
    repeated = VarEffect(@covstr(formulation|subject), DIAG),
    )
    fit!(lmm; solver = :nlopt, f_tol=1e-12, x_tol=1e-12, time_limit = 12)
    #Metida.fit_nlopt!(lmm; solver = :nlopt, rholinkf = :sigm, f_tol=0.0, x_tol = 0.0, f_rtol =0.0, x_rtol =1e-18)
    #Metida.m2logreml(lmm) ≈ 10.065239006121315
    #10.065239006121315
    #10.065456008797781
    @test Metida.m2logreml(lmm) ≈ 10.065238692021847 atol=1E-4


    lmm = LMM(@formula(var~sequence+period+formulation), df0;
    random = VarEffect(@covstr(formulation|subject), CSH),
    repeated = VarEffect(@covstr(formulation|subject), CS),
    )
    fit!(lmm, solver = :nlopt, f_tol=1e-12, x_tol=1e-12)
    @test Metida.m2logreml(lmm) ≈ 10.3039977509049 atol=1E-4

    lmm = LMM(@formula(var~sequence+period+formulation), df0;
    repeated = VarEffect(@covstr(period|subject), CSH),
    )
    fit!(lmm; solver = :nlopt, f_tol=1e-12, x_tol=1e-12)
    @test Metida.m2logreml(lmm) ≈ 8.740095420232805 atol=1E-5


    lmm = LMM(@formula(response ~1 + factor*time), ftdf;
    random = VarEffect(Metida.@covstr(1 + time|subject&factor), CSH),
    )
    fit!(lmm; solver = :nlopt, f_tol=1e-12, x_tol=1e-12)
    @test m2logreml(lmm) ≈ 1300.1807598168923 atol=1E-5

    lmm = LMM(@formula(response ~ 1 + factor*time), ftdf2;
    repeated = VarEffect(@covstr(time|subject&factor), ARMA),
    )
    fit!(lmm; solver = :nlopt, f_tol=1e-12, x_tol=1e-12)#, optmethod = :LN_NEWUOA
    @test m2logreml(lmm) ≈ 715.4528559688382 atol = 1E-5

    lmm = LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = [VarEffect(@covstr(r1|subject), CS), VarEffect(@covstr(r2|subject), CS)],
    )
    fit!(lmm; solver = :nlopt, f_tol=1e-12, x_tol=1e-12)
    @test m2logreml(lmm)  ≈ 710.4250214813896 atol=1E-5

    lmm = LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = VarEffect(@covstr(r1|subject), AR),
    repeated = VarEffect(@covstr(p|subject), DIAG),
    )
    fit!(lmm; solver = :nlopt, f_tol=1e-12, x_tol=1e-12)
    @test m2logreml(lmm)  ≈ 698.8792511057682 atol=1E-5

    lmm = LMM(@formula(response ~ 1 + factor), ftdf3; contrasts=Dict(:factor => DummyCoding(; base=1.0)),
    random = VarEffect(@covstr(p|r1&r2), ARMA),
    )
    fit!(lmm, solver = :nlopt)
    @test m2logreml(lmm)  ≈ 913.9176298311813 atol=1E-5

    lmm = LMM(@formula(response ~ 1 + factor), ftdf3;
    random = VarEffect(@covstr(1 + r1 + r2|subject), TOEPHP(3); coding = Dict(:r1 => DummyCoding(), :r2 => DummyCoding())),
    )
    fit!(lmm, solver = :nlopt, f_tol=1e-12, x_tol=1e-12)
    @test m2logreml(lmm)  ≈ 705.9946274598822 atol=1E-5


    lmm = LMM(@formula(response ~ 1 + factor), ftdf3;
    random = VarEffect(@covstr(r2|subject), DIAG),
    repeated = VarEffect(@covstr(p|subject), TOEPP(3)),
    )
    fit!(lmm, solver = :nlopt, f_tol=1e-12, x_tol=1e-12)
    @test m2logreml(lmm)  ≈ 773.9575538254085 atol=1E-5


    lmm = LMM(@formula(response ~ 1), ftdf;
    repeated = VarEffect(Metida.@covstr(response+time|subject), SPEXP),
    )
    fit!(lmm, solver = :nlopt, f_tol=1e-12, x_tol=1e-12)
    #SPSS 1528.715
    @test m2logreml(lmm) ≈ 1528.7150702624508 atol=1E-5
end
