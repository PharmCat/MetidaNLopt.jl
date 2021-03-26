# MetidaNLopt
# Copyright © 2019-2020 Vladimir Arnautov aka PharmCat <mail@pharmcat.net>
using Metida
using MetidaNLopt
using  Test, CSV, DataFrames, StatsModels

df0 = CSV.File(joinpath(dirname(pathof(Metida)),"..","test","csv","df0.csv"); types = [String, String, String, String, Float64, Float64]) |> DataFrame
transform!(df0, :subject => categorical, renamecols=false)
transform!(df0, :period => categorical, renamecols=false)
transform!(df0, :sequence => categorical, renamecols=false)
transform!(df0, :formulation=> categorical, renamecols=false)
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
end
