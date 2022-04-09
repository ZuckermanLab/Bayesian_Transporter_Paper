using Catalyst, DifferentialEquations, StatsPlots, Plots, DataFrames, CSV, Random, Turing, Latexify, ModelingToolkit, DynamicHMC,Noise
using ModelingToolkit: varmap_to_vars
using Optim
Random.seed!(14)


# Normal log-likelihood calculation
function calc_norm_log_likelihood(y_obs,y_pred,sigma)
    n = length(y_obs)
    t1 = -(n/2) * (log(2*pi))
    t2 = -(n/2) * (sigma^2)
    t3_sum = 0
    for i in 1:n
        t3_sum = t3_sum + (y_obs[i]-y_pred[i])^2
    end
    t3 = -(1/(2*sigma^2)) * t3_sum
    log_likelihood = t1+t2+t3
end


# load flux dataset of ~ 300 data points
df = CSV.read("data_1c.csv", DataFrame) 
y_obs = df[!, "data"]

# flux calculation function
function calc_flux(p,u,v)
    flux_in = v*(p[1]*u[1])
    flux_out = v*(p[2]*u[2].*u[3])
    net_flux = flux_in - flux_out         
end

# define reaction network in Catalyst
rn = @reaction_network begin
  (k1_f*H_out, k1_r), OF <--> OF_Hb
  (k2_f, k2_r), OF_Hb <--> IF_Hb
  (k3_f, k3_r), IF_Hb + S_in <--> IF_Hb_Sb
  (k4_f, k4_r), IF_Hb_Sb <--> IF_Sb + H_in
  (k5_f, k5_r), IF_Sb <--> OF_Sb
  (k6_f, k6_r*S_out), OF_Sb <--> OF
end H_out S_out k1_f k1_r k2_f k2_r k3_f k3_r k4_f k4_r k5_f k5_r k6_f k6_r

# convert reaction network to ODEsystem
odesys = convert(ODESystem, rn)
latexify(odesys)

# define initial variable values and parameters
@parameters  H_out S_out k1_f k1_r k2_f k2_r k3_f k3_r k4_f k4_r k5_f k5_r k6_f k6_r stdev_s
@variables t H_in(t) S_in(t) OF(t) OF_Hb(t) IF_Hb(t) IF_Hb_Sb(t) IF_Sb(t) OF_Sb(t)
u0map = [H_in => 1e-7, S_in => 1e-3, 
        OF => 2.833e-8, OF_Hb => 2.833e-8,  
        IF_Hb => 2.833e-8, IF_Hb_Sb => 2.833e-8,
        IF_Sb => 2.125e-8, OF_Sb => 2.125e-8, 
        ]
H_out_ref = 1e-7
S_out_ref = 1e-3
pmap = [H_out => H_out_ref, S_out => S_out_ref,
        k1_f =>1e10, k1_r =>1e3,
        k2_f =>1e2, k2_r =>1e2,
        k3_f =>1e7, k3_r =>1e3,
        k4_f =>1e3, k4_r =>1e10,
        k5_f =>1e2, k5_r =>1e2,
        k6_f =>1e3, k6_r =>1e7
        ]
stdev_ref = 1e-13

# create ODEproblem w/ timed events + discrete callback
tspan1 = (0.0,15.0)
oprob = ODEProblem(odesys, u0map, tspan1, pmap)
event_times = [5.0,10]
condition(u,t,integrator) = t ∈ event_times
function affect!(integrator)
    if integrator.t == 5.0
        integrator.p[1] = 5e-8
    elseif integrator.t == 10.0
        integrator.p[1] = 1e-7
    end
end
cb = DiscreteCallback(condition,affect!)

s_test = solve(oprob, TRBDF2(), abstol = 1e-18, reltol = 1e-12, saveat=0.04, callback=cb, tstops=[5.0, 10.0])
k = [pmap[9][2], pmap[10][2]]
c = [s_test[IF_Hb_Sb], s_test[H_in], s_test[IF_Sb]]
vol = 0.0001
y_pred_test = calc_flux(k,c,vol)


# add noise to create synthetic flux trace
stdev_test = 10*1e-13
y_obs_test = add_gauss(y_pred_test,stdev_test)
plot(y_pred_test, legend = false, ylim=[-1.5e-11,1.5e-11]);scatter!(y_obs_test, alpha = 0.5)
savefig("julia_test.png")
# df = DataFrame(data = y_obs)
# CSV.write("data.csv", df)
# df2 = CSV.read("data.csv", DataFrame)
# y_obs2 = df2[!, "data"]
# scatter(y_obs, alpha = 0.5, ylim=[-1.5e-11,1.5e-11])


println(calc_norm_log_likelihood(y_obs_test,y_pred_test,stdev_test))

# create Turing model w/ forward differentiation
Turing.setadbackend(:forwarddiff)
@model function rxn_sampler(data, prob, sys)
    
    # set priors
    #log_stdev ~ Uniform(5, 12.5)
    #stdev_sample = 1e-13
    #stdev_sample ~ Uniform(0.25e-13 , 2.5e-13) 
    #stdev_sample ~ InverseGamma(1e-13, 1e-13)
    log_k1_f ~ Uniform(6, 12)  # log10 rate constant (ref=1e10)
    log_k1_r ~ Uniform(-1,5)  # log10 rate constant (ref=1e3)  
    log_k2_f ~ Uniform(-2,4)  # log10 rate constant (ref=1e2)
    log_k2_r ~ Uniform(-2,4)  # log10 rate constant (ref=1e2)
    log_k3_f ~ Uniform(3,9)  # log10 rate constant (ref=1e7) 
    log_k3_r ~ Uniform(-1,5)  # log10 rate constant (ref=1e3)  
    log_k4_f ~ Uniform(-1,5)  # log10 rate constant (ref=1e3) 
    log_k4_r ~ Uniform(6, 12)  # log10 rate constant (ref=1e10)
    log_k5_f ~ Uniform(-2,4)  # log10 rate constant (ref=1e2)
    log_k5_r ~ Uniform(-2,4)   # log10 rate constant (ref=1e2)
    log_k6_f ~ Uniform(-1,5)  # log10  rate constant (ref=1e3)
    
    # set parameter values and initial concentrations
    log_k6_r = (log_k1_f+log_k2_f+log_k3_f+log_k4_f+log_k5_f+log_k6_f)-(log_k1_r+log_k2_r+log_k3_r+log_k4_r+log_k5_r) 
    pmap1 = [H_out => H_out_ref, S_out => S_out_ref,
            k1_f =>10.0^log_k1_f, k1_r =>10.0^log_k1_r,
            k2_f =>10.0^log_k2_f, k2_r =>10.0^log_k2_r,
            k3_f =>10.0^log_k3_f, k3_r =>10.0^log_k3_r,
            k4_f =>10.0^log_k4_f, k4_r =>10.0^log_k4_r,
            k5_f =>10.0^log_k5_f, k5_r =>10.0^log_k5_f,
            k6_f =>10.0^log_k6_f, k6_r =>10.0^log_k6_r,
    ]
    
    # solve ODE system
    pnew = varmap_to_vars(pmap1, parameters(sys))
    oprob1 = remake(prob, p=pnew)
    sol = solve(oprob1, TRBDF2(), abstol = 1e-18, reltol = 1e-12, saveat=0.04, callback=cb, tstops=[5.0, 10.0])
    
    # calc flux (y_pred)
    k = [pmap1[9][2], pmap1[10][2]]
    c = [sol[IF_Hb_Sb], sol[H_in], sol[IF_Sb]]
    vol = 0.0001
    y_pred = calc_flux(k,c,vol)
    #stdev_sample = log_stdev*1e-14
    stdev_sample = 1e-13
    for i = 1:length(data)
        data[i] ~ Normal(y_pred[i], stdev_sample)
    end
end
model = rxn_sampler(y_obs_test, oprob, odesys)

# set initial sample point (parameter set)
    
iterations = 10_000

p2 = [
      10.0, 3.0,
      2.0, 2.0,
      7.0, 3.0,
      3.0, 10.0,
      2.0, 2.0,
      3.0, 
        ]

#chain = sample(model, NUTS(0.65, init_ϵ=1e-5), iterations)
#chain = sample(model, HMC(1e-6, 100), iterations)
#chain = sample(model, MH(), iterations)
chain = sample(model, MH(), iterations, init_params=p2)

# analysis
println(chain[:log_k1_f][1])
println(chain[:log_k1_r][1])
println(chain[:log_k2_f][1])
println(chain[:log_k2_r][1])
println(chain[:log_k3_f][1])
println(chain[:log_k3_r][1])
println(chain[:log_k4_f][1])
println(chain[:log_k4_r][1])
println(chain[:log_k5_f][1])
println(chain[:log_k5_r][1])
println(chain[:log_k6_f][1])
println(chain[:lp][1])

