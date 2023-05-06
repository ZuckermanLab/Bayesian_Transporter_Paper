
import time
import os
import matplotlib.pyplot as plt
import tellurium as te
import numpy as np
import scipy as sp
import pocomc as pc
import emcee 
#import corner
import pandas as pd

import analysis
import sampler as ssme_sampler
import ssme 
import utility


def get_p0(b_list, n):
    '''get initial uniform distributed samples using boundaries from b_list, and number of samples n
    b_list[i][0] = parameter lower bound, b_list[i][1] = parameter upper bound
    '''
    p0_array = np.transpose(np.array([np.random.uniform(b[0],b[1],n) for b in b_list]))  # re-arrange array for sampler
    return p0_array



def mutate_mle(mle_x_list, n):

    mle_list = []
    for i in range(n):  
        idx_mle = np.random.choice(len(mle_x_list))  # pick random MLE estimate from list of estimates
        mle_x = mle_x_list[idx_mle]
        mle_x_tmp = mle_x[:]
        idx = np.random.choice(np.size(mle_x_tmp))
        old_p = mle_x_tmp[idx]
        mle_x_tmp[idx] = old_p*1.01
        mle_list.append(mle_x_tmp)
    return np.array(mle_list)



# def log_like_wrapper(params, rr_model, y_obs, sim_args):
    
#     k = params[:-1]
#     sigma = 10**params[-1]

#     try:
#         res = ssme.simulate_y_pred_rr(rr_model, k, **sim_args)
#         y_pred = res[1]
#         log_like = ssme_sampler.calc_normal_log_likelihood(y_obs, y_pred, sigma)
#         return log_like
#     except:
#         return -1e100


def log_like_wrapper(params, rr_model, y_obs, sim_args):

    k = params[:-1]
    sigma = 10**params[-1]
    
    # k = params[:-4]
    # buffer_concencration_error = params[-4:-2]
    # protein_concentration_error = np.array(params[-2:-1]).item()
    # sigma = 10**params[-1]

    # sim_args['buffer_uncertainty'] = buffer_concencration_error
    # sim_args['state_init_uncertainty'] = [protein_concentration_error, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    try:
        res = ssme.simulate_y_pred_rr(rr_model, k, **sim_args)
        y_pred = res[1]
        log_like = ssme_sampler.calc_normal_log_likelihood(y_obs, y_pred, sigma)
        return log_like
    except:
        return -1e100


def log_prior(params, param_lb, param_ub):  
    if ((param_lb < params) & (params < param_ub)).all():
        return 0
    else:
        return -1e100
   

def log_post_wrapper(params, rr_model, y_obs, sim_args, param_lb, param_ub):
    logl = log_like_wrapper(params, rr_model, y_obs, sim_args)
    logpr = log_prior(params, param_lb, param_ub)
    return logl+logpr


if __name__ == '__main__':
    
    config_fname = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/data/config/config_15D_c1off_c2on.json"
    run_name = "c1off_c2on_p0_true_long"

    config = utility.parse_config_file(config_fname)

    model_file = config['model_file']
    data_file = config['data_file']
    simulation_kwargs = config['simulation_kwargs']
    model_parameters = config['model_parameters']
    run_kwargs = config['run_kwargs']
    utility.save_config_file(config, f"{run_kwargs['output_label']}_log.txt")
    
    seed = run_kwargs['seed'] + 4
    np.random.seed(seed)
    param_ref = [value[2] for key, value in model_parameters.items()]
    param_bounds = [[value[0], value[1]] for key, value in model_parameters.items()]
    param_lb = [value[0] for key, value in model_parameters.items()]
    param_ub = [value[1] for key, value in model_parameters.items()]
    param_names = list(model_parameters.keys())
    k_ref = param_ref[:-1]
    sigma_ref = param_ref[-1]
 
    rr_model = utility.load_rr_model_from_sbml(model_file)
    res = ssme.simulate_y_pred_rr(rr_model,k_ref,**simulation_kwargs)
    y_ref = res[1]
    y_obs = np.loadtxt(data_file, delimiter=',')
    logl_ref = ssme_sampler.calc_normal_log_likelihood(y_obs,y_ref,10**sigma_ref)
    print(logl_ref)

    n_dim = int(len(param_bounds))
    print(n_dim)
    n_walkers = 1000
    #mle = [9.675829244193901, 4.899925883005722, 3.6848895674828293, 2.291313263472443, 7.026579307003583, 2.7262849582418998, 1.4796067698565472, 8.415071906534614, 1.9999606876441745, 2.230257712716666, 1.8264563681164319, 9.904052362045267, 2.253693579564178, 8.001547276877359, -10.006341630194365]
    mle = [
        [10.106054266899314, -0.514633031449957, 2.5044110520931175, 3.3545473933907584, 8.711054622854244, 3.3563965244791554, 2.958412885421001, 9.893622609125321, 2.0020276980824794, 2.032389866122289, 2.7238996206627513, 10.008156842404317, 1.2475554445501718, 5.805789385715351, -9.956283296375597],
        [10.938004289974636, 2.4736572526579463, 1.287404039221356, 1.9674006639837465, 5.264109501840987, 1.353323540116534, 3.5718164835359207, 10.557616010245905, 2.0106195274894323, 2.0388532879667123, 1.7105554930570475, 10.091686012556536, 3.169386690051113, 6.077408293340204, -9.963852437869358],
        [11.45093760088415, 1.9851214447380874, 2.4943822170498264, 3.32540293213005, 4.307622450921814, 0.4595523351569348, 2.2622414499664107, 9.19200601647027, 2.001183023671291, 2.0340381056513306, 4.0111943600460815, 10.304255422152435, 4.7815236707027875, 6.5522095852016164, -9.985247888552983],
        [11.716636158916629, 1.289369560916688, 1.4696846794192686, 3.380965233263826, 5.688923354489569, 0.6673893377994644, 1.310485200005657, 8.222481264986802, 1.9963751265832077, 2.0499002011739393, 3.141454820119232, 11.424659580985448, 3.2236342136573186, 5.96625201339349, -9.950948492223946],
        [11.395023161526485, 2.8963017526371653, 1.1536821494454093, 3.6594669820131047, 6.176803349446438, 0.7675285062247799, 3.220587707614868, 10.107712863584185, 1.9949597789972227, 2.070009811747542, 4.470933967207769, 9.446366258315845, 3.541354783116863, 4.35285371903819, -9.97235807376055],
        [10.673505245941307, 0.6545269125971083, 0.9681568155705396, 2.039644558587959, 6.9195788958145235, 2.8815670226763985, 3.839535191799305, 10.787298752423045, 1.999398919356603, 2.0503699542130054, 2.624576190101497, 11.071941283268385, 4.300777359137106, 5.9440370483800935, -9.922719716215036],
        [10.568692086607536, 2.9877074909802546, 1.0333770736395298, 2.5916341701487418, 5.6732052640244275, 1.1478105653065789, 3.373402357229988, 10.34089225767867, 2.0006984275926865, 2.1764272900211967, 3.6893341051603876, 8.462175867702337, 3.152867655990202, 3.509163326042084, -10.011042411532161],
        [9.812380196225028, 1.8513838470673096, 3.007399497472996, 1.6946835948818257, 8.226762457656532, 4.561017345015551, 2.562364323616091, 9.539599113839408, 2.0003341741726617, 2.029513445600144, -0.8100633126641078, 10.062281102932095, 3.49450151993392, 8.507522209258438, -9.919824910384653],
        [8.779550433886095, 0.08359150513129565, 2.943216635356658, 2.999823440371715, 7.002874142450488, 2.395326838663261, 1.3192481227838269, 8.233104877218741, 1.9955797466295582, 2.0490064433432256, 1.1845885741290838, 10.548993632758876, 0.24411966115771788, 4.878092096972677, -10.021557594457484],
        [11.490384670552743, 0.7644452026431314, 1.240912382456409, 3.3719156279370543, 7.05158687997097, 1.6718418740850136, 2.0057122126663485, 8.959190653378265, 1.9985955591579379, 2.025720274060759, 1.0649793682490931, 11.238794844630714, 2.6595121063337057, 5.676986208189613, -9.949211310865],
        [10.410027160360041, 2.3845881829010924, 1.606097104919796, 2.1674876766496536, 7.716795427033712, 3.544941722306271, 3.610011545950641, 10.57539162834038, 2.0014952161832076, 2.0475897442449758, 3.340621741915352, 10.120624112123204, 1.147211975575217, 4.687741792631671, -10.00428874032077],
        [9.922149629236191, 4.0951677248293095, 2.4087263594280937, 3.999861675019953, 7.037906460569812, 1.1181303491042014, 1.5207569238355552, 8.40813316444536, 1.998505379671758, 2.664904434954359, 2.4619399880935524, 9.819425215831414, 2.0955312692173074, 6.478206970235532, -10.002219049416974],
        [10.856119289321398, 4.427027989010118, 1.9018783881415764, 3.164541418609939, 6.460783211009927, 1.329633321475643, 3.556105492985338, 10.472685684509997, 2.001179097105756, 2.358673387666849, 2.8852843748345443, 9.47515403478394, 4.315354158958951, 8.190524048629948, -9.998642522604918],
        [9.428604387313685, 2.64677339249194, 1.7517571101943519, 2.575602912930529, 5.52662950537128, 1.055470866848847, 2.3219995365340833, 9.263745191900908, 2.0022959293846725, 2.232891308179016, 2.956306538203518, 10.106010231691158, 1.3654709504899614, 4.9780015761225105, -10.006405563571986],
        [8.910483085776086, 3.1487579741614877, 2.314819901874577, 2.6824311346372314, 6.951796193496907, 2.2545990536386173, 3.418943157668202, 10.324956052840683, 1.994233416063844, 2.642551112073747, 4.120803607796136, 10.522547434925672, 2.1551070078541845, 6.552829154637872, -9.992620109276968],
        [10.411608809616258, 2.9576614395096907, 1.2497857474946739, 2.185625696591986, 5.989882702490041, 1.811561543452579, 2.371364818476637, 9.337356505301305, 2.0042381025315663, 2.120497025207187, 2.4047483035519863, 11.979232903218538, 2.2343038658767314, 5.213263121377545, -9.908610372787699],
        [11.440840485742417, 3.8128600506990984, 1.449411232370439, 3.854477647521199, 6.561701176248841, 0.8884584167869651, 1.3450898154993567, 9.57070249417582, 3.960536874713288, 2.452119736273214, 0.932435759310568, 6.918591360573518, 1.5610472420568418, 4.623889717673142, -9.97969283864629],
        [11.051069439826861, 4.051297146804855, 2.461288938472734, 2.392736944144288, 4.2319402416950185, 0.5262335748146676, 1.3659475531517713, 8.332525558914373, 2.0206367492429833, 2.1449751396297887, 3.2837080544599195, 10.12169724615533, 2.335772903795191, 6.017015603815946, -9.978981441827191],
        [10.466477964665597, 0.5011025859558942, 2.0512182787402016, 1.5373277955401075, 5.284269853747109, 1.8004006217060915, 1.4171634059950122, 8.480666619087973, 2.010465640890927, 1.9966464819822967, 2.5950269626579576, 11.141024986305078, 2.8343023388888695, 6.771730885220591, -9.982100482109411],
        [6.012872241743758, -0.020834803201942073, 2.221962154397803, 1.3495624022114447, 6.0954454165742975, 2.913548330049141, 2.762143091564303, 10.089262242366802, 2.0074346372764453, 1.9755931713751553, 0.5381137793341496, 10.348777921388312, 2.2989295001436174, 6.042175370053803, -9.956755379199185],         
    ]
    

    
    
    p_0 = mutate_mle([param_ref], n_walkers)
    #p_0 = get_p0(param_bounds,n_walkers)
    print(np.shape(p_0))
    save_every = 10
    method='emcee'
    n_steps = 100000
   

    if method=='pocoMC':
        print('running pocoMC sampler')
        additional_samples = int(1e3)
        sampler = pc.Sampler(
                n_walkers,
                n_dim,
                log_likelihood=log_like_wrapper,
                log_prior=log_prior,
                vectorize_likelihood=False,
                vectorize_prior=False,
                bounds=np.array(param_bounds),
                random_state=seed,
                log_likelihood_args = [rr_model, y_obs, simulation_kwargs],
                log_prior_args = [param_lb, param_ub],
                infer_vectorization=False,
                output_dir='/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/scripts'
            )

        
        t0=time.time()
        sampler.run(p_0, ess= 0.95, gamma = 0.75)

        # We can add more samples at the end
        sampler.add_samples(additional_samples)

        # Get results
        results = sampler.results  

        ### write wall clock time to file
        wallclock = time.time() -t0
        print(f'runtime: {wallclock} sec')

        # Trace plot
        pc.plotting.trace(results)   
        plt.savefig('traceplot.png')
        plt.clf()

        # # Corner plot
        # pc.plotting.corner(results)
        # plt.savefig('cornerplot.png')
        # plt.clf()

        # Run plot
        pc.plotting.run(results)
        plt.savefig('runplot.png')
        plt.clf()

        np.savetxt('samples.csv',results['samples'],delimiter=',')
        np.savetxt('loglikelihood.csv',results['loglikelihood'],delimiter=',')
    

    elif method == 'emcee':
        print('running emcee sampler')
        # filename = "15D_c1on_c2off_run_test_gdx_2_1.h5"
        # backend = emcee.backends.HDFBackend(filename)
        # backend.reset(n_walkers, n_dim)
        #sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_post_wrapper, args=[rr_model, y_obs, simulation_kwargs, param_lb, param_ub], backend=backend)
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_post_wrapper, args=[rr_model, y_obs, simulation_kwargs, param_lb, param_ub])
        t0=time.time()
        sampler.run_mcmc(p_0, n_steps,skip_initial_state_check=True, progress=True)
        print(f'runtime: {time.time()-t0} sec')

        thin = 1000
        samples = sampler.get_chain(flat=True, thin=thin)
        log_prob_samples = sampler.get_log_prob(flat=True, thin=thin)
        log_prior_samples = sampler.get_blobs(flat=True, thin=thin)

        np.savetxt(f'samples_{run_name}_r{seed}.csv', samples, delimiter=',')
        np.savetxt(f'log_prob_{run_name}_r{seed}.csv', log_prob_samples, delimiter=',')
        

        print("thin: {0}".format(thin))
        print("flat chain shape: {0}".format(samples.shape))
        print("flat log prob shape: {0}".format(log_prob_samples.shape))


        labels = list(map(r"$\theta_{{{0}}}$".format, range(1, n_dim + 1)))
        labels += ["log prob"]

        #corner.corner(samples, range=param_bounds, labels=param_names, truths=param_ref);
        #plt.savefig(f'corner_{run_name}_r{seed}.png')