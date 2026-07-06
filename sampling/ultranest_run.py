import os
import pickle
import numpy as np
import ultranest
import h5py
from synthetic_experiment import prepare_synthetic_model
from fd_experiment import prepare_fd_model
from fd_marine_experiment import prepare_model_marine
from parametrization import diagnostic_checks
#import ultranest.stepsampler
from ultranest.stepsampler import OrthogonalDirectionGenerator, generate_mixture_random_direction, SliceSampler, generate_cube_oriented_direction

def run_ultranest_inference(bayes, param, v_true, save_dir='dir'):

    def log_likelihood(m):
        return bayes.log_likelihood(m)

    diagnostic_checks(bayes, param.prior_transform, v_true)
    param_names = [f"Vp_{i+1}" for i in range(len(v_true))]

    sampler = ultranest.ReactiveNestedSampler(
        param_names,
        log_likelihood,
        param.prior_transform,
        log_dir = save_dir,
        resume = "resume",
    )

    #sampler.stepsampler = SliceSampler(
    #    nsteps             = 12 * len(v_true),
    #    generate_direction = OrthogonalDirectionGenerator(generate_mixture_random_direction)
    #)	
    	# ultranest.stepsampler.generate_cube_oriented_direction
    	# from ultranest.stepsampler import OrthogonalDirectionGenerator
	# generate_direction=OrthogonalDirectionGenerator(generate_region_random_direction)
	# OrthogonalDirectionGenerator(generate_mixture_random_direction)
    result = sampler.run(min_num_live_points=400, dlogz=0.5, min_ess=2000, update_interval_volume_fraction=0.4, max_num_improvement_loops=5)  # 1000
    sampler.print_results()

    samples = result["samples"]
    print(f"\nEqual-weighted posterior samples: {samples.shape}")
    print(f"Effective sample size (ESS): {result['ess']:.0f}")
    print(f"log Z = {result['logz']:.3f} ± {result['logzerr']:.3f}")

    filename = save_dir+'/results_ultranest'
    with open(filename + '.pkl', 'wb') as fp:
        pickle.dump(result, fp)
        print("results saved !")
    
    sampler.plot_run()
    sampler.plot_trace()
    sampler.plot_corner() 


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    # case = 'synthetic'
    case = 'fd_vp'
    # case = 'fd_marine'

    if case == 'synthetic':
        # 1) synthetic case - [vp1, vp2, height]
        m_ref = np.array([1800.0, 2500.0, 3200., 150., 50.]) # reference model used to generate the observations
        bayes_model, param = prepare_synthetic_model(seed=seed)
    elif case == 'fd_vp':
        # 2) FD case - [vp1, vp2, vp3, vp4, vp5, vp6, vp7]
        m_ref = np.array([2700.0, 3200.0, 1900.0, 4200.0, 3800.0, 2200.0, 4500.0])
        path = "FD_comparison/data/seis_v3_nofs"
        bayes_model, param = prepare_fd_model(file_path=path, seed=seed, debug=False)
        
    elif case == 'fd_marine':
        # 3) Marine mockup case - [vp1, vp2, vp3, vp4, h1, h2, h3] 
        m_ref = np.array([2000.0, 1700.0, 2300.0, 3000.0, 1630., 300.0, 700.])
        path = "FD_comparison/data/seis_marine_fs"
        bayes_model, param = prepare_model_marine(seed=seed, debug=False)
    else :
        raise ValueError('scenario does not exists !')
    
    print("n cpu : ", os.cpu_count())
    # Launch UltraNest
    save_dir = 'ultranest_resultsv3' + case
    run_ultranest_inference(bayes_model, param, v_true=m_ref, save_dir=save_dir)
