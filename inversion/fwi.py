import numpy as np
from scipy.optimize import minimize
from sampling.synthetic_experiment import prepare_synthetic_model
from sampling.fd_marine_experiment import prepare_model_marine
from sampling.fd_experiment import prepare_fd_model
from src.utilities import timer

def make_map_objective(bayes, cost_history):
    """Define Regularized FWI objective function for MAP estimation"""
    def map_objective(model):
        with timer("likelihood eval"):
            phi_data = bayes.log_likelihood(model)
        phi_prior = bayes.log_prior(model)
        phi_map = phi_data + phi_prior

        # Gradient via adjoint
        with timer("grad likelihood eval"):
            grad_phi_data = bayes.grad_log_likelihood(model) 
        grad_phi_prior = bayes.grad_log_prior(model)
        grad_phi_map = grad_phi_data + grad_phi_prior

        exit(1)
        l2_misfit = bayes.l2_misfit(model)
        print("l2 misfit = ", l2_misfit)
        # Store cost
        cost_history.append(phi_map)
        m_str = "[" + ", ".join(f"{m:7.1f}" for m in model) + "]"
        print(f"Iter {len(cost_history):>2} | MAP: {phi_map:.4e} (Data: {phi_data:.2e}, Prior: {phi_prior:.2e}) | m: {m_str}")

        return -phi_map, -grad_phi_map # because we maximize the MAP, so minimize -MAP

    return map_objective

def fwi_scipy():
    seed=42
    
    #m_ref = np.array([2000.0, 1700.0, 2300.0, 3000.0, 1630., 300.0, 700.])
    #bayes_model, param = prepare_model_marine(seed=seed)
    m_ref = np.array([1505.0, 2700.0, 3200.0, 1900.0, 4200.0, 3800.0, 2200.0, 4500.0])
    path = "FD_comparison/data/seis_v3_nofs"
    bayes_model, param = prepare_fd_model(file_path=path, seed=seed)
    

    #m_ref = np.array([1800.0, 3500.0, 150.]) # reference model
    #bayes_model, param = prepare_synthetic_model(seed=seed)
    print("beta =", bayes_model.beta)


    vp_bounds_list = [param.vp_bounds] * param.n_vp
    h_bounds_list = [param.h_bounds] * param.n_h
    all_bounds = vp_bounds_list + h_bounds_list

    phi_data = bayes_model.log_likelihood(m_ref) + bayes_model.log_prior(m_ref)
    grad_phi_data = bayes_model.grad_log_likelihood(m_ref)
    print(phi_data)
    print("l2 misfit", bayes_model.l2_misfit(m_ref))
    print("\n")
    print(grad_phi_data)

    m_ref = np.array([2002., 1967., 2210., 2916.,  636., 1085.,  953.])
    phi_data = bayes_model.log_likelihood(m_ref) + bayes_model.log_prior(m_ref)
    grad_phi_data = bayes_model.grad_log_likelihood(m_ref)
    print(phi_data)
    print("l2 misfit", bayes_model.l2_misfit(m_ref))
    print("\n")
    print(grad_phi_data)


    unique_map_minima = []
    num_starts = 30

    for i in range(num_starts):
        # Generate a random initial guess
        #random_m_init = np.array([2002., 1967., 2210., 2916.,  1586., 485.,  953.])#
        random_m_init = param.draw_from_prior()
        print("Initial model :", random_m_init)
        # random_m_init = np.array([1800.0, 3500.0, 140.])
        print(f"--- Start {i+1}/{num_starts} ---")
        cost_history = []
        obj_func = make_map_objective(bayes_model, cost_history)
        res = minimize(
            obj_func,
            random_m_init,
            method="L-BFGS-B",
            jac=True,
            bounds=all_bounds,
            options={"maxiter": 50, "ftol": 1e-6, "gtol": 1e-4}
        )
        print(f"{'-'*50}\nInverted Vp: {res.x}")
        print(f"Completed in {len(cost_history)} evaluations.")
        
        if res.success:
            unique_map_minima.append({'vp': res.x, 'map_val': -res.fun})
            print(f"minimum found!")
        else:
            unique_map_minima.append({'vp': res.x, 'map_val': -res.fun})
            print(f"did not converged")
            
    print("\n=== All Discovered MAP Local Minima ===")
    # Sort them by objective value so the best candidate is at the top
    unique_map_minima = sorted(unique_map_minima, key=lambda x: x['map_val'])
    for idx, minimum in enumerate(unique_map_minima):
        print(f"MAP Mode #{idx}: Objective = {minimum['map_val']:.4e} | Vp = {minimum['vp'].astype(int)}")

if __name__ == "__main__":
    fwi_scipy()
