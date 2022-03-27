from generate_data import generate
import numpy as np


if __name__ == '__main__':
    job_name = 'EWD_xzzx3_alpha1'
    params = {'code':           'xzzx',
              'method':         'EWD',
              'size':           3,
              'noise':          'alpha',
              'eta':            0.5,
              'alpha':          1,
              'p_sampling':     0.3,
              'droplets':       1,
              'mwpm_init':      False,
              'fixed_errors':   None,
              'Nc':             None,
              'iters':          10,
              'conv_criteria':  'error_based',
              'SEQ':            2,
              'TOPS':           10,
              'eps':            0.01,
              'bias':           'Z',
              'onlyshortest':   True}
    
    # Steps is a function of code size
    params['steps'] = int(5*params['size']**5)
    
    n_ps = 20
    ps = np.linspace(0.01, 0.3, num=n_ps)
    for p_idx in range(n_ps):
        params['p_error'] = ps[p_idx]
    
        file_path = f'data/{job_name}_{p_idx}.xz'
        
        generate(file_path, params, nbr_datapoints=100)
