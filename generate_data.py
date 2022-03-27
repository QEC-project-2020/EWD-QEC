import copy
import os
import numpy as np
import pandas as pd
from scipy import optimize

# code models
from src.toric_model import Toric_code
from src.planar_model import Planar_code
from src.xzzx_model import xzzx_code
from src.rotated_surface_model import RotSurCode
from src.xyz2_model import xyz_code

# decoders
from decoders import MCMC, single_temp, single_temp_alpha, EWD, \
                        EWD_general_noise, EWD_general_noise_shortest, \
                        EWD_alpha_N_n, EWD_alpha, MCMC_biased, \
                        MCMC_alpha_with_shortest, MCMC_alpha 
from src.mwpm import class_sorted_mwpm, regular_mwpm, enhanced_mwpm


def get_individual_error_rates(params):
    assert params['noise'] in ['depolarizing', 'alpha', 'biased'], f'{params["noise"]} is not implemented.'
    
    if params['noise'] == 'biased':
        eta = params['eta']
        p = params['p_error']
        if params['bias'] == 'Z':
          p_z = p * eta / (eta + 1)
          p_x = p / (2 * (eta + 1))
          p_y = p_x
        elif params['bias'] == 'X':
          p_x = p * eta / (eta + 1)
          p_z = p / (2 * (eta + 1))
          p_y = p_z
        elif params['bias'] == 'Y':
          p_y = p * eta / (eta + 1)
          p_z = p / (2 * (eta + 1))
          p_x = p_z
    
    if params['noise'] == 'alpha':
        # Calculate p_bias_tilde from p_error (total error prob.)
        p = params['p_error']
        alpha = params['alpha']
        p_tilde = p / (1 - p)
        p_bias_tilde = optimize.fsolve(lambda x: x + 2*x**alpha - p_tilde, 0.5)[0]
        
        if params['bias'] == 'Z':
          p_z = p_bias_tilde*(1 - p)
          p_x = p_y = p_bias_tilde**alpha * (1 - p)
        elif params['bias'] == 'X':
          p_x = p_bias_tilde*(1 - p)
          p_z = p_y = p_bias_tilde**alpha * (1 - p)
        elif params['bias'] == 'Y':
          p_y = p_bias_tilde*(1 - p)
          p_x = p_z = p_bias_tilde**alpha * (1 - p)
        
    if params['noise'] == 'depolarizing':
        p_x = p_y = p_z = params['p_error']/3

    return p_x, p_y, p_z


# This function generates training data with help of the MCMC algorithm
def generate(file_path, params, nbr_datapoints=10**6, fixed_errors=None):

    # Creates df
    df = pd.DataFrame()

    # Add parameters as first entry in dataframe
    names = ['data_nr', 'type']
    index_params = pd.MultiIndex.from_product([[-1], np.arange(1)],
                                                names=names)
    df_params = pd.DataFrame([[params]],
                            index=index_params,
                            columns=['data'])
    df = df.append(df_params)

    print('\nDataFrame opened at: ' + str(file_path))

    # If using a fixed number of errors, let the max number of datapoins be huge
    if fixed_errors != None:
        nbr_datapoints = 10000000
    failed_syndroms = 0

    # Initiate temporary list with results (to prevent appending to dataframe each loop)
    df_list = []

    p_x, p_y, p_z = get_individual_error_rates(params)

    # Loop to generate data points
    for i in range(nbr_datapoints):
        print('Starting generation of point nr: ' + str(i + 1), flush=True)

        # Initiate code
        if params['code'] == 'toric':
            assert params['noise'] == 'depolarizing', f'{params["noise"]}-noise is not compatible with "{params["code"]}"-model.'
            init_code = Toric_code(params['size'])
            init_code.generate_random_error(params['p_error'])
        elif params['code'] == 'planar':
            assert params['noise'] in ['depolarizing', 'alpha'], f'{params["noise"]}-noise is not compatible with "{params["code"]}"-model.'
            init_code = Planar_code(params['size'])
            init_code.generate_random_error(p_x=p_x, p_y=p_y, p_z=p_z)
        elif params['code'] == 'xzzx':
            assert params['noise'] in ['depolarizing', 'alpha', 'biased'], f'{params["noise"]}-noise is not compatible with "{params["code"]}"-model.'
            init_code = xzzx_code(params['size'])
            init_code.generate_random_error(p_x=p_x, p_y=p_y, p_z=p_z)
        elif params['code'] == 'rotated':
            assert params['noise'] in ['depolarizing', 'alpha', 'biased'], f'{params["noise"]}-noise is not compatible with "{params["code"]}"-model.'
            init_code = RotSurCode(params['size'])
            init_code.generate_random_error(p_x=p_x, p_y=p_y, p_z=p_z)
        elif params['code'] == 'xyz2':
            assert params['noise'] in ['depolarizing', 'alpha', 'biased'], f'{params["noise"]}-noise is not compatible with "{params["code"]}"-model.'
            init_code = xyz_code(params['size'])
            init_code.generate_random_error(p_x=p_x, p_y=p_y, p_z=p_z)
 
        # Flatten initial qubit matrix to store in dataframe
        df_qubit = copy.deepcopy(init_code.qubit_matrix)
        eq_true = init_code.define_equivalence_class()

        # Create inital error chains for algorithms to start with
        if params['mwpm_init']: #get mwpm starting points
            assert params['code'] == 'planar', 'Can only use eMWPM for planar model.'
            init_code = class_sorted_mwpm(init_code)
            print('Starting in MWPM state')
        else: #randomize input matrix, no trace of seed.
            init_code.qubit_matrix, _ = init_code.apply_random_logical()
            init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
            print('Starting in random state')

        # Generate data for DataFrame storage  OBS now using full bincount, change this
        if params['method'] == "MCMC":
            if params['noise'] == 'depolarizing':
                df_eq_distr = MCMC(init_code,
                                   params['p_error'],
                                   Nc=params['Nc'],
                                   SEQ=params['SEQ'],
                                   TOPS=params['TOPS'],
                                   eps=params['eps'],
                                   iters=params['iters'],
                                   conv_criteria=params['conv_criteria'])
                if np.argmax(df_eq_distr) != eq_true:
                    print('Failed syndrom, total now:', failed_syndroms)
                    failed_syndroms += 1
            if params['noise'] == "biased":
                df_eq_distr = MCMC_biased(init_code,
                                          params['p_error'],
                                          eta=params['eta'],
                                          Nc=params['Nc'],
                                          SEQ=params['SEQ'],
                                          TOPS=params['TOPS'],
                                          eps=params['eps'],
                                          iters=params['iters'],
                                          conv_criteria=params['conv_criteria'])
                if np.argmax(df_eq_distr) != eq_true:
                    print('Failed syndrom, total now:', failed_syndroms)
                    failed_syndroms += 1
            if params['noise'] == "alpha":
                df_eq_distr = MCMC_alpha(init_code,
                                        params['p_error'],
                                        alpha=params['alpha'],
                                        Nc=params['Nc'],
                                        SEQ=params['SEQ'],
                                        TOPS=params['TOPS'],
                                        eps=params['eps'],
                                        iters=params['iters'],
                                        conv_criteria=params['conv_criteria'])
                if np.argmax(df_eq_distr[0]) != eq_true:
                    print('Failed syndrom, total now:', failed_syndroms)
                    failed_syndroms += 1
        if params['method'] == "MCMC_with_shortest":
            assert params['noise'] == 'alpha'
            if params['noise'] == "alpha":
                df_eq_distr = MCMC_alpha_with_shortest(init_code, params['p_error'], alpha=params['alpha'])
                if np.argmax(df_eq_distr[0:4]) != eq_true:
                    print('Failed syndrom, total now:', failed_syndroms)
                    failed_syndroms += 1
        elif params['method'] == "EWD":
            if params['noise'] == 'depolarizing':
                assert params['onlyshortest'] == False, "onlyshortest not implemented for deoplarizing"
                df_eq_distr = EWD(init_code, params['p_error'], params['p_sampling'], steps=params['steps'], droplets=params['droplets'])
                df_eq_distr = np.array(df_eq_distr)
                if np.argmax(df_eq_distr) != eq_true:
                    print('Failed syndrom, total now:', failed_syndroms)
                    failed_syndroms += 1
            elif params['noise'] == 'alpha':
                alpha=params['alpha']
                p_tilde_sampling = params['p_sampling'] / (1 - params['p_sampling'])
                p_bias_tilde_sampling = optimize.fsolve(lambda x: x + 2*x**alpha - p_tilde_sampling, 0.5)[0]
                p_tilde = params['p_error'] / (1 - params['p_error'])
                p_bias_tilde = optimize.fsolve(lambda x: x + 2*x**alpha - p_tilde, 0.5)[0]
                df_eq_distr = EWD_alpha(init_code,
                                         p_bias_tilde,
                                         alpha,
                                         params['steps'],
                                         p_bias_tilde_sampling=p_bias_tilde_sampling,
                                         onlyshortest=params['onlyshortest'],
                                         bias=bias)
                df_eq_distr = np.array(df_eq_distr)
            elif params['noise'] == 'biased':
                p = params['p_error']
                bias = params['bias']
                if bias == 'Z':
                  p_bias_tilde = p_z / (1 - p)
                  alpha = log(p_x/(1-p)) / log(p_z/(1-p))
                elif bias == 'X':
                  p_bias_tilde = p_x / (1 - p)
                  alpha = log(p_z/(1-p)) / log(p_x/(1-p))
                elif bias == 'Y':
                  p_bias_tilde = p_y / (1 - p)
                  alpha = log(p_x/(1-p)) / log(p_y/(1-p))
                p_bias_tilde_sampling = params['p_sampling'] / (1 - params['p_sampling'])
                df_eq_distr = EWD_alpha(init_code,
                                         p_bias_tilde,
                                         alpha,
                                         params['steps'],
                                         p_bias_tilde_sampling=p_bias_tilde_sampling,
                                         onlyshortest=params['onlyshortest'],
                                         bias=bias)
                df_eq_distr = np.array(df_eq_distr)
            else:
                raise ValueError(f'''EWD does not support "{params['noise']}" noise''')
        elif params['method'] == "ST":
            if params['noise'] == 'depolarizing':
                df_eq_distr = single_temp(init_code, params['p_error'], params['steps'])
                df_eq_distr = np.array(df_eq_distr)
                if np.argmin(df_eq_distr) != eq_true:
                    print('Failed syndrom, total now:', failed_syndroms)
                    failed_syndroms += 1
            elif params['noise'] == 'alpha':
                p_tilde = params['p_error'] / (1 - params['p_error'])
                pz_tilde = optimize.fsolve(lambda x: x + 2*x**params['alpha'] - p_tilde, 0.5)[0]
                df_eq_distr = single_temp_alpha(init_code,
                                                pz_tilde,
                                                params['alpha'],
                                                params['steps'])
                df_eq_distr = np.array(df_eq_distr)
                if np.argmin(df_eq_distr) != eq_true:
                    print('Failed syndrom, total now:', failed_syndroms)
                    failed_syndroms += 1
            else:
                raise ValueError(f'''ST does not support "{params['noise']}" noise''')
        elif params['method'] == "eMWPM":
            out = class_sorted_mwpm(copy.deepcopy(init_code))
            lens = np.zeros((4))
            for j in range(4):
                lens[j] = sum(out[j].chain_lengths())
            choice = np.argmin(lens)
            df_eq_distr = np.zeros((4)).astype(np.uint8)
            df_eq_distr[choice] = 100
            if np.argmax(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1
        elif params['method'] == "MWPM":
            choice = regular_mwpm(copy.deepcopy(init_code))
            df_eq_distr = np.zeros((4)).astype(np.uint8)
            df_eq_distr[choice] = 100
            if np.argmax(df_eq_distr) != eq_true:
                print('Failed syndrom, total now:', failed_syndroms)
                failed_syndroms += 1

        # Generate data for DataFrame storage  OBS now using full bincount, change this

        # Create indices for generated data
        names = ['data_nr', 'type']
        index_qubit = pd.MultiIndex.from_product([[i], np.arange(1)],
                                                 names=names)
        index_distr = pd.MultiIndex.from_product([[i], np.arange(1)+1], names=names)

        # Add data to Dataframes
        df_qubit = pd.DataFrame([[df_qubit.astype(np.uint8)]], index=index_qubit,
                                columns=['data'])
        df_distr = pd.DataFrame([[df_eq_distr]],
                                index=index_distr, columns=['data'])

        # Add dataframes to temporary list to shorten computation time
        
        df_list.append(df_qubit)
        df_list.append(df_distr)

        # Every x iteration adds data to data file from temporary list
        # and clears temporary list
        
        if (i + 1) % 50 == 0:
            df = df.append(df_list)
            df_list.clear()
            print('Intermediate save point reached (writing over)')
            df.to_pickle(file_path)
            print('Total number of failed syndroms:', failed_syndroms)
        
        # If the desired amount of errors have been achieved, break the loop and finish up
        if failed_syndroms == fixed_errors:
            print('Desired amount of failed syndroms achieved, stopping data generation.')
            break

    # Adds any remaining data from temporary list to data file when run is over
    if len(df_list) > 0:
        df = df.append(df_list)
        print('\nSaving all generated data (writing over)')
        df.to_pickle(file_path)
    
    print('\nCompleted')


if __name__ == '__main__':
    # Get job array id, working directory
    job_id = os.getenv('SLURM_ARRAY_JOB_ID')
    array_id = os.getenv('SLURM_ARRAY_TASK_ID')
    local_dir = os.getenv('TMPDIR')

    # Use environment variables to get parameters
    size = int(os.getenv('CODE_SIZE'))
    code = str(os.getenv('CODE_TYPE'))
    alpha = float(os.getenv('CODE_ALPHA'))

    job_name = str(os.getenv('JOB_NAME'))
    start_p = float(os.getenv('START_P'))
    end_p = float(os.getenv('END_P'))
    num_p = int(os.getenv('NUM_P'))
    mwpm_init = bool(int(os.getenv('MWPM_INIT')))
    p_sampling = float(os.getenv('P_SAMPLE'))

    alg = str(os.getenv('ALGORITHM'))
    only_shortest = bool(int(os.getenv('ONLY_SHORTEST')))

    params = {'code': code,
            'method': alg,
            'size': size,
            'noise': 'alpha',
            'p_error': np.linspace(start_p, end_p, num=num_p)[int(array_id)],
            'eta': 0.5,
            'alpha': alpha,
            'p_sampling': p_sampling,
            'droplets': 1,
            'mwpm_init': mwpm_init,
            'fixed_errors':None,
            'Nc': None,
            'iters': 10,
            'conv_criteria': 'error_based',
            'SEQ': 2,
            'TOPS': 10,
            'eps': 0.01,
            'bias': 'Z',
            'onlyshortest': only_shortest}
    # Steps is a function of code size L
    params.update({'steps': int(5*params['size']**5)})
    
    print('Nbr of steps to take if applicable:', params['steps'])

    # Build file path
    file_path = os.path.join(local_dir, f'data_paper_{job_name}_{job_id}_{array_id}.xz')
    
    # Generate data
    generate(file_path, params, nbr_datapoints=10000, fixed_errors=params['fixed_errors'])
