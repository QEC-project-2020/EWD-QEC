import numpy as np
import copy
from math import log, exp
from multiprocessing import Pool

from src.mcmc import Chain, Ladder
from src.mcmc_biased import Chain_biased, Ladder_biased
from src.mcmc_alpha import Chain_alpha, Ladder_alpha


def MCMC(init_code, p, Nc=None, SEQ=2, TOPS=10, tops_burn=2, eps=0.1, steps=50000000, iters=10, conv_criteria='error_based'):
    '''
    Original MCMC Parallel tempering method as descibed in high threshold paper
    Parameters also adapted from that paper.
    steps has an upper limit on 50 000 000, which should not be met during operation
    '''
    # either 4 or 16 depending on choice of code topology
    nbr_eq_classes = init_code.nbr_eq_classes

    # If not specified, use size as per paper
    Nc = Nc or init_code.system_size

    # Warn about incorrect parameter inputs
    if tops_burn >= TOPS:
        print('tops_burn has to be smaller than TOPS')

    # initialize variables
    since_burn = 0
    resulting_burn_in = 0
    nbr_errors_bottom_chain = np.zeros(steps)

    # list of class counts after burn in
    eq = np.zeros([steps, nbr_eq_classes], dtype=np.uint32)

    # used in error_based/majority_based instead of setting tops0 = TOPS
    conv_start = 0
    conv_streak = 0

    # Convergence flag
    convergence_reached = False

    # initialize ladder of chains sampled at different temperatures
    ladder = Ladder(p, init_code, Nc, 0.5)

    # Main loop that runs until convergence or max steps (steps) are reached
    for step in range(steps):
        # run metropolis on every chain and perform chain swaps
        ladder.step(iters)

        # Get sample from eq-class of chain in lowest layer of ladder
        current_eq = ladder.chains[0].code.define_equivalence_class()

        # Start saving stats once burn-in period is over
        if ladder.tops0 >= tops_burn:
            since_burn = step - resulting_burn_in

            eq[since_burn] = eq[since_burn - 1]
            eq[since_burn][current_eq] += 1
            nbr_errors_bottom_chain[since_burn] = ladder.chains[0].code.count_errors()
        else:
            # number of steps until tops0 = 2
            resulting_burn_in += 1

        # Check for convergence every 10 samples if burn-in period is over (and conv-crit is set)
        if conv_criteria == 'error_based' and ladder.tops0 >= TOPS:
            accept, convergence_reached = conv_crit_error_based_PT(nbr_errors_bottom_chain, since_burn, conv_streak, SEQ, eps)
            if accept:
                if convergence_reached:
                    break
                conv_streak = ladder.tops0 - conv_start
            else:
                conv_streak = 0
                conv_start = ladder.tops0
    
    # print warning if loop is exited without convergence
    else:
        if conv_criteria == 'error_based':
            print('\n\nWARNING: MCMC hit max number of steps before convergence:\t', step + 1, '\n\n')

    return (np.divide(eq[since_burn], since_burn + 1) * 100).astype(np.uint8)


def conv_crit_error_based_PT(nbr_errors_bottom_chain, since_burn, tops_accepted, SEQ, eps):
    '''
    convergence criteria used in paper and called ''felkriteriet''
    '''
    # last nonzero element of nbr_errors_bottom_chain is since_burn. Length of nonzero part is since_burn + 1
    l = since_burn + 1
    # Calculate average number of errors in 2nd and 4th quarter
    Average_Q2 = np.average(nbr_errors_bottom_chain[(l // 4): (l // 2)])
    Average_Q4 = np.average(nbr_errors_bottom_chain[(3 * l // 4): l])

    # Compare averages
    error = abs(Average_Q2 - Average_Q4)
    if error < eps:
        return True, tops_accepted >= SEQ
    else:
        return False, False


def single_temp(init_code, p, max_iters):
    # check if init_code is provided as a list of inits for different classes
    if type(init_code) == list:
        nbr_eq_classes = init_code[0].nbr_eq_classes
        # make sure one init per class is provided
        assert len(init_code) == nbr_eq_classes, 'if init_code is a list, it has to contain one code for each class'
        # initiate ladder
        chains = [Chain(p, copy.deepcopy(code)) for code in init_code]

    # if init_code is a single code, inits for every class have to be generated
    else:
        nbr_eq_classes = init_code.nbr_eq_classes
        chains = [None] * nbr_eq_classes # list of chain objects
        for eq in range(nbr_eq_classes):
            chains[eq] = Chain(p, copy.deepcopy(init_code))
            chains[eq].code.qubit_matrix = chains[eq].code.to_class(eq) # apply different logical operator to each chain

    nbr_errors_chain = np.zeros((nbr_eq_classes, max_iters))
    mean_array = np.zeros(nbr_eq_classes, dtype=float)

    for eq in range(nbr_eq_classes):
        for j in range(max_iters):
            chains[eq].update_chain_fast(5)
            nbr_errors_chain[eq ,j] = chains[eq].code.count_errors()
            if j == max_iters-1:
                mean_array[eq] = np.average(nbr_errors_chain[eq ,:j])

    return mean_array


def single_temp_alpha(init_code, pz_tilde, alpha, max_iters):
    # check if init_code is provided as a list of inits for different classes
    if type(init_code) == list:
        nbr_eq_classes = init_code[0].nbr_eq_classes
        # make sure one init per class is provided
        assert len(init_code) == nbr_eq_classes, 'if init_code is a list, it has to contain one code for each class'
        chains = [Chain_alpha(copy.deepcopy(code), pz_tilde, alpha) for code in init_code]

    # if init_code is a single code, inits for every class have to be generated
    else:
        nbr_eq_classes = init_code.nbr_eq_classes
        chains = [None] * nbr_eq_classes # list of chain objects
        for eq in range(nbr_eq_classes):
            chains[eq] = Chain_alpha(copy.deepcopy(init_code), pz_tilde, alpha)
            chains[eq].code.qubit_matrix = chains[eq].code.to_class(eq) # apply different logical operator to each chain

    nbr_errors_chain = np.zeros((nbr_eq_classes, max_iters))
    mean_array = np.zeros(nbr_eq_classes, dtype=float)

    for eq in range(nbr_eq_classes):
        for j in range(max_iters):
            chains[eq].update_chain_fast(5)
            nx, ny, nz = chains[eq].code.chain_lengths()
            n_eff = nz + alpha*(nx + ny)
            nbr_errors_chain[eq ,j] = n_eff
            if j == max_iters-1:
                mean_array[eq] = np.average(nbr_errors_chain[eq ,:j])

    return mean_array


def EWD_droplet(chain, steps, randomize, conv_mult):
    # All unique chains will be saved in samples
    samples = {}

    # Convergence variables
    stop = steps
    shortest = 2 * chain.code.system_size ** 2

    # Start in high energy state
    if randomize:
        chain.code.qubit_matrix = chain.code.apply_stabilizers_uniform()

    # Do the metropolis steps and add to samples if new chains are found
    for step in range(int(steps)):
        chain.update_chain_fast(5)
        key = hash(chain.code.qubit_matrix.tobytes())
        if key not in samples:
            length = chain.code.count_errors()
            samples[key] = length

            # if new shortest chain found, extend sampling time
            if conv_mult and length <= shortest:
                shortest = length
                stop = step * conv_mult
        
        # if no new shortest chain found, end sampling
        if conv_mult and step >= stop and step * 100 >= steps:
            break

    return samples


def EWD(init_code, p_error, p_sampling=None, droplets=10, steps=20000, conv_mult=0):
    # set p_sampling equal to p_error by default
    p_sampling = p_sampling or p_error

    if type(init_code) == list:
        # this is either 4 or 16, depending on what type of code is used.
        nbr_eq_classes = init_code[0].nbr_eq_classes
        # make sure one init code is provided for each class
        assert len(init_code) == nbr_eq_classes, 'if init_code is a list, it has to contain one code for each class'
        eq_chains = [Chain(p_sampling, copy.deepcopy(code)) for code in init_code]
        # don't apply uniform stabilizers if low energy inits are provided
        randomize = False

    else:
        # this is either 4 or 16, depending on what type of code is used.
        nbr_eq_classes = init_code.nbr_eq_classes
        # Create chain with p_sampling, this is allowed since N(n) is independet of p.
        eq_chains = [None] * nbr_eq_classes
        for eq in range(nbr_eq_classes):
            eq_chains[eq] = Chain(p_sampling, copy.deepcopy(init_code))
            eq_chains[eq].code.qubit_matrix = eq_chains[eq].code.to_class(eq)
        # apply uniform stabilizers, i.e. rain
        randomize = True

    # this is where we save all samples in a dict, to find the unique ones.
    qubitlist = {}

    # Z_E will be saved in eqdistr
    eqdistr = np.zeros(nbr_eq_classes)

    # error-model
    beta = -log((p_error / 3) / (1 - p_error))

    if droplets > 1:
        pool = Pool(droplets)

    for eq in range(nbr_eq_classes):
        # go to class eq and apply stabilizers
        chain = eq_chains[eq]

        if droplets == 1:
            qubitlist = EWD_droplet(copy.deepcopy(chain), steps, randomize, conv_mult)
        else:
            args = [(copy.deepcopy(chain), steps, randomize, conv_mult) for _ in range(droplets)]
            output = pool.starmap_async(EWD_droplet, args).get()
            for res in output:
                qubitlist.update(res)

        # compute Z_E
        for key in qubitlist:
            eqdistr[eq] += exp(-beta * qubitlist[key])
        qubitlist.clear()

    # Retrun normalized eq_distr
    return (np.divide(eqdistr, sum(eqdistr)) * 100)


def EWD_droplet_general_noise(chain, steps, randomize):
    # All unique chains will be saved in samples
    samples = {}

    # Start in high energy state
    if randomize:
        chain.code.qubit_matrix = chain.code.apply_stabilizers_uniform()

    # Do the metropolis steps and add to samples if new chains are found
    for step in range(int(steps)):
        chain.update_chain_fast(5)
        # TODO needs to consider noise here (not use fast variant?)
        key = hash(chain.code.qubit_matrix.tobytes())
        if key not in samples:
            lengths = chain.code.count_errors_xyz()
            samples[key] = lengths

    return samples


def EWD_general_noise(init_code, p_xyz, p_sampling=None, droplets=10, steps=20000, shortest_only=False):
    # p_xyz is an array (p_x, p_y, p_z)
    # set p_sampling equal to sum of p_xyz by default
    if p_sampling is None:
        p_sampling = p_xyz.sum()

    if type(p_sampling) == np.ndarray:
        chain_class = Chain_xyz
    else:
        chain_class = Chain

    if type(init_code) == list:
        # this is either 4 or 16, depending on what type of code is used.
        nbr_eq_classes = init_code[0].nbr_eq_classes
        # make sure one init code is provided for each class
        assert len(init_code) == nbr_eq_classes, 'if init_code is a list, it has to contain one code for each class'
        eq_chains = [chain_class(p_sampling, copy.deepcopy(code)) for code in init_code]

        # don't apply uniform stabilizers if low energy inits are provided
        randomize = False

    else:
        # this is either 4 or 16, depending on what type of code is used.
        nbr_eq_classes = init_code.nbr_eq_classes
        # Create chain with p_sampling, this is allowed since N(n) is independet of p.
        eq_chains = [None] * nbr_eq_classes
        for eq in range(nbr_eq_classes):
            tmp_code = copy.deepcopy(init_code)
            tmp_code.qubit_matrix = tmp_code.to_class(eq)
            eq_chains[eq] = chain_class(p_sampling, tmp_code)
        # apply uniform stabilizers, i.e. rain
        randomize = False

    # this is where we save all samples in a dict, to find the unique ones.
    qubitlist = {}

    # Z_E will be saved in eqdistr
    eqdistr = np.zeros(nbr_eq_classes)

    # deal with infinities
    check_finite = np.any(p_xyz == 0)
    p_infinite = (p_xyz == 0)

    # error-model
    beta = -np.log((p_xyz) / (1 - sum(p_xyz)))

    if droplets > 1:
        pool = Pool(droplets)

    for eq in range(nbr_eq_classes):
        # go to class eq and apply stabilizers
        chain = eq_chains[eq]

        if droplets == 1:
            qubitlist = EWD_droplet_general_noise(copy.deepcopy(chain), steps, randomize)
        else:
            args = [(copy.deepcopy(chain), steps, randomize) for _ in range(droplets)]
            output = pool.starmap_async(EWD_droplet_general_noise, args).get()
            for res in output:
                qubitlist.update(res)

        qubit_lengths = np.array(list(qubitlist.values()))

        # beta and the elements of qubit_lenghts are arrays whose dot product corresponds to the chain (log) probability
        weighted_lengths = np.sum(beta * qubit_lengths, axis=1, where=(qubit_lengths > 0))

        # if only the shortest chains should be used, drop all longer chains
        # might want to choose better tolerances for np.isclose
        if shortest_only:
            weighted_lengths = weighted_lengths[np.isclose(weighted_lengths, np.min(weighted_lengths))]

        # compute Z_E
        eqdistr[eq] = np.sum(np.exp(-weighted_lengths))
        qubitlist.clear()
        # deal with infinities
        #if check_finite:
        #    for counts in qubitlist.values():
        #        # if p_i = 0, a chain with i-errors has probability 0 and need not be counted
        #        if not np.any(counts[p_infinite]):
        #            eqdistr[eq] += np.exp(-np.sum(beta * counts))
        ## if all p are nonzero, no need to deal with infinities
        #else:
        #    for key in qubitlist:
        #        eqdistr[eq] += np.exp(-np.sum(beta * qubitlist[key]))
        #qubitlist.clear()

    # Retrun normalized eq_distr
    return (np.divide(eqdistr, sum(eqdistr)) * 100)


def EWD_general_noise_shortest(init_code, p_xyz, p_sampling=None, droplets=10, steps=20000):
    # p_xyz is an array (p_x, p_y, p_z)
    # set p_sampling equal to sum of p_xyz by default
    if p_sampling is None:
        p_sampling = p_xyz.sum()

    if type(p_sampling) == np.ndarray:
        chain_class = Chain_xyz
    else:
        chain_class = Chain

    if type(init_code) == list:
        # this is either 4 or 16, depending on what type of code is used.
        nbr_eq_classes = init_code[0].nbr_eq_classes
        # make sure one init code is provided for each class
        assert len(init_code) == nbr_eq_classes, 'if init_code is a list, it has to contain one code for each class'
        eq_chains = [chain_class(p_sampling, copy.deepcopy(code)) for code in init_code]

        # don't apply uniform stabilizers if low energy inits are provided
        randomize = False

    else:
        # this is either 4 or 16, depending on what type of code is used.
        nbr_eq_classes = init_code.nbr_eq_classes
        # Create chain with p_sampling, this is allowed since N(n) is independet of p.
        eq_chains = [None] * nbr_eq_classes
        for eq in range(nbr_eq_classes):
            tmp_code = copy.deepcopy(init_code)
            tmp_code.qubit_matrix = tmp_code.to_class(eq)
            eq_chains[eq] = chain_class(p_sampling, tmp_code)
        # apply uniform stabilizers, i.e. rain
        randomize = False

    # this is where we save all samples in a dict, to find the unique ones.
    qubitlist = {}

    # Z_E will be saved in eqdistr
    eqdistr = np.zeros(nbr_eq_classes)
    eqdistr_shortest = np.zeros(nbr_eq_classes)

    # deal with infinities
    check_finite = np.any(p_xyz == 0)
    p_infinite = (p_xyz == 0)

    # error-model
    beta = -np.log((p_xyz) / (1 - sum(p_xyz)))

    if droplets > 1:
        pool = Pool(droplets)

    for eq in range(nbr_eq_classes):
        # go to class eq and apply stabilizers
        chain = eq_chains[eq]

        if droplets == 1:
            qubitlist = EWD_droplet_general_noise(copy.deepcopy(chain), steps, randomize)
        else:
            args = [(copy.deepcopy(chain), steps, randomize) for _ in range(droplets)]
            output = pool.starmap_async(EWD_droplet_general_noise, args).get()
            for res in output:
                qubitlist.update(res)

        qubit_lengths = np.array(list(qubitlist.values()))

        # beta and the elements of qubit_lenghts are arrays whose dot product corresponds to the chain (log) probability
        weighted_lengths = np.sum(beta * qubit_lengths, axis=1, where=(qubit_lengths > 0))

        # compute Z_E
        eqdistr[eq] = np.sum(np.exp(-weighted_lengths))
        eqdistr_shortest[eq] = np.sum(np.exp(-weighted_lengths), where=np.isclose(weighted_lengths, np.min(weighted_lengths)))
        qubitlist.clear()

    # Retrun normalized eq_distr
    return (np.divide(eqdistr, sum(eqdistr)) * 100), (np.divide(eqdistr_shortest, sum(eqdistr_shortest)) * 100)


def EWD_droplet_alpha(chain, steps, alpha, onlyshortest):

    #chain.code.qubit_matrix = chain.code.apply_stabilizers_uniform()
    # All unique chains will be saved in samples
    all_seen = set()
    seen_chains = {}
    shortest = 1000000
    # Do the metropolis steps and add to samples if new chains are found
    for _ in range(steps):
        chain.update_chain_fast(5)
        key = hash(chain.code.qubit_matrix.tobytes())
        if key not in all_seen:
            all_seen.add(key)
            lengths = chain.code.chain_lengths()
            eff_len = lengths[2] + alpha * sum(lengths[0:2])
            if onlyshortest:
                if eff_len < shortest: # New shortest chain
                    shortest = eff_len
                    seen_chains = {}
                    seen_chains[key] = eff_len
                elif eff_len == shortest:
                    seen_chains[key] = eff_len
            else:
                seen_chains[key] = eff_len
    
    return seen_chains


def EWD_alpha_N_n(init_code, pz_tilde, alpha, steps, pz_tilde_sampling=None, onlyshortest=True):

    pz_tilde_sampling = pz_tilde_sampling if pz_tilde_sampling is not None else pz_tilde

    if type(init_code) == list:
        nbr_eq_classes = init_code[0].nbr_eq_classes
        # make sure one init code is provided for each class
        assert len(init_code) == nbr_eq_classes, 'if init_code is a list, it has to contain one code for each class'
        eq_chains = [Chain_alpha(copy.deepcopy(code), pz_tilde_sampling, alpha) for code in init_code]

    else:
        nbr_eq_classes = init_code.nbr_eq_classes
        # Create chain with p_sampling, this is allowed since N(n) is independent of p.
        eq_chains = [None] * nbr_eq_classes
        for eq in range(nbr_eq_classes):
            eq_chains[eq] = Chain_alpha(copy.deepcopy(init_code), pz_tilde_sampling, alpha)
            eq_chains[eq].code.qubit_matrix = eq_chains[eq].code.to_class(eq)

    # Z_E will be saved in eqdistr
    eqdistr = np.zeros(nbr_eq_classes)

    beta = - np.log(pz_tilde)

    Nobs_n = [{}, {}, {}, {}]

    for eq in range(nbr_eq_classes):
        # go to class eq and apply stabilizers
        chain = eq_chains[eq]

        out = EWD_droplet_alpha(chain, steps, alpha, onlyshortest)

        Nobs_n[eq] = out

        # for eff_len in out.values():
        #     eqdistr[eq] += exp(-beta*eff_len)
        # out.clear()

    return Nobs_n#(np.divide(eqdistr, sum(eqdistr)) * 100)


def EWD_alpha(init_code, pz_tilde, alpha, steps, pz_tilde_sampling=None, onlyshortest=True):

    pz_tilde_sampling = pz_tilde_sampling if pz_tilde_sampling is not None else pz_tilde

    if type(init_code) == list:
        nbr_eq_classes = init_code[0].nbr_eq_classes
        # make sure one init code is provided for each class
        assert len(init_code) == nbr_eq_classes, 'if init_code is a list, it has to contain one code for each class'
        eq_chains = [Chain_alpha(copy.deepcopy(code), pz_tilde_sampling, alpha) for code in init_code]

    else:
        nbr_eq_classes = init_code.nbr_eq_classes
        # Create chain with p_sampling, this is allowed since N(n) is independent of p.
        eq_chains = [None] * nbr_eq_classes
        for eq in range(nbr_eq_classes):
            eq_chains[eq] = Chain_alpha(copy.deepcopy(init_code), pz_tilde_sampling, alpha)
            eq_chains[eq].code.qubit_matrix = eq_chains[eq].code.to_class(eq)

    # Z_E will be saved in eqdistr
    eqdistr = np.zeros(nbr_eq_classes)

    beta = - np.log(pz_tilde)

    for eq in range(nbr_eq_classes):
        # go to class eq and apply stabilizers
        chain = eq_chains[eq]

        out = EWD_droplet_alpha(chain, steps, alpha, onlyshortest)

        for eff_len in out.values():
            eqdistr[eq] += exp(-beta*eff_len)
        out.clear()

    return (np.divide(eqdistr, sum(eqdistr)) * 100)



def MCMC_biased(init_code, p, eta=0.5, Nc=None, SEQ=2, TOPS=10, tops_burn=2, eps=0.1, steps=50000000, iters=10, conv_criteria='error_based'):
    nbr_eq_classes = init_code.nbr_eq_classes
    Nc = Nc or init_code.system_size
    if tops_burn >= TOPS:
        print('tops_burn has to be smaller than TOPS')
    # initialize variables
    since_burn = 0
    resulting_burn_in = 0
    nbr_errors_bottom_chain = np.zeros(steps)
    # list of class counts after burn in
    eq = np.zeros([steps, nbr_eq_classes], dtype=np.uint32)
    # used in error_based/majority_based instead of setting tops0 = TOPS
    conv_start = 0
    conv_streak = 0
    # Convergence flag
    convergence_reached = False
    # initialize ladder of chains sampled at different temperatures
    ladder = Ladder_biased(p, init_code, eta, Nc, 0.5)
    # Main loop that runs until convergence or max steps (steps) are reached
    for step in range(steps):
        # run metropolis on every chain and perform chain swaps
        ladder.step(iters)
        # Get sample from eq-class of chain in lowest layer of ladder
        current_eq = ladder.chains[0].code.define_equivalence_class()
        # Start saving stats once burn-in period is over
        if ladder.tops0 >= tops_burn:
            since_burn = step - resulting_burn_in
            eq[since_burn] = eq[since_burn - 1]
            eq[since_burn][current_eq] += 1
            nbr_errors_bottom_chain[since_burn] = ladder.chains[0].code.count_errors()
        else:
            # number of steps until tops0 = 2
            resulting_burn_in += 1
        # Check for convergence every 10 samples if burn-in period is over (and conv-crit is set)
        if conv_criteria == 'error_based' and ladder.tops0 >= TOPS:
            accept, convergence_reached = conv_crit_error_based_PT_biased(nbr_errors_bottom_chain, since_burn, conv_streak, SEQ, eps)
            if accept:
                if convergence_reached:
                    break
                conv_streak = ladder.tops0 - conv_start
            else:
                conv_streak = 0
                conv_start = ladder.tops0
    # print warning if loop is exited without convergence
    else:
        if conv_criteria == 'error_based':
            print('\n\nWARNING: MCMC hit max number of steps before convergence:\t', step + 1, '\n\n')
    return (np.divide(eq[since_burn], since_burn + 1) * 100).astype(np.uint8)


# convergence criteria used in paper and called ''felkriteriet''
def conv_crit_error_based_PT_biased(nbr_errors_bottom_chain, since_burn, tops_accepted, SEQ, eps):
    # last nonzero element of nbr_errors_bottom_chain is since_burn. Length of nonzero part is since_burn + 1
    l = since_burn + 1
    # Calculate average number of errors in 2nd and 4th quarter
    Average_Q2 = np.average(nbr_errors_bottom_chain[(l // 4): (l // 2)])
    Average_Q4 = np.average(nbr_errors_bottom_chain[(3 * l // 4): l])
    # Compare averages
    error = abs(Average_Q2 - Average_Q4)
    if error < eps:
        return True, tops_accepted >= SEQ
    else:
        return False, False


def MCMC_alpha_with_shortest(init_code, pz_tilde, alpha=1, Nc=None, SEQ=2, TOPS=10, tops_burn=2, eps=0.1, steps=50000000, iters=10, conv_criteria='error_based'):
    nbr_eq_classes = init_code.nbr_eq_classes
    Nc = Nc or init_code.system_size
    if tops_burn >= TOPS:
        print('tops_burn has to be smaller than TOPS')
    # initialize variables
    since_burn = 0
    resulting_burn_in = 0
    nbr_errors_bottom_chain = np.zeros(steps)
    # list of class counts after burn in
    eq = np.zeros([steps, nbr_eq_classes], dtype=np.uint32)
    # used in error_based/majority_based instead of setting tops0 = TOPS
    conv_start = 0
    conv_streak = 0
    # Convergence flag
    convergence_reached = False
    # initialize ladder of chains sampled at different temperatures
    ladder = Ladder_alpha(pz_tilde, init_code, alpha, Nc, 0.5)

    unique_chains = [{}, {}, {}, {}]
    shortest_n = [0, 0, 0, 0]
    shortest = [100000, 100000, 100000, 100000]


    # Main loop that runs until convergence or max steps (steps) are reached
    for step in range(steps):
        # run metropolis on every chain and perform chain swaps
        ladder.step(iters)
        # Get sample from eq-class of chain in lowest layer of ladder
        current_eq = ladder.chains[0].code.define_equivalence_class()
        # Start saving stats once burn-in period is over
        if ladder.tops0 >= tops_burn:
            since_burn = step - resulting_burn_in
            eq[since_burn] = eq[since_burn - 1]
            eq[since_burn][current_eq] += 1
            nbr_errors_bottom_chain[since_burn] = ladder.chains[0].n_eff

            if nbr_errors_bottom_chain[since_burn] < shortest[current_eq]:
                shortest_n[current_eq] = 1
                
                shortest[current_eq] = nbr_errors_bottom_chain[since_burn]
                unique_chains[current_eq] = {}
                
                qubit_hash = hash(ladder.chains[0].code.qubit_matrix.tobytes())
                if qubit_hash not in unique_chains[current_eq].keys():
                    unique_chains[current_eq][qubit_hash] = nbr_errors_bottom_chain[since_burn]
            elif nbr_errors_bottom_chain[since_burn] == shortest[current_eq]:
                shortest_n[current_eq] += 1
                
                qubit_hash = hash(ladder.chains[0].code.qubit_matrix.tobytes())
                if qubit_hash not in unique_chains[current_eq].keys():
                    unique_chains[current_eq][qubit_hash] = nbr_errors_bottom_chain[since_burn]
        else:
            # number of steps until tops0 = 2
            resulting_burn_in += 1
        # Check for convergence every 10 samples if burn-in period is over (and conv-crit is set)
        if conv_criteria == 'error_based' and ladder.tops0 >= TOPS:
            accept, convergence_reached = conv_crit_error_based_PT_alpha(nbr_errors_bottom_chain, since_burn, conv_streak, SEQ, eps)
            if accept:
                if convergence_reached:
                    break
                conv_streak = ladder.tops0 - conv_start
            else:
                conv_streak = 0
                conv_start = ladder.tops0
    # print warning if loop is exited without convergence
    else:
        if conv_criteria == 'error_based':
            print('\n\nWARNING: MCMC hit max number of steps before convergence:\t', step + 1, '\n\n')

    eqdistr = np.zeros(nbr_eq_classes)

    beta = - np.log(pz_tilde)

    for eq_n in range(nbr_eq_classes):
        for eff_len in unique_chains[eq_n].values():
            eqdistr[eq_n] += exp(-beta*eff_len)


    return (np.divide(eq[since_burn], since_burn + 1) * 100).astype(np.uint8), (np.divide(eqdistr, sum(eqdistr)) * 100), (np.array(shortest_n) / sum(shortest_n) * 100)


def MCMC_alpha(init_code, pz_tilde, alpha=1, Nc=None, SEQ=2, TOPS=10, tops_burn=2, eps=0.1, steps=50000000, iters=10, conv_criteria='error_based'):
    nbr_eq_classes = init_code.nbr_eq_classes
    Nc = Nc or init_code.system_size
    if tops_burn >= TOPS:
        print('tops_burn has to be smaller than TOPS')
    # initialize variables
    since_burn = 0
    resulting_burn_in = 0
    nbr_errors_bottom_chain = np.zeros(steps)
    # list of class counts after burn in
    eq = np.zeros([steps, nbr_eq_classes], dtype=np.uint32)
    # used in error_based/majority_based instead of setting tops0 = TOPS
    conv_start = 0
    conv_streak = 0
    # Convergence flag
    convergence_reached = False
    # initialize ladder of chains sampled at different temperatures
    ladder = Ladder_alpha(pz_tilde, init_code, alpha, Nc, 0.5)
    # Main loop that runs until convergence or max steps (steps) are reached
    for step in range(steps):
        # run metropolis on every chain and perform chain swaps
        ladder.step(iters)
        # Get sample from eq-class of chain in lowest layer of ladder
        current_eq = ladder.chains[0].code.define_equivalence_class()
        # Start saving stats once burn-in period is over
        if ladder.tops0 >= tops_burn:
            since_burn = step - resulting_burn_in
            eq[since_burn] = eq[since_burn - 1]
            eq[since_burn][current_eq] += 1
            nx, ny, nz = ladder.chains[0].code.chain_lengths()
            n_eff = nz + ladder.chains[0].alpha * (nx + ny)
            nbr_errors_bottom_chain[since_burn] = n_eff
        else:
            # number of steps until tops0 = 2
            resulting_burn_in += 1
        # Check for convergence every 10 samples if burn-in period is over (and conv-crit is set)
        if conv_criteria == 'error_based' and ladder.tops0 >= TOPS:
            accept, convergence_reached = conv_crit_error_based_PT_alpha(nbr_errors_bottom_chain, since_burn, conv_streak, SEQ, eps)
            if accept:
                if convergence_reached:
                    break
                conv_streak = ladder.tops0 - conv_start
            else:
                conv_streak = 0
                conv_start = ladder.tops0
    # print warning if loop is exited without convergence
    else:
        if conv_criteria == 'error_based':
            print('\n\nWARNING: MCMC hit max number of steps before convergence:\t', step + 1, '\n\n', flush=True)
    return (np.divide(eq[since_burn], since_burn + 1) * 100).astype(np.uint8), step*iters*Nc


# convergence criteria used in paper and called ''felkriteriet''
def conv_crit_error_based_PT_alpha(nbr_errors_bottom_chain, since_burn, tops_accepted, SEQ, eps):
    # last nonzero element of nbr_errors_bottom_chain is since_burn. Length of nonzero part is since_burn + 1
    l = since_burn + 1
    # Calculate average number of errors in 2nd and 4th quarter
    Average_Q2 = np.average(nbr_errors_bottom_chain[(l // 4): (l // 2)])
    Average_Q4 = np.average(nbr_errors_bottom_chain[(3 * l // 4): l])
    # Compare averages
    error = abs(Average_Q2 - Average_Q4)
    if error < eps:
        return True, tops_accepted >= SEQ
    else:
        return False, False


if __name__ == '__main__':
    from src.mwpm import class_sorted_mwpm, regular_mwpm
    from src.toric_model import Toric_code
    from src.planar_model import Planar_code
    from src.xzzx_model import xzzx_code
    from src.xyz2_model import xyz_code
    import time


    size = 9
    steps = 10 * size ** 5
    p_error = 0.10
    p_sampling = 0.30
    p_xyz = np.array([0.09, 0.01, 0.09])
    init_code = Planar_code(size)
    
    pz_tilde_sampling=0.25
    pz_tilde = 0.3
    alpha=2
    
    tries = 1
    distrs = np.zeros((tries, init_code.nbr_eq_classes))

    for i in range(2):
        init_code.generate_random_error(p_error)

        # p_tilde = pz_tilde + 2*pz_tilde**alpha
        # p_z = pz_tilde*(1-p_tilde)
        # p_x = p_y = pz_tilde**alpha * (1-p_tilde)
        # init_code.generate_random_error(p_x=p_x, p_y=p_y, p_z=p_z)
        
        ground_state = init_code.define_equivalence_class()
        print('Ground state:', ground_state)
        
        #init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
        #init_qubit = np.copy(init_code.qubit_matrix)

        #class_init = class_sorted_mwpm(init_code)
        #mwpm_init = regular_mwpm(init_code)

        print('################ Chain', i+1 , '###################')
        
        for i in range(tries):
            t0 = time.time()
            distrs[i] = EWD(copy.deepcopy(init_code), p_error=p_error, p_sampling=p_sampling, steps=steps, droplets=4, conv_mult=0)
            print('Try EWD       ', i+1, ':', distrs[i], 'most_likely_eq', np.argmax(distrs[i]), 'time:', time.time()-t0)
            t0 = time.time()
            distrs[i] = MCMC(copy.deepcopy(init_code), p=p_error)
            print('Try MCMC       ', i+1, ':', distrs[i], 'most_likely_eq', np.argmax(distrs[i]), 'time:', time.time()-t0)
