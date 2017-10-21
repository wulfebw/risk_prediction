'''
This script extract IDM parameters from individual NGSIM vehicles using 
the `local maximum likelihood`, dicussed e.g., in 
"Calibration of microscopic traffic-flow models using multiple data sources"
By Hoogendoorn.
'''

import h5py
import multiprocessing as mp
import numpy as np
import os
import sys
import time

import idm_parameter_estimation

def maybe_mkdir(d):
    if not os.path.exists(d):
        os.mkdir(d)

def compute_lengths(arr):
    sums = np.sum(np.array(arr), axis=2)
    lengths = []
    for sample in sums:
        zero_idxs = np.where(sample == 0.)[0]
        if len(zero_idxs) == 0:
            lengths.append(len(sample))
        else:
            lengths.append(zero_idxs[0])
    return np.array(lengths)

def remove_collisions_from_x_y(
        xs, 
        ys, 
        col_thresh=10., 
        ttc_thresh=3.):
    # removes indices where the ttc is below a threshold
    # compares distance between vehicles and the length of the rear vehicle
    safe_idxs = []
    for i, x in enumerate(xs):
        ds = x[1]
        # check for near collision
        if ds < col_thresh:
            continue

        # check for ttc below threshold (we know ds > 0 at this point)
        dv = x[3]
        if dv < 0:
            ttc = ds / (abs(dv) + 1e-8)
            if ttc < ttc_thresh:
                continue

        safe_idxs.append(i)

    return xs[safe_idxs], ys[safe_idxs]

def load_simulated_data(
        filepath,
        remove_collisions=True,
        reaction_time_offset=0):

    f = h5py.File(filepath)
    features = f['risk/features'].value
    feature_names = f['risk'].attrs['feature_names']

    idm_feature_names = ['velocity','fore_m_dist', 'length','fore_m_vel']
    idm_feature_idxs = [list(feature_names).index(name) for name in idm_feature_names]

    idm_target_names = ['accel']
    idm_target_idxs = [i for (i,name) in enumerate(feature_names) if name in idm_target_names]
    idm_beh_names = ['beh_lon_a_max', 'beh_lon_desired_velocity', 'beh_lon_s_min', 'beh_lon_T','beh_lon_d_cmf']
    idm_beh_idxs = [list(feature_names).index(name) for name in idm_beh_names]

    idm_features = features[:,:,idm_feature_idxs]
    idm_features[:,:,-1] = idm_features[:,:,-1] - idm_features[:,:,0] 
    idm_targets = features[:,:,idm_target_idxs]
    
    xs, ys = [], []
    for i in range(len(idm_features)):
        x = idm_features[i]
        y = idm_targets[i]

        if remove_collisions:
            x, y = remove_collisions_from_x_y(x, y)

        if reaction_time_offset > 0:
            x = x[:-reaction_time_offset]
            y = y[reaction_time_offset:]

        xs.append(x)
        ys.append(y)

    thetas = features[:,:,idm_beh_idxs]

    return xs, ys, thetas

def load_ngsim_data(
        filepath,
        remove_collisions=True,
        idm_id_idx=0,
        idm_feature_idxs=[1,3,2,4],
        idm_target_idx=5,
        reaction_time_offset=0):
    
    infile = h5py.File(filepath, 'r')
    traj_key = None
    if traj_key is None:
        features = np.vstack([infile[k].value for k in infile.keys()])

    idm_ego_ids = features[:,0,idm_id_idx]
    idm_features = features[:,:,idm_feature_idxs]
    # compute relative velocity
    idm_features[:,:,-1] = idm_features[:,:,-1] - idm_features[:,:,0] 
    idm_targets = features[:,:,idm_target_idx]
    idm_lengths = compute_lengths(idm_features)

    xs = []
    ys = []
    for i in range(len(idm_features)):

        x = idm_features[i,:idm_lengths[i]]
        y = idm_targets[i,:idm_lengths[i]]

        if remove_collisions:
            x, y = remove_collisions_from_x_y(x, y)

        if reaction_time_offset > 0:
            x = x[:-reaction_time_offset]
            y = y[reaction_time_offset:]

        xs.append(x)
        ys.append(y)

    return xs, ys, idm_ego_ids

def extract_idm_params_from_x_y(
        ego_id,
        return_dict,
        x, 
        y,
        a_bounds,
        v0_bounds,
        s0_bounds,
        T_bounds,
        b_bounds,
        min_len=1):
        
        if len(x) < min_len:
            return_dict[ego_id] = None

        else:
            result = idm_parameter_estimation.estimate_idm_params(
                x, y,
                a_bounds=a_bounds,
                v0_bounds=v0_bounds,
                s0_bounds=s0_bounds,
                T_bounds=T_bounds,
                b_bounds=b_bounds
            )

            return_dict[ego_id] = result

def extract_idm_params(
        xs, 
        ys,
        ids,
        processes=1,
        a_bounds=(.2,6.),
        v0_bounds=(5.,90.),
        s0_bounds=(.2,12.),
        T_bounds=(.1,6.),
        b_bounds=(.1,5.)):
    manager = mp.Manager()
    results = manager.dict()
    with mp.Pool(processes=processes) as pool:
        st = time.time()
        n_veh = len(ids)
        pool_results = []
        for i, ego_id in enumerate(ids):
            
            pool_result = pool.apply_async(
                extract_idm_params_from_x_y,
                args=(
                    ego_id, 
                    results, 
                    xs[i], 
                    ys[i],
                    a_bounds,
                    v0_bounds,
                    s0_bounds,
                    T_bounds,
                    b_bounds
                )
            )
            pool_results.append(pool_result)

        for i, pool_result in enumerate(pool_results):
            sys.stdout.write('\r{} / {} time: {:.5f}'.format(i+1, n_veh, time.time() - st))
            pool_result.get()

    return dict(results)
   
if __name__ == '__main__':

    # ngsim
    filepath = '../../data/datasets/oct/ngsim_feature_trajectories_behavior_inferece.h5'
    results_filepath = '../../data/datasets/ngsim_idm_parameters/results.npy'
    xs, ys, ids = load_ngsim_data(
        filepath, 
        remove_collisions=True,
        reaction_time_offset=4
    )
    processes = 20
    results = extract_idm_params(xs, ys, ids, processes)

    # save the results
    np.save(results_filepath, results)

    losses = np.array([r.fun for (k,r) in results.items() if r is not None])
    print('\nmean loss: {}'.format(np.mean(losses)))

    # thetas = np.array([r.x for (k,r) in results.items()])
    # idm_parameter_estimation.plot_theta(thetas)

    # # simulated
    # directory = '../../data/datasets/oct/'
    # # filepath = os.path.join(directory, 'estimation_without_noise_without_reaction_time.h5')
    # # filepath = os.path.join(directory, 'estimation_without_noise_with_reaction_time_.3.h5')
    # # filepath = os.path.join(directory, 'estimation_with_noise_with_reaction_time_.3.h5')
    # # filepath = os.path.join(directory, 'estimation_debug_without_noise_without_reaction_time.h5')
    # filepath = os.path.join(directory, 'estimation_debug_with_noise_with_reaction_time.h5')
    # xs, ys, thetas = load_simulated_data(
    #     filepath, 
    #     remove_collisions=True,
    #     reaction_time_offset=3)
    # results = extract_idm_params(xs, ys)
