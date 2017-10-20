'''
This script extracts IDM parameters from individual NGSIM vehicles using 
the `local maximum likelihood`, dicussed e.g., in 
"Calibration of microscopic traffic-flow models using multiple data sources"
By Hoogendoorn.
'''

import numpy as np
import os
import scipy.optimize

backend = 'Agg' if os.system == 'linux' else 'TkAgg'
import matplotlib
matplotlib.use(backend)
import matplotlib.pyplot as plt

def plot_idm_features(f, bins=100):
    plt.figure(figsize=(16,8))
    plt.subplot(2,2,1)
    plt.hist(f[:,:,0].flatten(), bins=bins)
    plt.title('velocity')
    
    plt.subplot(2,2,2)
    plt.hist(f[:,:,1].flatten(), bins=bins)
    plt.title('fore dist')
    
    plt.subplot(2,2,3)
    plt.hist(f[:,:,2].flatten(), bins=bins)
    plt.title('vehicle length')
    
    plt.subplot(2,2,4)
    plt.hist(f[:,:,3].flatten(), bins=bins)
    plt.title('relative velocity')

    plt.show()

def plot_theta(t, labels = ['a', 'v0', 's0', 'T', 'b'], bins=50):

    for (i, l) in enumerate(labels):
        plt.subplot(2,3,i+1)
        plt.hist(t[:,i], bins=bins)
        plt.title(l)

    plt.tight_layout()
    plt.show()

def build_idm_accel(vi,si,l,dvi):
    def idm_accel(x):       
        a,v0,s0,T,b = x
        s_des = s0 + T*vi - vi*dvi / (2*np.sqrt(a*b))
        return a * (1 - (vi/v0)**4 - (s_des / (si-l))**2)
    return idm_accel

def compute_idm_accel(x,vi,si,l,dvi,clip=100):
    a, v0, s0, T, b = x
    s_des = s0 + T*vi - vi*dvi / (2*np.sqrt(a*b))
    return np.clip(a * (1 - (vi/v0)**4 - (s_des / (si-l))**2), -clip, clip)

def build_simple_test_fn(a):
    def test_fn(x):
        return x * a
    return test_fn

def build_vector_test_fn(b):
    def test_fn(theta):
        return np.dot(theta, b)
    return test_fn

def finite_difference_gradient(f, theta, eps=1e-5):
    grad = np.zeros(len(theta))
    for i in range(len(theta)):
        orig = theta[i]
        theta[i] = orig + eps
        high = f(theta)
        theta[i] = orig - eps
        low = f(theta)
        theta[i] = orig
        grad[i] = (high - low) / (eps * 2)
    return grad

def grad_idm_accel(x, vi, si, l, dvi):
    # common
    a,v0,s0,T,b = x
    ds_des = -(dvi*vi) / (2 * np.sqrt(a * b)) + s0 + T * vi
    sqdist = (si - l) ** 2
    
    # grad a
    a_1 = a * b * dvi * vi * ds_des
    a_2 = 2 * (a * b) ** (3./2) * sqdist
    a_3 = ds_des ** 2 / sqdist
    d_a = -a_1 / a_2 - a_3 - (vi / v0) ** 4 + 1
    
    # grad v0
    d_v0 = 4 * a * vi ** 4 / v0 ** 5
    
    # grad s0
    d_s0 = -2 * a * ds_des / sqdist
    
    # grad T
    d_T = d_s0 * vi
    
    # grad b 
    b_1 = dvi * vi * (2 * np.sqrt(a*b) * (s0 + T * vi) - dvi * vi)
    b_2 = 4 * b ** 2 * (l - si) ** 2
    d_b = - b_1 / b_2
    
    return np.array([d_a, d_v0, d_s0, d_T, d_b])

def rand_idm_constants():
    vi = np.random.uniform(15, 25)
    si = np.random.uniform(10, 100)
    l = np.random.uniform(4, 5)
    dvi = np.random.randn() * .1
    return vi, si, l, dvi

def rand_idm_theta(
        correlated=False,
        a_bounds=(1,3),
        v0_bounds=(30,35),
        s0_bounds=(2,4),
        T_bounds=(.25,1),
        b_bounds=(1,2)):
    if correlated:
        agg = np.random.uniform(0,1)
        a = 1 + agg * (3-1)
        v0 = 30 + agg * (35-30)
        s0 = 2 + agg * (4-2)
        T = .25 + agg * (1-.25)
        b = 1 + agg * (2-1)
    else:
        a = np.random.uniform(*a_bounds)
        v0 = np.random.uniform(*v0_bounds)
        s0 = np.random.uniform(*s0_bounds)
        T = np.random.uniform(*T_bounds)
        b = np.random.uniform(*b_bounds)
    return np.array([a,v0,s0,T,b])

def check_grad_idm_accel(grad_fn, n_itr=10, eps=1e-6):
    diffs = []
    for itr in range(n_itr):
        vi, si, l, dvi = rand_idm_constants()
        idm_accel = build_idm_accel(vi, si, l, dvi)
        theta = rand_idm_theta()
        if theta[0] < 0:
            print(theta)
            break
        pred = grad_fn(theta, vi, si, l, dvi)
        orig_theta = np.copy(theta)
        true = finite_difference_gradient(idm_accel, theta)
        absdiff = np.abs(pred - true)
        if any(np.isnan(absdiff)):
            print(vi, si, l, dvi)
            print(theta)
        diffs.append(absdiff)
    return diffs

def generate_dataset(
        n_timesteps=100, 
        correlated_x_y=False, 
        correlated_x_y_scales=[.05,.01,.005,.005]):
    theta = rand_idm_theta(correlated=False)
    if correlated_x_y:
        xi = rand_idm_constants()
        noise = np.random.randn(n_timesteps, len(xi)) * correlated_x_y_scales
        noise = np.cumsum(noise, axis=0)
        xs = xi + noise
        ys = [compute_idm_accel(theta, *x) for x in xs]
    else:
        xs = []
        ys = []
        for i in range(n_timesteps):
            xi = rand_idm_constants()
            xs.append(xi)
            yi = compute_idm_accel(theta, *xi)
            ys.append(yi)
    return np.array(xs), np.array(ys), theta

def uncorrelated_loss(theta, x, y):
    '''
    theta are the idm parameters
    x are the state variables
    y are the observed accelerations
    '''
    total = 0
    for xi, yi in zip(x,y):
        total += .5 * (yi - compute_idm_accel(theta, *xi)) ** 2
    return total / len(x)

def uncorrelated_gradient(theta, x, y):
    grad = np.zeros_like(theta)
    for (xi,yi) in zip(x,y):
        grad += (yi - compute_idm_accel(theta, *xi)) * -grad_idm_accel(theta, *xi)
    return grad / len(x)

def estimate_idm_params(
        x, 
        y,
        a_bounds=(1.,3.),
        v0_bounds=(30.,35.),
        s0_bounds=(2.,4.),
        T_bounds=(.25,1.),
        b_bounds=(1.,2.)):
    return scipy.optimize.minimize(
        uncorrelated_loss, 
        x0=rand_idm_theta(
            a_bounds=a_bounds,
            v0_bounds=v0_bounds,
            s0_bounds=s0_bounds,
            T_bounds=T_bounds,
            b_bounds=b_bounds
        ), 
        args=(x,y), 
        jac=uncorrelated_gradient,
        bounds=(
            a_bounds,
            v0_bounds,
            s0_bounds,
            T_bounds,
            b_bounds
        )
    )

if __name__ == '__main__':

    test_fn = build_simple_test_fn(2)
    theta = np.array([5.])
    result = finite_difference_gradient(test_fn, theta)
    print('should be 2\n', result)

    test_fn = build_vector_test_fn([1,1])
    theta = np.array([0.,0.])
    result = finite_difference_gradient(test_fn, theta)
    print('should be [1,1]\n', result)

    vi = 1
    si = 10
    l = 5
    dvi = 10
    idm_accel = build_idm_accel(vi, si, l, dvi)
    theta = np.array([3,29,5,1.5,2])
    result = finite_difference_gradient(idm_accel, theta)
    print('finite difference gradient of idm_accel:\n', result)

    diffs = check_grad_idm_accel(grad_idm_accel, n_itr=5000)
    max_diffs = np.max(diffs, axis=0)
    print('difference between finite difference and analytical gradients:\n', max_diffs)
