
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import time 

def run_limited_random_walk(x, s, max_steps, sigma):
    for step in range(max_steps):
        x += np.random.randn() * sigma
        if x > s:
            return 1
    return 0

def run_limited_random_walks(num_runs, s, max_steps, sigma=1):
    right_count = 0.
    starts = (np.random.rand(num_runs) - .5) * 2 * s
    # starts = s - np.random.rand(num_runs) * 100
    for x in starts:
        right_count += run_limited_random_walk(x, s, max_steps, sigma)
    return right_count / num_runs
    # return right_count / num_runs * 100 / (2 * s)

def run_random_walk(x, s, sigma):
    c, r = 0, True
    while True:
        steps = np.random.randn(s)
        for step in steps:
            if x > s:
                r = True
                return c, r
            elif x < -s:
                r = False
                return c, r
            c += 1
            x += step * sigma

def run_random_walks(start, num_runs, s, sigma):
    cs = np.zeros(num_runs)
    right_count = 0.
    for run in range(num_runs):
        cs[run], r = run_random_walk(start, s, sigma)
        right_count += 1 if r else 0
    return cs, right_count / num_runs

def run_random_walks_from_starts(num_starts, num_runs, s, sigma=1):
    starts = np.linspace(-s, s, num_starts)
    right_probs = np.empty(num_starts)
    counts = np.empty(num_starts)
    for i, start in enumerate(starts):
        st = time.time()
        cs, right_prob = run_random_walks(start, num_runs, s, sigma)
        right_probs[i] = right_prob
        counts[i] = np.mean(cs)
        print('{} \ {}\t sec: {}'.format(start, num_starts, time.time() - st))
    return counts, right_probs

if __name__ == '__main__':
    # # check the exit right/left prob for various start positions
    # counts, right_probs = run_random_walks_from_starts(10, 100, 100)
    # plt.subplot(1,2,1)
    # plt.plot(range(len(right_probs)), right_probs)
    # plt.subplot(1,2,2)
    # plt.plot(range(len(counts)), counts)
    # plt.show()
    
    # check the right hit prob for random starts for some horizon
    right_prob = run_limited_random_walks(num_runs=100000, s=100000, max_steps=200)
    print(right_prob)