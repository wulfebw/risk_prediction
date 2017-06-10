import numpy as np

def determine_stationary_dist(p_atoi, p_itoa):
    # could solve the sys of equations
    # or could just simulate it 
    T = np.array([
        [1-p_atoi, p_atoi],
        [p_itoa, 1-p_itoa]
    ])
    pi = np.array([.5,.5])
    for _ in range(1000):
        pi = np.dot(pi, T)
    return pi

if __name__ == '__main__':
    pi = determine_stationary_dist(0.01, 0.3)
    print(pi)