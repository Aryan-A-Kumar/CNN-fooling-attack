import numpy as np

def attacker(func, numPix, bounds, success_fun, maxiter = 400):

    cilia_size0 = 0.5
    cilia_size = cilia_size0
    c = 1.0   # step_size/cilia_size
    decay_rho = 1.2
    decay_eta = 0.7

    nDim = len(bounds)

    bounds_ = np.array(bounds)
    lb = bounds_[:,0]
    ub = bounds_[:,1]
    bounds_diff = ub - lb - 1

    def bound(x_in):
        return np.clip(x_in, 0, 1)

    def scale(x_in):
        return (x_in*bounds_diff + lb).astype(int)

    x_best = np.zeros((numPix, nDim))
    f_best = np.ones((numPix, 1))

    x = np.random.random((1, nDim))
    x = bound(x)
    x_scale = scale(x)
    f = func(x_scale)

    x_best[0, :] = x
    f_best[0, :] = f
    # print('Confidence:', f_best)

    # x_store = []
    # f_store = []
    
    for i in range(maxiter):
        dir = np.random.random((1, nDim))-0.5
        dir = dir / np.linalg.norm(dir)

        x_left = x + cilia_size*dir
        x_left = bound(x_left)
        x_left_scale = scale(x_left)
        f_left = func(x_left_scale)

        x_right = x - cilia_size*dir
        x_right = bound(x_right)
        x_right_scale = scale(x_right)
        f_right = func(x_right_scale)

        step_size = c*cilia_size
        x = x - step_size*np.sign(f_left - f_right)*dir
        x = bound(x)
        x_scale = scale(x)
        f = func(x_scale)

        if np.any(f < f_best):
            ind_to_remove = np.argmax(f_best)
            f_best[ind_to_remove, :] = f
            x_best[ind_to_remove, :] = x
            # print('iter:', i, 'Confidence:', f_best)
            if success_fun(scale(x_best).reshape((1,-1)), 0):
                return True
        else:
            best_ind = np.argmin(f_best)
            f = f_best[best_ind, :]
            x = x_best[best_ind, :]
        
        # x_store += [x]
        # f_store += [f]

        # cilia_size *= decay_factor
        # cilia_size = cilia_size0/np.power(1 + decay_rho*i/10, decay_eta)

    return False