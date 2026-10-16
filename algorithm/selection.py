import numpy as np

def conformal_labeling(Y, Yhat, confidence, cal_indices, alpha, random=True):
    """"""
    all_indices = np.arange(len(Y))
    test_indices = np.setdiff1d(all_indices, cal_indices)

    y_calib, y_hat_calib, conf_calib = Y[cal_indices], Yhat[cal_indices], confidence[cal_indices]
    y_test, y_hat_test, conf_test = Y[test_indices], Yhat[test_indices], confidence[test_indices]
    n_calib = y_calib.shape[0]
    n_test = y_test.shape[0]

    # H0: y_hat != y
    cal0_idx = (y_hat_calib != y_calib)
    y_calib_0, y_hat_calib_0, conf_calib_0 = y_calib[cal0_idx], y_hat_calib[cal0_idx], conf_calib[cal0_idx]
    n_calib_0 = y_calib_0.shape[0]

    # Compute nonconformity score
    cal0_score = 1 - conf_calib_0
    test_score = 1 - conf_test

    # Construct conformal p-values
    if random:
        p_values = np.sum((test_score[:, None] > cal0_score), axis=-1) + np.random.rand(n_test) * (np.sum((test_score[:, None] == cal0_score), axis=-1) + 1)
        p_values /= (1 + n_calib_0)
    else:
        p_values = np.sum((test_score[:, None] >= cal0_score), axis=-1) + 1
        p_values /= (1 + n_calib_0)

    # selection procedure
    m = p_values.shape[0]
    p_values_sorted = np.sort(p_values)

    largest_i = 0
    threshold = [k * alpha * (1 + n_calib) / (m  * (1 + n_calib_0))  for k in range(1, m + 1)]
    for i in range(m - 1, -1, -1):
        if p_values_sorted[i] <= threshold[i]:
            largest_i = i
            break
    t = threshold[largest_i]

    selection_indices = (p_values <= t)

    # A sample is rejected means it is selected, certifying AI predictions as trustworthy
    y_reject, y_hat_reject = y_test[selection_indices], y_hat_test[selection_indices]

    selection_size = y_reject.shape[0]

    flr = np.sum(y_reject != y_hat_reject) / selection_size if selection_size > 0 else 0
    power = np.sum(y_reject == y_hat_reject) / np.sum(y_test == y_hat_test)

    return flr, power, selection_size



