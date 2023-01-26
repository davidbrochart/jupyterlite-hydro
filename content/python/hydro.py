import matplotlib.pyplot as plt
import numpy as np


def plot_peq(peq):
    fig, ax1 = plt.subplots(figsize=(15, 5))

    ax1.set_ylim([peq.p.max() * 2, 0])
    ax1.plot(peq.p, color='b', alpha=0.5, label='Rainfall')
    ax1.set_ylabel('Rainfall (mm/day)', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    ax2 = ax1.twinx()
    ax2.set_ylim([0, peq.q.max() * 2])
    ax2.plot(peq.e, color='r', alpha=0.3, label='PET')
    ax2.plot(peq.q, color='g', label='Discharge')
    ax2.set_ylabel('Discharge (mm/day)', color='g')
    for tl in ax2.get_yticklabels():
        tl.set_color('g')

    ax1.legend(loc='center left')
    ax2.legend(loc='center right')
    plt.show()


def run_gr4j(x, p, e, q, s, uh1_array, uh2_array, l, m):
    for t in range(p.size):
        if p[t] > e[t]:
            pn = p[t] - e[t]
            en = 0.
            tmp = s[0] / x[0]
            ps = x[0] * (1. - tmp ** 2) * np.tanh(pn / x[0]) / (1. + tmp * np.tanh(pn / x[0]))
            s[0] += ps
        elif p[t] < e[t]:
            ps = 0.
            pn = 0.
            en = e[t] - p[t]
            tmp = s[0] / x[0]
            es = s[0] * (2. - tmp) * np.tanh(en / x[0]) / (1. + (1. - tmp) * np.tanh(en / x[0]))
            tmp = s[0] - es
            if tmp > 0.:
                s[0] = tmp
            else:
                s[0] = 0.
        else:
            pn = 0.
            en = 0.
            ps = 0.
        tmp = (4. * s[0] / (9. * x[0]))
        perc = s[0] * (1. - (1. + tmp ** 4) ** (-1. / 4.))
        s[0] -= perc
        pr_0 = perc + pn - ps
        q9 = 0.
        q1 = 0.
        for i in range(m):
            if i == 0:
                pr_i = pr_0
            else:
                pr_i = s[2 + i - 1]
            if i < l:
                q9 += uh1_array[i] * pr_i
            q1 += uh2_array[i] * pr_i
        q9 *= 0.9
        q1 *= 0.1
        f = x[1] * ((s[1] / x[2]) ** (7. / 2.))
        tmp = s[1] + q9 + f
        if tmp > 0.:
            s[1] = tmp
        else:
            s[1] = 0.
        tmp = s[1] / x[2]
        qr = s[1] * (1. - ((1. + tmp ** 4) ** (-1. / 4.)))
        s[1] -= qr
        tmp = q1 + f
        if tmp > 0.:
            qd = tmp
        else:
            qd = 0.
        q[t] = qr + qd
        if s.size > 2:
            s[3:] = s[2:-1]
            s[2] = pr_0


class GR4J:

    def __init__(self, x):
        self.x = np.array(x)
        self.s = np.empty(2 + int(2. * self.x[3]))
        self.s[0] = self.x[0] / 2.
        self.s[1] = self.x[2] / 2.
        self.s[2:] = 0.
        self.l = int(self.x[3]) + 1
        self.m = int(2. * self.x[3]) + 1
        self.uh1_array = np.empty(self.l)
        self.uh2_array = np.empty(self.m)
        for i in range(self.m):
            if i < self.l:
                self.uh1_array[i] = self.uh1(i + 1)
            self.uh2_array[i] = self.uh2(i + 1)

    def sh1(self, t):
        if t == 0:
            res = 0.
        elif t < self.x[3]:
            res = (float(t) / self.x[3]) ** (5. / 2.)
        else:
            res = 1.
        return res

    def sh2(self, t):
        if t == 0:
            res = 0.
        elif t < self.x[3]:
            res = 0.5 * ((float(t) / self.x[3]) ** (5. / 2.))
        elif t < 2. * self.x[3]:
            res = 1. - 0.5 * ((2. - float(t) / self.x[3]) ** (5. / 2.))
        else:
            res = 1.
        return res

    def uh1(self, j):
        return self.sh1(j) - self.sh1(j - 1)

    def uh2(self, j):
        return self.sh2(j) - self.sh2(j - 1)

    def run(self, pe):
        q = np.empty_like(pe[0])
        run_gr4j(self.x, pe[0], pe[1], q, self.s, self.uh1_array, self.uh2_array, self.l, self.m)
        return [q]


def calibration(x, in_obs, out_obs, warmup_period, crit_func, model, x_range, x_fix=None):
    _x = []
    for i in range(len(x_range)):
        if x_fix is None or x_fix[i] is None:
            if x[i] < x_range[i][0]:
                return np.inf
            if x[i] > x_range[i][1]:
                return np.inf
            _x.append(x[i])
        else:
            _x.append(x_fix[i])
    q_mod = model(_x)
    out_sim = q_mod.run(in_obs)
    error = crit_func(out_obs[0][warmup_period:], out_sim[0][warmup_period:])
    return error


def nse(x_obs, x_est):
    _x_obs = x_obs[~np.isnan(x_obs)]
    _x_est = x_est[~np.isnan(x_obs)]
    return 1. - (np.sum(np.square(_x_obs - _x_est)) / np.sum(np.square(_x_obs - np.mean(_x_obs))))


def nse_min(x_obs, x_est):
    return 1. - nse(x_obs, x_est)
