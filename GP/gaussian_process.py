import numpy as np
from numpy import (exp, absolute, pi, log, sqrt, mean)
from scipy.interpolate import UnivariateSpline

class GP:
    def __init__(self, x, var):
        self.x = x
        self.n = len(x)
        self.var = var
        self.hypr_L = 2
        self.tscale = 1
        self.K = np.ones((2, 2))
        self.orig_scale = x[-1]
        # print(len(x), len(var))

    def kron_del(self, x0, x1):
        return x0==x1

    def kern_fun(self, x0, x1, hypr):
        temp = self.var[np.argmin(abs(self.x-x0))]
        return (self.hypr_L**2) * exp((-np.linalg.norm(x0-x1)**2) / (2*hypr[0]**2)) + (temp)*self.kron_del(x0, x1)

    def get_kern(self, hypr):
        K = np.zeros((self.n, self.n))
        for i in range(0, self.n):
            for j in range(0, self.n):
                K[i, j] = self.kern_fun(self.x[i], self.x[j], hypr)
        # print(K)
        return K

    def log_pdf(self, hypr):
        lpdf = -.5 * (np.matmul(np.matmul(self.y.T , np.linalg.inv(self.get_kern(hypr))) , self.y) - log(np.linalg.norm(self.get_kern(hypr))) - self.n * log(2*pi)) #this has been negated so that minimize actually finds the maximum
        # print(lpdf)
        return -1 * lpdf

    def add_prior(self, x, y):
        self.x = np.append(self.x, x)
        self.n = len(self.x)
        self.opt(np.append(self.y, y))
        # print(self.x, self.y)

    def opt(self, y):
        self.y = y
        self.hypr = [10]
        # hypr0 = [1]
        # sol = minimize(self.log_pdf, hypr0, method='CG', options={'maxiter': 20})
        # # val = sol.fun
        # self.hypr = sol.x
        # self.hypr = [10, 10]
        old_K = self.K
        self.K = self.get_kern(self.hypr)
        # if np.isfinite(np.linalg.cond(self.K)):
        self.K_inv = np.linalg.inv(self.K)
            # self.K = old_K
        self.sample_values_x = np.linspace(self.x[0], self.x[-1], 100)
        self.sample_values_y = self.draw_sample(self.sample_values_x)
        # print(len(self.sample_values_y[1]))

    def get_K_star(self, x_star):
        temp = []
        for j in self.x:
            temp.append(self.kern_fun(x_star, j, self.hypr))
        self.K_star = np.array(temp)
        return np.array(temp)

        # self.K_star = np.array(self.K_star[1:len(x_star)+1])

    def get_K_starstar(self, x_star):
        self.K_starstar = self.kern_fun(x_star, x_star, self.hypr)
        return self.kern_fun(x_star, x_star, self.hypr)

    def get_hypr(self):
        return (self.hypr_L, self.hypr[0], sqrt(mean(self.var)))

    def eval_continuous(self, density):
        self.cont_sample = self.draw_sample(np.linspace(0, self.orig_scale, density))

    def draw_sample(self, x_star):
        y_bar_star = [None for _ in range(len(x_star))]
        var_y_star = [None for _ in range(len(x_star))]
        for i in range(len(x_star)):
            K_star = self.get_K_star(x_star[i])
            K_starstar = self.get_K_starstar(x_star[i])
            y_bar_star[i] = np.matmul(np.matmul(self.K_star, self.K_inv), self.y)
            var_y_star[i] = K_starstar - np.matmul(np.matmul(K_star, self.K_inv), K_star.T)
        return (y_bar_star, abs(np.array(var_y_star)))

    def calc_likelihood(self, scale, shift, x_eval, y_eval, show, mode):
        ll = 0
        # print(scale)
        if mode==1:
            x_eval_scale = (x_eval*scale) + shift
            (y_bar_star, var_y_star) = self.draw_sample(x_eval_scale)
            for i in range(len(x_eval)):
                s = (y_eval[i] - y_bar_star[i])**2
                ll_temp = min(1/sqrt(2*pi*self.hypr[0]) * exp(-s/(2*self.hypr[0])), 1)
                ll += 2 * ll_temp * exp(-(i-len(x_eval))**2 / (2*len(x_eval))) / sqrt(2*pi*len(x_eval))
        elif mode==2:
            x_eval_scale = (x_eval*scale) + shift
            (y_bar_star, var_y_star) = self.draw_sample(x_eval_scale)
            for i in range(len(x_eval)):
                s = (y_eval[i] - y_bar_star[i])**2
                ll_temp = min(1/sqrt(2*pi*self.hypr[0]) * exp(-s/(2*self.hypr[0])), 1)
                ll += 2 * ll_temp * exp(-(i)**2 / (2*len(x_eval))) / sqrt(2*pi*len(x_eval))
        elif mode==3:
            gp_scale = self.sample_values_x/scale + shift
            # print(scale, gp_scale[0], gp_scale[-1])
            for i in range(len(x_eval)):
                nearest = self.get_closest(gp_scale, self.sample_values_y[0], x_eval[i])
                if nearest[0]:
                    y_bar_star = self.sample_values_y[0][nearest[1]]
                    var_y_star = self.sample_values_y[1][nearest[1]]
                    s = (y_eval[i] - y_bar_star)**2
                    # print(x_eval[i], nearest[2])
                    # print(y_eval[i], y_bar_star)
                    ll_temp = min(1/sqrt(2*pi*self.hypr[0]) * exp(-s/(2*self.hypr[0])), 1)
                else:
                    ll_temp = 0
                ll += 2 * ll_temp * exp(-(i)**2 / (2*len(x_eval))) / sqrt(2*pi*len(x_eval))
        elif mode==4:
            gp_scale = self.sample_values_x/scale + shift
            # print(scale, gp_scale[0], gp_scale[-1])
            for i in range(len(x_eval)):
                nearest = self.get_closest(gp_scale, self.sample_values_y[0], x_eval[i])
                if nearest[0]:
                    y_bar_star = self.sample_values_y[0][nearest[1]]
                    var_y_star = self.sample_values_y[1][nearest[1]]
                    s = (y_eval[i] - y_bar_star)**2
                    # print(x_eval[i], nearest[2])
                    # print(y_eval[i], y_bar_star)
                    ll_temp = min(1/sqrt(2*pi*self.hypr[0]) * exp(-s/(2*self.hypr[0])), 1)
                else:
                    ll_temp = 0
                ll += ll_temp
        elif mode == 5:
            gp_scale = self.sample_values_x/scale + shift
            nearest = self.get_closest(gp_scale, self.sample_values_y[0], x_eval)
            var_y_star = self.sample_values_y[1][nearest[1]]
            y_bar_star = nearest[3]
            s = (y_eval - y_bar_star)**2
            ll = sum(np.clip(1/sqrt(2*pi*self.hypr[0]) * exp(-s/(2*self.hypr[0])), None, 1))
        # print(ll)
        return abs(ll)

    def get_closest(self, arr_x, arr_y, target):
        if not isinstance(target, (list, tuple, np.ndarray)):
            loc = np.argmin(abs(arr_x - target))
            return(True, np.array([loc]), np.array([arr_x[loc]]), np.array([arr_y[loc]]))
        else:
            temp_loc = []
            temp_x = []
            temp_y = []
            for i in target:
                temp_loc.append(np.argmin(abs(arr_x - i)))
                temp_x.append(arr_x[temp_loc[-1]])
                temp_y.append(arr_y[temp_loc[-1]])
            return(True, np.array(temp_loc), np.array(temp_x), np.array(temp_y))

    def plot_process(self, ax, c):
        x_plot = np.linspace(0, 1, len(self.cont_sample[0]))
        x_plot_ss = np.linspace(0, 1, 1*len(self.cont_sample[0]))
        y_plot = self.cont_sample[0]
        y_plot_upper = y_plot + np.sqrt(self.cont_sample[1])
        # s = UnivariateSpline(x_plot, y_plot_upper, None)
        # y_plot_upper = s(x_plot_ss)
        y_plot_lower = y_plot - np.sqrt(self.cont_sample[1])
        # s = UnivariateSpline(x_plot, y_plot_lower, None)
        # y_plot_lower = s(x_plot_ss)
        ax.fill_between(x_plot_ss[:-1], y_plot_upper[:-1], y_plot_lower[:-1], color = c, alpha = 0.2)
        ax.plot(x_plot[:-1], y_plot[:-1], color = c, linestyle = 'dashed')

    def plot_pt(self, t, pt, m, S):
        x = m + np.linspace(-3*sqrt(S), 3*sqrt(S), 50)
        f = exp(-((x-m)**2) / (2*S)) / sqrt(2*pi*S)
        plt.figure(num = 2)
        # plt.subplot(layout[0], layout[1], 35)
        plt.ion()
        input()
        plt.clf()
        plt.plot(x, f)
        plt.title('t: %.3f' % t)
        plt.axvline(x = pt, color = 'red')
        fig.canvas.draw()
        plt.pause(0.001)
        plt.show()
        fig.canvas.draw()
