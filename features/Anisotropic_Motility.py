"""Calculate Anisotropic Motility features"""

from scipy import stats
from utils.DrawingGraph import *
import pandas as pd


class AnisotropicMotility(object):
    def __init__(self, trajectories, time_unit):

        self.trajectories = trajectories
        self.time_unit = time_unit

        self.avg_speed_x, self.max_speed_x, self.min_speed_x, self.net_distance_x, self.progressivity_x, self.speed_distribution_x, self.displacement_distribution_x, \
        self.avg_speed_y, self.max_speed_y, self.min_speed_y, self.net_distance_y, self.progressivity_y, self.speed_distribution_y, self.displacement_distribution_y, \
        self.avg_speed_z, self.max_speed_z, self.min_speed_z, self.net_distance_z, self.progressivity_z, self.speed_distribution_z, self.displacement_distribution_z, \
        self.exy_max, self.eyz_max, self.exz_max, self.phi_max, self.exy_total, self.eyz_total, self.exz_total, self.phi_total \
            = self.get_various_distances(self.trajectories)

        self.msd_x, self.msd_y, self.msd_z, self.alpha_x, self.alpha_y, self.alpha_z = self.get_msd_alpha(self.trajectories)

        self.displ_variance_x, self.displ_cov_x, self.displ_skewness_x, self.displ_kurtosis_x, self.displ_ngaussalpha_x, \
        self.displ_variance_y, self.displ_cov_y, self.displ_skewness_y, self.displ_kurtosis_y, self.displ_ngaussalpha_y, \
        self.displ_variance_z, self.displ_cov_z, self.displ_skewness_z, self.displ_kurtosis_z, self.displ_ngaussalpha_z, = self.get_displ_distribution_props()

        self.displ_autocorr_x, self.displ_partial_autocorr_x, self.displ_autocorr_y, self.displ_partial_autocorr_y, \
        self.displ_autocorr_z, self.displ_partial_autocorr_z = self.get_displ_autocorr()

        #self.displ_hurst_RS_x, self.displ_hurst_RS_y, self.displ_hurst_RS_z = self.get_displ_hurst_mandelbrot()

    def calc_max_distance(self, traj):
        all_distance_list = []
        for t in range(1, traj.shape[0]):
            distance = traj[t:] - traj[:-t]
            all_distance_list.append(max(abs(distance)))
        return max(all_distance_list)

    def find_primary_axis(self, traj):
        max_dist, arg, tlag_max = -1, -1, -1
        for tlag in range(1, traj.shape[0]):
            dxyz = traj[tlag:] - traj[:-tlag]  # Displacement between two nearby points
            avg = np.ones((len(traj[:, 0]), 1)) * np.mean(traj, axis=0)
            xyr = traj - avg

            # determine the rotational matrix
            u, s, rotational_matrix = np.linalg.svd(dxyz)
            rotational_matrix = rotational_matrix.T

            # project major axis of trajectories onto rotational matrix
            xyr_r = xyr @ rotational_matrix
            x = xyr_r[:, 0]
            y = xyr_r[:, 1]
            z = xyr_r[:, 2]
            list_dist = [self.calc_max_distance(x), self.calc_max_distance(y), self.calc_max_distance(z)]
            dist = max(list_dist)
            arg = np.argmax(list_dist)
            # print(tlag, calc_max_distance(x), calc_max_distance(y), calc_max_distance(z))
            if dist > max_dist:
                max_dist = dist
                max_arg = arg
                tlag_max = tlag
                list_dist[max_arg] = -1
                max_arg2 = np.argmax(list_dist)
                list_dist[max_arg2] = -1
                max_arg3 = np.argmax(list_dist)

            # print(tlag, max_dist, max_arg, max_arg2, max_arg3)

        # print(tlag_max, max_dist, max_arg, max_arg2, max_arg3)

        dxyz = traj[tlag_max:] - traj[:-tlag_max]  # Displacement between two nearby points
        avg = np.ones((len(traj[:, 0]), 1)) * np.mean(traj, axis=0)
        xyr = traj - avg

        # determine the rotational matrix
        u, s, rotational_matrix = np.linalg.svd(dxyz)
        rotational_matrix = rotational_matrix.T

        # project major axis of trajectories onto rotational matrix
        rotated_traj = xyr @ rotational_matrix
        pc1 = rotated_traj[:, max_arg]
        pc2 = rotated_traj[:, max_arg2]
        pc3 = rotated_traj[:, max_arg3]

        registered_traj = np.vstack((pc1,pc2,pc3)).T

        return pc1, pc2, pc3, registered_traj

    def calc_distance(self, coor1, coor2):
        '''Euclidean distance'''
        x1 = coor1
        x2 = coor2
        distance = math.sqrt((x2 - x1) ** 2.00) # absolute value of distance

        return distance

    def plot_original_trajectories(self, directory):
        draw_3D_trajectory(directory, self.trajectories)

    def plot_rotated_trajectories(self, directory):
        registered_trajectories = {}
        for traj_idx in self.trajectories:  # u는 세포 하나의 index
            traj = self.trajectories[traj_idx]
            _, _, _, registered_traj = self.find_primary_axis(traj)
            registered_trajectories[traj_idx] = registered_traj

        draw_3D_trajectory(directory, registered_trajectories)

    def get_various_distances(self, trajectories):
        avg_speed_x = {}
        max_speed_x = {}
        min_speed_x = {}
        progressivity_x = {}
        net_distance_x = {}
        speed_distribution_x = {}
        displacement_distribution_x = {}

        avg_speed_y = {}
        max_speed_y = {}
        min_speed_y = {}
        progressivity_y = {}
        net_distance_y = {}
        speed_distribution_y = {}
        displacement_distribution_y = {}

        avg_speed_z = {}
        max_speed_z = {}
        min_speed_z = {}
        progressivity_z = {}
        net_distance_z = {}
        speed_distribution_z = {}
        displacement_distribution_z = {}

        total_distance_x = {}
        total_distance_y = {}
        total_distance_z = {}
        max_displacement_x = {}
        max_displacement_y = {}
        max_displacement_z = {}
        exy_max = {}
        eyz_max = {}
        exz_max = {}
        phi_max = {}
        exy_total = {}
        eyz_total = {}
        exz_total = {}
        phi_total = {}

        for traj_idx in trajectories:
            traj = trajectories[traj_idx]
            traj_x, traj_y, traj_z, _ = self.find_primary_axis(traj)

            avg_speed_x[traj_idx], max_speed_x[traj_idx], min_speed_x[traj_idx], net_distance_x[traj_idx], progressivity_x[traj_idx], \
            speed_distribution_x[traj_idx], displacement_distribution_x[traj_idx], total_distance_x[traj_idx], max_displacement_x[traj_idx] = self.calc_various_distances(traj_x)

            avg_speed_y[traj_idx], max_speed_y[traj_idx], min_speed_y[traj_idx], net_distance_y[traj_idx], progressivity_y[traj_idx], \
            speed_distribution_y[traj_idx], displacement_distribution_y[traj_idx], total_distance_y[traj_idx], max_displacement_y[traj_idx] = self.calc_various_distances(traj_y)

            avg_speed_z[traj_idx], max_speed_z[traj_idx], min_speed_z[traj_idx], net_distance_z[traj_idx], progressivity_z[traj_idx], \
            speed_distribution_z[traj_idx], displacement_distribution_z[traj_idx], total_distance_z[traj_idx], max_displacement_z[traj_idx] = self.calc_various_distances(traj_z)


            exy_max[traj_idx] = max_displacement_y[traj_idx] / max_displacement_x[traj_idx]  # elongation parameters
            eyz_max[traj_idx] = max_displacement_z[traj_idx] / max_displacement_y[traj_idx]  # elongation parameters
            exz_max[traj_idx] = max_displacement_z[traj_idx] / max_displacement_x[traj_idx]  # elongation parameters
            phi_max[traj_idx] = (max_displacement_z[traj_idx] ** 2 / (max_displacement_x[traj_idx] * max_displacement_y[traj_idx])) ** (1 / 3)  # sphericity index

            exy_total[traj_idx] = total_distance_y[traj_idx] / total_distance_x[traj_idx]  # elongation parameters
            eyz_total[traj_idx] = total_distance_z[traj_idx] / total_distance_y[traj_idx]  # elongation parameters
            exz_total[traj_idx] = total_distance_z[traj_idx] / total_distance_x[traj_idx]  # elongation parameters
            phi_total[traj_idx] = (total_distance_z[traj_idx] ** 2 / (total_distance_x[traj_idx] * total_distance_y[traj_idx])) ** (1 / 3)  # sphericity index



        return avg_speed_x, max_speed_x, min_speed_x, net_distance_x, progressivity_x, speed_distribution_x, displacement_distribution_x, \
               avg_speed_y, max_speed_y, min_speed_y, net_distance_y, progressivity_y, speed_distribution_y, displacement_distribution_y, \
               avg_speed_z, max_speed_z, min_speed_z, net_distance_z, progressivity_z, speed_distribution_z, displacement_distribution_z, \
               exy_max, eyz_max, exz_max, phi_max, exy_total, eyz_total, exz_total, phi_total

    def calc_various_distances(self, traj):
        start_point = traj[0]  # start_point = (x0)
        end_point = traj[-1]

        distance_start_to_end = self.calc_distance(start_point, end_point)
        net_distance = distance_start_to_end
        distance_list = []
        for coor in traj[1:]:  # coor = (x1) (x2), ...
            distance = self.calc_distance(start_point, coor)  # distance x0 ~ x1, x1~x2, ... xn-1 ~ xn
            distance_list.append(distance)
            start_point = coor  # start_point change to (x1)
        all_distance_list = []
        for t in range(1, traj.shape[0]):
            distance = traj[t:] - traj[:-t]
            all_distance_list.append( max( abs(distance) ) )

        max_displacement = max(all_distance_list)
        total_distance = sum(distance_list)
        avg_speed = np.mean(distance_list) / self.time_unit
        max_speed = max(distance_list) / self.time_unit
        min_speed = min(distance_list) / self.time_unit
        speed_distribution = [distance / self.time_unit for distance in distance_list]
        displacement_distribution = [distance for distance in distance_list]
        progressivity = float(net_distance) / float(total_distance)

        return avg_speed, max_speed, min_speed, net_distance, progressivity, speed_distribution, displacement_distribution, total_distance, max_displacement

    def get_msd_alpha(self, trajectories):
        '''
        Calculates mean squared displacement for a cell path
        for a given time lag tau
        Parameters
        ----------
        path : list
            tuples of sequential XY coordinates.
        tau  : int
            time lag to consider when calculating MSD.
        Returns
        -------
        msd : float
            mean squared displacement of path given time lag tau.
        alpha: float

        '''
        msds_x = {}
        msds_y = {}
        msds_z = {}
        alphas_x = {}
        alphas_y = {}
        alphas_z = {}

        for traj_idx in trajectories:
            traj = trajectories[traj_idx]
            traj_x, traj_y, traj_z, _ = self.find_primary_axis(traj)
            msds_x[traj_idx], alphas_x[traj_idx] = self.calc_msd_alpha(traj_x)
            msds_y[traj_idx], alphas_y[traj_idx] = self.calc_msd_alpha(traj_y)
            msds_z[traj_idx], alphas_z[traj_idx] = self.calc_msd_alpha(traj_z)

        return msds_x, msds_y, msds_z, alphas_x, alphas_y, alphas_z

    def calc_msd_alpha(self, traj):
        max_tau = traj.shape[0] - 1
        traj_msds = {}
        tau = 1
        while tau <= max_tau:
            distance_sqs = []
            t = 0
            while (t + tau) < len(traj):
                distance_sq = (self.calc_distance(traj[t + tau], traj[t])) ** 2
                distance_sqs.append(distance_sq)
                t += 1

            msd = sum(distance_sqs) / len(distance_sqs)  # msd = msd for each tau
            traj_msds[tau] = msd  # traj_msds = msds for each trajectory (each has msd[1], msd[2], ... msd[T])
            tau += 1

        log_tau = [np.log(tau) for tau in traj_msds.keys()]
        log_msd = [np.log(msd) for msd in traj_msds.values()]
        slope, intercept, r_val, p_val, SE = scipy.stats.linregress(log_tau, log_msd)
        return traj_msds, slope

    def plot_msd_alpha(self, directory):
        for traj_idx in self.msd_x:
            traj_msds_x = self.msd_x[traj_idx]
            traj_msds_y = self.msd_y[traj_idx]
            traj_msds_z = self.msd_z[traj_idx]

            self.draw_msd_alpha_each(directory, traj_msds_x, traj_idx, axis='x')
            self.draw_msd_alpha_each(directory, traj_msds_y, traj_idx, axis='y')
            self.draw_msd_alpha_each(directory, traj_msds_z, traj_idx, axis='z')

    def draw_msd_alpha_each(self, directory, traj_msds, traj_idx, axis='x'):
        log_tau = [np.log(tau) for tau in traj_msds.keys()]
        log_msd = [np.log(msd) for msd in traj_msds.values()]
        slope, intercept, r_val, p_val, SE = scipy.stats.linregress(log_tau, log_msd)

        plt.figure(figsize=(15, 10))
        plt.plot(traj_msds.keys(), traj_msds.values())
        plt.title('Mean Square Displacement', fontsize=20)
        plt.xlabel('Tau', fontsize=15)
        plt.ylabel('MSD_%s' % axis, fontsize=15)
        if not os.path.isdir(directory + 'msd_%s/' % axis):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(directory + 'msd_%s/' % axis)
        plt.savefig(directory + 'msd_%s/%s' % (axis, traj_idx))
        plt.clf()
        plt.close()

        plt.figure(figsize=(15, 10))
        plt.plot(log_tau, log_msd)
        plt.plot(log_tau, [slope * tau for tau in log_tau] + intercept, color='red')
        plt.title('alpha: %s' % slope, fontsize=20)
        plt.xlabel('log(Tau)', fontsize=15)
        plt.ylabel('log(MSD_%s)' % axis, fontsize=15)
        if not os.path.isdir(directory + 'msd_%s/' % axis):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
            os.makedirs(directory + 'msd_%s/' % axis)
        plt.savefig(directory + 'msd_%s/%s_log' % (axis, traj_idx))
        plt.clf()
        plt.close()

    def get_displ_distribution_props(self):
        '''
        Calculates displacement distribution properties (variance, skewness, kurtosis, non gaussian parameter
        for each cell in trajectories
        Returns
        -------
        variance : dict keyed by cell_id with variance of displacement distribution
        skewness : dict keyed by trajectories with skewness of displacement distribution
        kurtosis : dict keyed by trajectories with kurtosis of displacement distribution
        ngaussalpha : dict keyed by trajectories with non gaussian parameter of displacement distribution
        '''
        displ_variance_x = {}
        displ_cov_x = {} # coefficient of variation
        displ_skewness_x = {}
        displ_kurtosis_x = {}
        displ_ngaussalpha_x = {}

        displ_variance_y = {}
        displ_cov_y = {}  # coefficient of variation
        displ_skewness_y = {}
        displ_kurtosis_y = {}
        displ_ngaussalpha_y = {}

        displ_variance_z = {}
        displ_cov_z = {}  # coefficient of variation
        displ_skewness_z = {}
        displ_kurtosis_z = {}
        displ_ngaussalpha_z = {}

        for traj_idx in self.displacement_distribution_x:  # u 는 하나의 세포 index
            displ_variance_x[traj_idx], displ_cov_x[traj_idx], displ_skewness_x[traj_idx], \
            displ_kurtosis_x[traj_idx], displ_ngaussalpha_x[traj_idx] = self.calc_displ_distribution_props(self.displacement_distribution_x, traj_idx)

            displ_variance_y[traj_idx], displ_cov_y[traj_idx], displ_skewness_y[traj_idx], \
            displ_kurtosis_y[traj_idx], displ_ngaussalpha_y[traj_idx] = self.calc_displ_distribution_props(
                self.displacement_distribution_y, traj_idx)

            displ_variance_z[traj_idx], displ_cov_z[traj_idx], displ_skewness_z[traj_idx], \
            displ_kurtosis_z[traj_idx], displ_ngaussalpha_z[traj_idx] = self.calc_displ_distribution_props(
                self.displacement_distribution_z, traj_idx)


        return displ_variance_x, displ_cov_x, displ_skewness_x, displ_kurtosis_x, displ_ngaussalpha_x, \
               displ_variance_y, displ_cov_y, displ_skewness_y, displ_kurtosis_y, displ_ngaussalpha_y, \
               displ_variance_z, displ_cov_z, displ_skewness_z, displ_kurtosis_z, displ_ngaussalpha_z

    def calc_displ_distribution_props(self, displacement_distribution, traj_idx):
        traj_displacement = np.array(displacement_distribution[traj_idx])
        displ_variance = np.var(traj_displacement)
        displ_cov = np.std(traj_displacement)/np.mean(traj_displacement)
        displ_skewness = scipy.stats.skew(traj_displacement)
        displ_kurtosis = scipy.stats.kurtosis(traj_displacement)
        displ_ngaussalpha = np.mean(traj_displacement ** 4) / (3 * np.mean(traj_displacement ** 2) ** 2) - 1

        return displ_variance, displ_cov, displ_skewness, displ_kurtosis, displ_ngaussalpha

    def get_displ_autocorr(self):
        '''
        Estimates the autocorrelation coefficient for each series of cell
        displacements over a range of time lags.
        Parameters
        ----------
        trajectories : dict of lists keyed by cell_id
        ea. list represents a cell. lists contain sequential tuples
        containing XY coordinates of a cell at a given timepoint
        Returns
        -------
        autocorr : dict of lists, containing autocorrelation coeffs for
        sequential time lags
        qstats : dict of lists containing Q-Statistics (Ljung-Box)
        pvals : dict of lists containing p-vals, as calculated from Q-Statistics
        Notes
        -----
        Estimation method:
        https://en.wikipedia.org/wiki/Autocorrelation#Estimation
        R(tau) = 1/(n-tau)*sigma**2 [sum(X_t - mu)*(X_t+tau - mu)] | t = [1,n-tau]
        X as a time series, mu as the mean of X, sigma**2 as variance of X
        tau as a given time lag (sometimes referred to as k in literature)
        Implementation uses statsmodels.tsa.stattools.acf()
        n.b. truncated to taus [1,10], to expand to more time lags, simply
        alter the indexing being loaded into the return dicts
        '''
        displ_autocorr_x = {}
        displ_partial_autocorr_x = {}
        displ_autocorr_y = {}
        displ_partial_autocorr_y = {}
        displ_autocorr_z = {}
        displ_partial_autocorr_z = {}

        for traj_idx in self.displacement_distribution_x:
            displ_autocorr_x[traj_idx], displ_partial_autocorr_x[traj_idx] = self.calc_displ_autocorr(self.displacement_distribution_x, traj_idx)
            displ_autocorr_y[traj_idx], displ_partial_autocorr_y[traj_idx] = self.calc_displ_autocorr(self.displacement_distribution_y, traj_idx)
            displ_autocorr_z[traj_idx], displ_partial_autocorr_z[traj_idx] = self.calc_displ_autocorr(self.displacement_distribution_z, traj_idx)

        return displ_autocorr_x, displ_partial_autocorr_x, displ_autocorr_y, displ_partial_autocorr_y, displ_autocorr_z, displ_partial_autocorr_z

    def calc_displ_autocorr(self, displacement_distribution, traj_idx):
        from statsmodels.tsa.stattools import acf, pacf
        traj_displacement = np.array(displacement_distribution[traj_idx])
        # Perform Ljung-Box Q-statistic calculation to determine if autocorrelations detected are significant or random
        ac = acf(traj_displacement)  # time lag 0 ~ max_tau 까지 autocorrelation 계산
        pac = pacf(traj_displacement)  # time lag 0 ~ max_tau 까지 partial autocorrelation 계산
        traj_autocorr = {}
        traj_partial_autocorr = {}
        for i in range(1, ac.shape[0]):  # always autocorr[0] = 1
            traj_autocorr[i] = ac[i]
        for i in range(2, pac.shape[0]):  # always autocorr[1] = partial_autocorr[1]
            traj_partial_autocorr[i] = pac[i]

        return traj_autocorr, traj_partial_autocorr

    def calc_largest_pow2(self, num):
        '''Finds argmax_n 2**n < `num`'''
        for i in range(0, 10):
            if int(num / 2 ** i) > 1:  # num (number of time frames = 49) > 2^i 이면 다음 i 시도
                continue
            else:
                return i - 1  # num < 2^i 이 처음 되는 순간 (i = 6) i-1 (5)을 반환

    def calc_rescaled_range(self, X, n):
        '''
        Finds rescaled range <R(n)/S(n)> for all sub-series of size `n` in `X`
        '''
        N = len(X)  # X = time series displacement data of one cell, len(X) = number of time frames - 1 = 48
        if n > N:
            return None
        # Create subseries of size n
        num_subseries = int(N / n)
        Xs = np.zeros((num_subseries, n))
        for i in range(0, num_subseries):
            Xs[i,] = X[int(i * n): int(n + (i * n))]

        # Calculate mean rescaled range R/S
        # for subseries size n
        RS = []
        for subX in Xs:

            m = np.mean(subX)
            Y = subX - m
            Z = np.cumsum(Y)
            R = max(Z) - min(Z)
            S = np.std(subX)
            if S <= 0:
                print("S = ", S)
                continue
            RS.append(R / S)
        RSavg = np.mean(RS)

        return RSavg

    def get_displ_hurst_mandelbrot(self):
        '''
        Calculates the Hurst coefficient `H` of displacement time series for each cell in `trajectories`.
        Notes
        -----
        for E[R(n)/S(n)] = Cn**H as n --> inf
        H : 0.5 - 1 ; long-term positive autocorrelation
        H : 0.5 ; fractal Brownian motion
        H : 0-0.5 ; long-term negative autocorrelation
        N.B. SEQUENCES MUST BE >= 18 units long, otherwise
        linear regression for log(R/S) vs log(n) will have
        < 3 points and cannot be performed
        '''

        displ_hurst_RS_x = {}
        displ_hurst_RS_y = {}
        displ_hurst_RS_z = {}

        for traj_idx in self.displacement_distribution_x:
            traj_displacement_x = np.array(self.displacement_distribution_x[traj_idx])
            traj_displacement_y = np.array(self.displacement_distribution_y[traj_idx])
            traj_displacement_z = np.array(self.displacement_distribution_z[traj_idx])

            RSl_x = []
            ns_x = []
            RSl_y = []
            ns_y = []
            RSl_z = []
            ns_z = []

            for i in range(0, self.calc_largest_pow2(len(traj_displacement_x))):  # range(0,5)
                ns_x.append(int(len(traj_displacement_x) / 2 ** i))
            for n in ns_x:
                RSl_x.append(self.calc_rescaled_range(traj_displacement_x, n))
            m, b, r, pval, sderr = scipy.stats.linregress(np.log(ns_x), np.log(RSl_x))
            displ_hurst_RS_x[traj_idx] = m

            for i in range(0, self.calc_largest_pow2(len(traj_displacement_y))):  # range(0,5)
                ns_y.append(int(len(traj_displacement_y) / 2 ** i))
            for n in ns_y:
                RSl_y.append(self.calc_rescaled_range(traj_displacement_y, n))
            m, b, r, pval, sderr = scipy.stats.linregress(np.log(ns_y), np.log(RSl_y))
            displ_hurst_RS_y[traj_idx] = m

            for i in range(0, self.calc_largest_pow2(len(traj_displacement_z))):  # range(0,5)
                ns_z.append(int(len(traj_displacement_z) / 2 ** i))
            for n in ns_z:
                RSl_z.append(self.calc_rescaled_range(traj_displacement_z, n))
            m, b, r, pval, sderr = scipy.stats.linregress(np.log(ns_z), np.log(RSl_z))
            displ_hurst_RS_z[traj_idx] = m

        return displ_hurst_RS_x, displ_hurst_RS_y, displ_hurst_RS_z

    def extract_features(self, feature_list, tau_limit):
        motility_data = pd.DataFrame()
        for feature_name in feature_list:
            feature_dict = getattr(self, feature_name)
            if any(txt in feature_name for txt in ('msd', 'autocorr')):
                for tau in feature_dict[0].keys():
                    if tau <= tau_limit:
                        temp = {keys: feature_dict[keys][tau] for keys in feature_dict}
                        df_temp = pd.DataFrame(temp.values(), columns=[feature_name + '_%s' % str(tau)])
                        motility_data = pd.concat([motility_data, df_temp], axis=1)
            else:
                df_temp = pd.DataFrame(feature_dict.values(), columns=[feature_name])
                motility_data = pd.concat([motility_data, df_temp], axis=1)

        dist_list = ['speed_distribution_x', 'speed_distribution_y', 'speed_distribution_z']
        dict_temp = {}
        for feature_name in dist_list:
            feature_dict = getattr(self, feature_name)
            temp = []
            for idx in feature_dict:
                distribution = feature_dict[idx]
                distribution = np.array(distribution)
                temp.append(distribution)
            dict_temp[feature_name] = temp
        df_distribution = pd.DataFrame(dict_temp, columns=dist_list)

        df_motility = pd.concat([motility_data, df_distribution], axis=1)

        return df_motility




