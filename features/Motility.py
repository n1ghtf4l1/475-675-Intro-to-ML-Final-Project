"""Extract Basic Motility features"""

from scipy import stats
from utils.DrawingGraph import *

class BasicMotility(object):
    def __init__(self, trajectories, time_unit, feature_list):

        self.trajectories = trajectories
        self.time_unit = time_unit
        self.feature_list = feature_list

    def calc_distance(self, coor1, coor2):
        '''Euclidean distance'''
        if coor1.shape[0] == 2:
            x1, y1 = coor1
            x2, y2 = coor2
            distance = math.sqrt((x2 - x1) ** 2.00 + (y2 - y1) ** 2.00)

        elif coor1.shape[0] == 3:
            x1, y1, z1 = coor1
            x2, y2, z2 = coor2
            distance = math.sqrt((x2 - x1) ** 2.00 + (y2 - y1) ** 2.00 + (z2 - z1) ** 2.00)

        return distance

    def calc_angle(self, start, middle, end):  # Start = [x0, y0, z0], middle = [x1, y1, z1], end = [x2, y2, z2],
        '''Turning angle in radians, range from 0 to pi'''

        ba = middle - start
        bc = end - middle

        if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:  # cell didn't move
            angle = 0

        else:
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            if cosine_angle > 1:  # because of limited number of digits, sometimes cosine_angle = 1.00000001
                cosine_angle = 1
            elif cosine_angle < -1:  # because of limited number of digits, sometimes cosine_angle = -1.00000001
                cosine_angle = -1
            angle = np.arccos(cosine_angle)

        return angle

    def calc_gini(self, timeseries):
        ''' Calculate Gini coefficient, where 0 is perfect equality, 1 is perfect inequality
        Parameters:
        ----------
        timeseries: np.array()
            1D array with all elements positive
        Returns:
        -------
        Gini coefficient: float
            Gini coefficient
        '''
        total = 0
        for i, xi in enumerate(timeseries[:-1], 1):
            total += np.sum(np.abs(xi - timeseries[i:]))
        return total / (len(timeseries) ** 2 * np.mean(timeseries))

    def plot_3D_trajectories(self, directory):
        draw_3D_trajectory(directory, self.trajectories)

    def get_various_distances(self):
        avg_speed = {}
        max_speed = {}
        min_speed = {}
        progressivity = {}
        total_distance = {}
        net_distance = {}
        self.speed_distribution = {}
        self.displacement_distribution = {}
        for traj_idx in self.trajectories:  # u는 세포 하나의 index
            traj = self.trajectories[traj_idx]  # cell은 t=0 ~ t=T 까지
            start_point = traj[0]  # start_point = (x0, y0)
            end_point = traj[-1]
            distance_start_to_end = self.calc_distance(start_point, end_point)
            net_distance[traj_idx] = distance_start_to_end
            distance_list = []
            for coor in traj[1:]:  # coor = (x1, y1), (x2, y2), ...
                distance = self.calc_distance(start_point, coor)  # distance x0 ~ x1, x1~x2, ... xn-1 ~ xn
                distance_list.append(distance)
                start_point = coor  # start_point change to (x1, y1)

            total_distance[traj_idx] = sum(distance_list)
            avg_speed[traj_idx] = np.mean(distance_list) / self.time_unit
            max_speed[traj_idx] = max(distance_list) / self.time_unit
            min_speed[traj_idx] = min(distance_list) / self.time_unit
            self.speed_distribution[traj_idx] = [distance / self.time_unit for distance in distance_list]
            self.displacement_distribution[traj_idx] = [distance for distance in distance_list]
            progressivity[traj_idx] = float(net_distance[traj_idx]) / float(total_distance[traj_idx])

        result = self._get_requested_features(locals())

        return result

    def get_various_angles(self):

        self.angle_distribution = {}
        avg_angle= {}
        max_angle = {}
        min_angle = {}
        total_angle = {}

        for traj_idx in self.trajectories:  # u는 세포 하나의 index
            traj = self.trajectories[traj_idx]  # cell은 t=0 ~ t=T 까지

            start = traj[0]
            middle = traj[1]
            angle_list = []
            for coor in traj[2:]:
                angle = self.calc_angle(start, middle, coor)
                angle_list.append(angle)
                start = middle
                middle = coor

            total_angle[traj_idx] = sum(angle_list)
            avg_angle[traj_idx] = np.mean(angle_list)
            max_angle[traj_idx] = max(angle_list)
            min_angle[traj_idx] = min(angle_list)
            self.angle_distribution[traj_idx] = [angle for angle in angle_list]

        result = self._get_requested_features(locals())
        return result

    def get_msd_alpha(self):
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
        msds = {}
        alphas = {}
        for traj_idx in self.trajectories:
            traj = self.trajectories[traj_idx]
            max_tau = traj.shape[0] - 1
            traj_msds = {}
            tau = 1
            while tau <= max_tau:
                distance_sqs = []
                t = 0
                while (t + tau) < len(traj):
                    distance_sq = ( self.calc_distance(traj[ t + tau ], traj[ t ]) )**2
                    distance_sqs.append(distance_sq)
                    t += 1

                msd = sum(distance_sqs)/len(distance_sqs) # msd = msd for each tau
                if msd == 0:  # if msd = 0, log(msd) -> inf
                    tau += 1
                    continue
                traj_msds[tau] = msd # traj_msds = msds for each trajectory (each has has msd[1], msd[2], ... msd[T])
                tau += 1
            msds[traj_idx] = traj_msds

            log_tau = [np.log(tau) for tau in traj_msds.keys()]
            log_msd = [np.log(msd) for msd in traj_msds.values()]
            slope, intercept, r_val, p_val, SE = scipy.stats.linregress(log_tau, log_msd)
            alphas[traj_idx] = slope

        result = self._get_requested_features(locals())

        return result

    def plot_msd_alpha(self, directory):
        for traj_idx in self.msd:
            traj_msds = self.msd[traj_idx]
            log_tau = [np.log(tau) for tau in traj_msds.keys()] # log with base e
            log_msd = [np.log(msd) for msd in traj_msds.values()] # log with base e
            slope, intercept, r_val, p_val, SE = scipy.stats.linregress(log_tau, log_msd)

            plt.figure(figsize=(15, 10))
            plt.plot(traj_msds.keys(), traj_msds.values())
            plt.title('Mean Square Displacement', fontsize=20)
            plt.xlabel('Tau', fontsize=15)
            plt.ylabel('MSD', fontsize=15)
            if not os.path.isdir(directory + 'msd/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
                os.makedirs(directory + 'msd/')
            plt.savefig(directory + 'msd/%s' % traj_idx)
            plt.clf()
            plt.close()

            plt.figure(figsize=(15, 10))
            plt.plot(log_tau, log_msd)
            plt.plot(log_tau, [slope * tau for tau in log_tau] + intercept, color='red')
            plt.title('alpha: %s' % slope, fontsize=20)
            plt.xlabel('log(Tau)', fontsize=15)
            plt.ylabel('log(MSD)', fontsize=15)
            if not os.path.isdir(directory + 'msd/'):  # Returns Boolean (if UMAP_fig folder doesn't exist, False)
                os.makedirs(directory + 'msd/')
            plt.savefig(directory + 'msd/%s_log' % traj_idx)
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
        displ_variance = {}
        displ_cov = {}  # coefficient of variation
        displ_skewness = {}
        displ_kurtosis = {}
        displ_ngaussalpha = {}
        displ_gini = {}

        for traj_idx in self.displacement_distribution:  # u 는 하나의 세포 index
            traj_displacement = np.array(self.displacement_distribution[traj_idx])
            displ_variance[traj_idx] = np.var(traj_displacement)
            displ_cov[traj_idx] = np.std(traj_displacement)/np.mean(traj_displacement)
            displ_skewness[traj_idx] = scipy.stats.skew(traj_displacement)
            displ_kurtosis[traj_idx] = scipy.stats.kurtosis(traj_displacement)
            displ_ngaussalpha[traj_idx] = np.mean(traj_displacement ** 4) / (3 * np.mean(traj_displacement ** 2) ** 2) - 1
            displ_gini[traj_idx] = self.calc_gini(traj_displacement)

        result = self._get_requested_features(locals())

        return result

    def get_angle_distribution_props(self):
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
        angle_variance = {}
        angle_cov = {} # coefficient of variation
        angle_skewness = {}
        angle_kurtosis = {}
        angle_ngaussalpha = {}
        angle_gini = {}

        for traj_idx in self.angle_distribution:  # u 는 하나의 세포 index
            traj_angle = np.array(self.angle_distribution[traj_idx])
            angle_variance[traj_idx] = np.var(traj_angle)
            angle_cov[traj_idx] = np.std(traj_angle) / np.mean(traj_angle)
            angle_skewness[traj_idx] = scipy.stats.skew(traj_angle)
            angle_kurtosis[traj_idx] = scipy.stats.kurtosis(traj_angle)
            angle_ngaussalpha[traj_idx] = np.mean(traj_angle ** 4) / (3 * np.mean(traj_angle ** 2) ** 2) - 1
            angle_gini[traj_idx] = self.calc_gini(traj_angle)

        result = self._get_requested_features(locals())
        return result

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
        from statsmodels.tsa.stattools import acf, pacf
        displ_autocorr = {}
        displ_partial_autocorr = {}
        for traj_idx in self.displacement_distribution:
            traj_displacement = np.array(self.displacement_distribution[traj_idx])
            # Perform Ljung-Box Q-statistic calculation to determine if autocorrelations detected are significant or random
            ac = acf(traj_displacement)  # time lag 0 ~ max_tau 까지 autocorrelation 계산
            pac = pacf(traj_displacement)  # time lag 0 ~ max_tau 까지 partial autocorrelation 계산
            traj_autocorr = {}
            traj_partial_autocorr = {}
            for i in range(1, ac.shape[0]): # always autocorr[0] = 1
                traj_autocorr[i] = ac[i]
            for i in range(2, pac.shape[0]): # always autocorr[1] = partial_autocorr[1]
                traj_partial_autocorr[i] = pac[i]
            displ_autocorr[traj_idx] = traj_autocorr
            displ_partial_autocorr[traj_idx] = traj_partial_autocorr

        result = self._get_requested_features(locals())

        return result

    def get_angle_autocorr(self):
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
        from statsmodels.tsa.stattools import acf, pacf
        angle_autocorr = {}
        angle_partial_autocorr = {}
        for traj_idx in self.angle_distribution:
            traj_angle = np.array(self.angle_distribution[traj_idx])
            # Perform Ljung-Box Q-statistic calculation to determine if autocorrelations detected are significant or random
            ac = acf(traj_angle)  # time lag 0 ~ max_tau 까지 autocorrelation 계산
            pac = pacf(traj_angle)  # time lag 0 ~ max_tau 까지 partial autocorrelation 계산
            traj_autocorr = {}
            traj_partial_autocorr = {}
            for i in range(1, ac.shape[0]): # always autocorr[0] = 1
                traj_autocorr[i] = ac[i]
            for i in range(2, pac.shape[0]): # always autocorr[1] = partial_autocorr[1]
                traj_partial_autocorr[i] = pac[i]
            angle_autocorr[traj_idx] = traj_autocorr
            angle_partial_autocorr[traj_idx] = traj_partial_autocorr

        result = self._get_requested_features(locals())
        return result

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

        displ_hurst_RS = {}
        for traj_idx in self.displacement_distribution:
            traj_displacement = np.array(self.displacement_distribution[traj_idx])
            RSl = []
            ns = []
            for i in range(0, self.calc_largest_pow2(len(traj_displacement))):  # range(0,5)
                ns.append(int(len(traj_displacement) / 2 ** i))
            for n in ns:
                RSl.append(self.calc_rescaled_range(traj_displacement, n))
            m, b, r, pval, sderr = scipy.stats.linregress(np.log(ns), np.log(RSl))
            displ_hurst_RS[traj_idx] = m

        result = self._get_requested_features(locals())
        return result

    def get_angle_hurst_mandelbrot(self):
        '''
        Calculates the Hurst coefficient `H` of angle time series for each cell in `trajectories`.
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

        angle_hurst_RS = {}
        for traj_idx in self.angle_distribution:
            traj_angle = np.array(self.angle_distribution[traj_idx])
            RSl = []
            ns = []
            for i in range(0, self.calc_largest_pow2(len(traj_angle))):  # range(0,5)
                ns.append(int(len(traj_angle) / 2 ** i))
            for n in ns:
                RSl.append(self.calc_rescaled_range(traj_angle, n))
            m, b, r, pval, sderr = scipy.stats.linregress(np.log(ns), np.log(RSl))
            angle_hurst_RS[traj_idx] = m

        result = self._get_requested_features(locals())
        return result

    def _get_requested_features(self, local_vars):
        local_vars_dict_only = {key: local_vars[key] for key in local_vars if type(local_vars[key]) == dict}
        # Pick only local variables that are dict type
        result = {feature: local_vars_dict_only[feature] for feature in self.feature_list if feature in local_vars_dict_only}
        #local_vars = locals(), which is {'variable': values, 'variable2':values2 ...} where variables are local within functions
        return result

    def extract_features(self, tau_limit):

        result = self.get_various_distances()
        result2 = self.get_various_angles()
        result3 = self.get_msd_alpha()

        result4 = self.get_displ_distribution_props()
        result5 =  self.get_angle_distribution_props()

        result6 = self.get_displ_autocorr()
        result7 = self.get_angle_autocorr()
        result8 = self.get_displ_hurst_mandelbrot()
        result9 = self.get_angle_hurst_mandelbrot()

        tau_dataset = {**result3, **result6, **result7}
        features = tau_dataset.keys()
        df_tau = pd.DataFrame()
        for feature in features:
            if ('msd' in feature) or ('autocorr' in feature):
                for tau in tau_dataset[feature][0].keys():
                    if tau <= tau_limit:
                        temp = {traj_idx: tau_dataset[feature][traj_idx][tau] for traj_idx in tau_dataset[feature]}
                        # tau_dataset['msds'] = {traj_idx: {0: x0, 1: x1, ... tau: x_tau} }
                        df_temp = pd.DataFrame(temp.values(), columns=[feature + '_%s' % str(tau)])
                        df_tau = pd.concat([df_tau, df_temp], axis=1)
            else:
                df_temp = pd.DataFrame(tau_dataset[feature].values(), columns=[feature])
                df_tau = pd.concat([df_tau, df_temp], axis=1)


        merged = {**result, **result2,  **result4, **result5, **result8, **result9}
        df_motility = pd.concat([pd.DataFrame(merged), df_tau], axis=1)


        return df_motility