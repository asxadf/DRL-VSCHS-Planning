import gymnasium as gym
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from gymnasium.spaces import Box, Dict
from operation_model.lib_operation_model import auto_DA_RT_single_day_with_mp, get_cr, generate_LP_file_DA
from datetime import datetime, timedelta

class OperationEnvClass(gym.Env):
    def __init__(self, DatabaseSpec, date_1st, date_end):
        self.database = DatabaseSpec
        self.mode = self.database.mode

        self.set_date_eps = pd.date_range(start=date_1st, end=date_end).strftime(self.database.tf_date).tolist()

        # Define the maximum and minimum vectors
        self.vol_max_array = np.array([self.database.vol_max['rsv_0'], self.database.vol_max['rsv_1']])
        self.vol_min_array = np.array([self.database.vol_min['rsv_0'], self.database.vol_min['rsv_1']])

        self.step_idx = 1

        if self.mode == 'EODS':
            self.action_space = Box(low=-np.ones(len(self.database.set_rsv)),
                                    high=np.ones(len(self.database.set_rsv)),
                                    shape=(len(self.database.set_rsv),),
                                    dtype=np.float32)
        elif self.mode == 'LMWV':
            self.action_space = Box(low=np.zeros(len(self.database.set_rsv)),
                                    high=300*np.ones(len(self.database.set_rsv)),
                                    shape=(len(self.database.set_rsv),),
                                    dtype=np.float32)

        # Initialize observations
        self.observation_space = Dict({'vol_ini_DA': Box(low=np.zeros(len(self.database.set_rsv)),
                                                         high=np.ones(len(self.database.set_rsv)),
                                                         shape=(len(self.database.set_rsv),),
                                                         dtype=np.float32),

                                       'pre_wi_natural_rate_avg': Box(low=np.zeros(1),
                                                                      high=np.ones(1),
                                                                      shape=(1,),
                                                                      dtype=np.float32),

                                       'pre_p_ver_avg': Box(low=np.zeros(len(self.database.set_ver)),
                                                            high=np.ones(len(self.database.set_ver)),
                                                            shape=(len(self.database.set_ver),),
                                                            dtype=np.float32),

                                       'pre_lmp_avg': Box(low=np.full(1, -1),
                                                          high=np.ones(1),
                                                          shape=(1,),
                                                          dtype=np.float32),

                                       'day_in_year': Box(low=np.zeros(1),
                                                          high=np.ones(1),
                                                          shape=(1,),
                                                          dtype=np.float32),

                                       'sys_vol_req_daily_ub': Box(low=np.zeros(1),
                                                                   high=np.ones(1),
                                                                   shape=(1,),
                                                                   dtype=np.float32),

                                       'sys_vol_req_daily_lb': Box(low=np.zeros(1),
                                                                   high=np.ones(1),
                                                                   shape=(1,),
                                                                   dtype=np.float32)}
                                      )

        # Initialize state
        self.initialize_state()

        # CR results
        self.cr_result = get_cr()

        # Initialize LP files
        print('Initializing models...')
        self.model_DA_public = generate_LP_file_DA(self.database, date_1st)
        print('Initializing is done!')

    def initialize_state(self):
        self.state_date = self.set_date_eps[0]
        # Obs: From org to normalize
        self.state_vol_ini_DA_org = (np.array(list(self.database.vol_max.values())) - np.array(list(self.database.vol_min.values()))) + np.array(list(self.database.vol_min.values()))
        self.state_pre_wi_natural_rate_avg_org = np.array([np.mean(self.database.pre_wi_natural_rate.loc[self.state_date])])
        self.state_pre_p_ver_avg_org = np.array(np.mean(self.database.pre_p_ver.loc[self.state_date], axis=0))
        self.state_pre_lmp_avg_org = np.array([np.mean(self.database.pre_lmp.loc[self.state_date])])
        self.state_day_in_year_org = np.array([datetime.strptime(self.state_date, self.database.tf_date).timetuple().tm_yday])
        self.state_day_in_eps_org = np.array([[datetime.strptime(date, self.database.tf_date)
                                               for date in self.set_date_eps].index(datetime.strptime(self.state_date, self.database.tf_date)) + 1])

        self.state_vol_ini_DA = self.normalize(self.state_vol_ini_DA_org, 'vol')
        self.state_pre_wi_natural_rate_avg = self.normalize(self.state_pre_wi_natural_rate_avg_org, 'rate')
        self.state_pre_p_ver_avg = self.normalize(self.state_pre_p_ver_avg_org, 'ver')
        self.state_pre_lmp_avg = self.normalize(self.state_pre_lmp_avg_org, 'lmp')
        self.state_day_in_year = self.normalize(self.state_day_in_year_org, 'year')
        self.state_day_in_eps = self.normalize(self.state_day_in_eps_org, 'episode')
        self.state_sys_vol_req_daily_ub = np.array(self.database.sys_vol_req_daily.loc[self.state_day_in_year_org, 'UB'])
        self.state_sys_vol_req_daily_lb = np.array(self.database.sys_vol_req_daily.loc[self.state_day_in_year_org, 'LB'])

        self.wf_in_delay_from_yesterday_DA_by_date = {date: {rsv: 0 for rsv in self.database.set_rsv}
                                                      for date in self.set_date_eps}

        self.wf_in_delay_from_lasthour_RT_by_datesubhour = {(date, subhour): {rsv: 0 for rsv in self.database.set_rsv}
                                                            for date in self.set_date_eps
                                                            for subhour in self.database.set_subhour}

    def _get_obs(self):
        return {'vol_ini_DA': self.state_vol_ini_DA,
                'pre_wi_natural_rate_avg': self.state_pre_wi_natural_rate_avg,
                'pre_p_ver_avg': self.state_pre_p_ver_avg,
                'pre_lmp_avg': self.state_pre_lmp_avg,
                'day_in_year': self.state_day_in_year,
                'sys_vol_req_daily_ub': self.state_sys_vol_req_daily_ub,
                'sys_vol_req_daily_lb': self.state_sys_vol_req_daily_lb}

    def step(self, action):
        print(f'Step #{self.step_idx}')
        self.step_idx += 1

        vol_ini_DA_orig = self.normalize_reverse(self.state_vol_ini_DA, 'vol') # in cfs-day
        vol_ini_DA = {rsv: vol_ini_DA_orig[i] for i, rsv in enumerate(self.database.set_rsv)}

        # Manually modify action
        if self.mode == 'EODS':
            rsv_at_V_max = vol_ini_DA_orig == self.vol_max_array  # Check whether a rsv is at max SoC
            rsv_at_V_min = vol_ini_DA_orig == self.vol_min_array  # Check whether a rsv is at min SoC
            action[rsv_at_V_max] = np.minimum(action[rsv_at_V_max],
                                              0)  # If rsv is at max SoC, action has to maintain or decrease SoC
            action[rsv_at_V_min] = np.maximum(action[rsv_at_V_min],
                                              0)  # If rsv is at min SoC, action has to maintain or increase SoC
            vol_end_exp_DA_orig = self.de_normalize_V_end(action, vol_ini_DA_orig)  # in cfs-day
            mid_term_planning = {rsv: vol_end_exp_DA_orig[i] for i, rsv in enumerate(self.database.set_rsv)}
        elif self.mode == 'LMWV':
            mid_term_planning = {rsv: action[i] for i, rsv in enumerate(self.database.set_rsv)}

        result_DA, result_RT_by_mp = auto_DA_RT_single_day_with_mp(self.database,
                                                                   self.state_date,
                                                                   self.model_DA_public,
                                                                   vol_ini_DA,
                                                                   mid_term_planning,
                                                                   self.wf_in_delay_from_yesterday_DA_by_date,
                                                                   self.wf_in_delay_from_lasthour_RT_by_datesubhour,
                                                                   self.cr_result)

        vol_end_act = (result_RT_by_mp['var_z_vol_end'].iloc[-1]).values

        print(f'vol_ini is {np.round(vol_ini_DA_orig, 2)}; '
              f'mid_term_planning ({self.mode}) is {np.round(np.array(list(mid_term_planning.values())), 2)}; '
              f'vol_end_act is {np.round(vol_end_act, 2)};')

        # Calculate the reward
        lmp_DA = result_DA['para_pre_lmp']
        lmp_RT = result_RT_by_mp['para_act_lmp']
        z_p_act = result_RT_by_mp['var_z_p_act']
        z_p_difference = result_RT_by_mp['var_z_p_difference']
        z_p_ver_ub = result_RT_by_mp['para_act_p_ver']
        x_p_cmit = z_p_act + z_p_difference + 0.1

        revenue_gross = (lmp_DA.iloc[:, 0] * result_DA['var_x_p_cmit'].iloc[:, 0]).sum()
        #revenue_gross = revenue_gross / ((np.multiply(lmp_DA.iloc[:, 0].values, np.sum(result_DA['para_pre_p_ver'].values,axis=1))).sum())
        revenue_gross = revenue_gross / 10000

        charge_imb = (lmp_RT.iloc[:, 0] * z_p_difference.iloc[:, 0]).sum() * (1 / 12)
        #charge_imb = charge_imb/(np.multiply(lmp_RT.values*(1 / 12), x_p_cmit.values)).sum()
        charge_imb = 10*charge_imb / 10000

        penalty_z_wf_spill = 0*(result_RT_by_mp['var_z_wf_spill'].values.sum()*(1 / 12) / (self.vol_max_array - self.vol_min_array).sum())

        reward = revenue_gross - charge_imb - penalty_z_wf_spill

        print(f'Gross revenue: {np.round(revenue_gross, 2)}; '
              f'Imbalance charge: {np.round(charge_imb, 2)}; '
              f'Spill penalty: {np.round(penalty_z_wf_spill, 2)}; '
              f'Reward is {np.round(reward, 2)}; ')
        print(f'----------------------------------------------------')

        # Check flag
        flag_is_safe = self.state_sys_vol_req_daily_lb[0] \
                       <= (sum(vol_end_act)-sum(self.vol_min_array)) / (sum(self.vol_max_array) - sum(self.vol_min_array)) \
                       <= self.state_sys_vol_req_daily_ub[0]
        flag_is_not_end_day_of_eps = self.state_date != self.set_date_eps[-1]

        if flag_is_safe and flag_is_not_end_day_of_eps:
            is_terminated = False
        else:
            is_terminated = True
            print(f'This episode has ended.')
            print(f'----------------------------------------------------')

        if is_terminated is False:
            # Move to next day
            date_last = self.state_date
            self.state_date = (datetime.strptime(self.state_date, self.database.tf_date)
                               + timedelta(days=1)).strftime(self.database.tf_date)

            # Obs: From org to normalize
            self.state_vol_ini_DA_org = (result_RT_by_mp['var_z_vol_end'].iloc[-1]).values
            self.state_pre_wi_natural_rate_avg_org = np.array([np.mean(self.database.pre_wi_natural_rate.loc[self.state_date])])
            self.state_pre_p_ver_avg_org = np.array(np.mean(self.database.pre_p_ver.loc[self.state_date], axis=0))
            self.state_pre_lmp_avg_org = np.array([np.mean(self.database.pre_lmp.loc[self.state_date])])
            self.state_day_in_year_org = np.array([datetime.strptime(self.state_date, self.database.tf_date).timetuple().tm_yday])
            self.state_day_in_eps_org = np.array([[datetime.strptime(date, self.database.tf_date)
                                                   for date in self.set_date_eps].index(datetime.strptime(self.state_date, self.database.tf_date)) + 1])

            self.state_vol_ini_DA = self.normalize(self.state_vol_ini_DA_org, 'vol')
            self.state_pre_wi_natural_rate_avg = self.normalize(self.state_pre_wi_natural_rate_avg_org, 'rate')
            self.state_pre_p_ver_avg = self.normalize(self.state_pre_p_ver_avg_org, 'ver')
            self.state_pre_lmp_avg = self.normalize(self.state_pre_lmp_avg_org, 'lmp')
            self.state_day_in_year = self.normalize(self.state_day_in_year_org, 'year')
            self.state_day_in_eps = self.normalize(self.state_day_in_eps_org, 'episode')
            self.state_sys_vol_req_daily_ub = np.array(self.database.sys_vol_req_daily.loc[self.state_day_in_year_org, 'UB'])
            self.state_sys_vol_req_daily_lb = np.array(self.database.sys_vol_req_daily.loc[self.state_day_in_year_org, 'LB'])

            self.wf_in_delay_from_yesterday_DA_by_date[self.state_date][self.database.set_rsv[1]] \
                = sum(result_RT_by_mp['var_z_wf_out'].loc[(date_last, self.database.set_subhour_last_hour[0]):
                                                          (date_last, self.database.set_subhour_last_hour[-1]), self.database.set_rsv[0]])

            for t, subhour_first_hour in enumerate(self.database.set_subhour_first_hour):
                self.wf_in_delay_from_lasthour_RT_by_datesubhour[(self.state_date, subhour_first_hour)][self.database.set_rsv[1]]\
                    = result_RT_by_mp['var_z_wf_out'].at[(date_last, self.database.set_subhour_last_hour[t]), self.database.set_rsv[0]]

        observation = self._get_obs()
        info = {}

        # Return step information
        return observation, reward, is_terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.initialize_state()
        observation = self._get_obs()
        info = {}
        return observation, info

    def normalize(self, var_org, var_type):
        if var_type == 'vol':
            var_nor = np.divide(var_org - np.array(list(self.database.vol_min.values())),
                                np.array(list(self.database.vol_max.values()))
                                - np.array(list(self.database.vol_min.values())))

        elif var_type == 'rate':
            var_nor = np.divide(var_org - self.database.wi_natural_rate_lb,
                                self.database.wi_natural_rate_ub - self.database.wi_natural_rate_lb)

        elif var_type == 'ver':
            var_nor = np.divide(var_org - np.array([self.database.wind_lb, self.database.solar_lb]),
                                np.array([self.database.wind_ub, self.database.solar_ub]) - np.array([self.database.wind_lb, self.database.solar_lb]))

        elif var_type == 'lmp':
            var_nor = np.divide(var_org - self.database.act_lmp.median().item(),
                                self.database.act_lmp.quantile(0.75).item() - self.database.act_lmp.quantile(0.25).item())

        elif var_type == 'year':
            var_nor = np.divide(var_org, 366)

        elif var_type == 'episode':
            var_nor = np.divide(var_org, len(self.set_date_eps))

        return var_nor

    def normalize_reverse(self, var_nor, var_type):
        if var_type == 'vol':
            var_org = np.multiply(np.array(list(self.database.vol_max.values())) - np.array(list(self.database.vol_min.values())),
                                  var_nor)\
                      + np.array(list(self.database.vol_min.values()))

        elif var_type == 'rate':
            var_org = np.multiply(self.database.wi_natural_rate_ub - self.database.wi_natural_rate_lb,
                                  var_nor)\
                      + self.database.wi_natural_rate_lb

        elif var_type == 'ver':
            var_org = np.multiply(np.array([self.database.wind_ub, self.database.solar_ub]) - np.array([self.database.wind_lb, self.database.solar_lb]),
                                  var_nor)\
                      + np.array([self.database.wind_lb, self.database.solar_lb])

        elif var_type == 'lmp':
            var_org = np.multiply(self.database.act_lmp.quantile(0.75).item() - self.database.act_lmp.quantile(0.25).item(),
                                  var_nor)\
                      + self.database.act_lmp.median().item()

        elif var_type == 'day':
            var_org = var_nor*len(self.set_date)

        return var_org

    def normalize_V_end(self, V_end_orig, V_ini_orig):
        V_max = self.vol_max_array
        V_min = self.vol_min_array

        # Initialize the normalized action array
        V_end_norm = np.zeros_like(V_end_orig, dtype=np.float32)

        # Normalize elements
        ele_max = V_ini_orig == V_max
        V_end_norm[ele_max] = -(V_max[ele_max] - V_end_orig[ele_max]) / (V_max[ele_max] - V_min[ele_max])

        ele_min = V_ini_orig == V_min
        V_end_norm[ele_min] =  (V_end_orig[ele_min] - V_min[ele_min]) / (V_max[ele_min] - V_min[ele_min])

        ele_geq = V_end_orig > V_ini_orig
        V_end_norm[ele_geq] =  (V_end_orig[ele_geq] - V_ini_orig[ele_geq]) / (V_max[ele_geq] - V_ini_orig[ele_geq])

        ele_leq = V_end_orig < V_ini_orig
        V_end_norm[ele_leq] = -(V_ini_orig[ele_leq] - V_end_orig[ele_leq]) / (V_ini_orig[ele_leq] - V_min[ele_leq])

        ele_eqq = V_end_orig == V_ini_orig
        V_end_norm[ele_eqq] = 0

        return V_end_norm

    def de_normalize_V_end(self, V_end_norm, V_ini_orig):
        V_max = self.vol_max_array
        V_min = self.vol_min_array

        # Initialize the original action array
        V_end_orig = np.zeros_like(V_end_norm, dtype=np.float32)

        # De-normalize elements
        ele_max = V_ini_orig == V_max
        V_end_orig[ele_max] = V_max[ele_max] + V_end_norm[ele_max] * (V_max[ele_max] - V_min[ele_max])

        ele_min = V_ini_orig == V_min
        V_end_orig[ele_min] = V_min[ele_min] + V_end_norm[ele_min] * (V_max[ele_min] - V_min[ele_min])

        ele_geq = V_end_norm > 0
        V_end_orig[ele_geq] = V_ini_orig[ele_geq] + V_end_norm[ele_geq] * (V_max[ele_geq] - V_ini_orig[ele_geq])

        ele_leq = V_end_norm < 0
        V_end_orig[ele_leq] = V_ini_orig[ele_leq] + V_end_norm[ele_leq] * (V_ini_orig[ele_leq] - V_min[ele_leq])

        ele_eqq = V_end_norm == 0
        V_end_orig[ele_eqq] = V_ini_orig[ele_eqq]

        return V_end_orig



