import pandas as pd
import numpy as np
import os

class DatabaseClass:
    def __init__(self):
        self.mode = 'EODS'
        #self.mode = 'LMWV'


        self.root_path = os.path.dirname(__file__) + os.path.sep

        self.ver_cap = 1500

        if self.ver_cap == 750:
            self.wind_scaler = 0.25
            self.wind_ub = 250
            self.wind_lb = 0
            self.solar_scaler = 0.25
            self.solar_ub = 500
            self.solar_lb = 0

        if self.ver_cap == 1125:
            self.wind_scaler = 0.375
            self.wind_ub = 375
            self.wind_lb = 0
            self.solar_scaler = 0.375
            self.solar_ub = 750
            self.solar_lb = 0

        if self.ver_cap == 1500:
            self.wind_scaler = 0.5
            self.wind_ub = 500
            self.wind_lb = 0
            self.solar_scaler = 0.5
            self.solar_ub = 1000
            self.solar_lb = 0

        self.num_scen = 5

        self.num_rsv = 2
        self.num_seg = 1
        self.num_order = 2
        self.minute_per_hour = 60
        self.minute_per_subhour = 5
        self.penalty_slack = 500
        self.penalty_diff = 1000
        self.water_value = 10
        self.tf_date = '%Y-%m-%d'
        self.tf_hour = '%H:00:00'
        self.tf_subhour = '%H:%M:%S'
        self.rate_to_vol_DA = round((self.minute_per_hour * 60) / 86400, 4)
        self.rate_to_vol_RT = round((self.minute_per_subhour * 60) / 86400, 4)
        self.set_hour = pd.date_range(start="00:00:00", end="23:00:00", freq='1h').strftime(self.tf_hour).tolist()
        self.set_subhour = pd.date_range(start="00:00:00", end="23:55:00", freq='5min').strftime(self.tf_subhour).tolist()
        self.set_subhour_first_hour = pd.date_range(start="00:00:00", end="00:55:00", freq='5min').strftime(self.tf_subhour).tolist()
        self.set_subhour_last_hour = pd.date_range(start="23:00:00", end="23:55:00", freq='5min').strftime(self.tf_subhour).tolist()

        self.set_rsv = [f'rsv_{i_rsv}'
                        for i_rsv in range(self.num_rsv)]
        self.set_ver = ['wind', 'solar']
        self.num_ver = int(len(self.set_ver))
        self.set_scen = [f'Scenario_{str(scen)}' for scen in range(self.num_scen)]

        # Read system data
        self.rate_to_power = {rsv: pd.read_csv(f'{self.root_path}rate_to_power_{rsv}.csv')
                              for rsv in self.set_rsv}
        self.power_range = {rsv: pd.read_csv(f'{self.root_path}unit_power_range_{rsv}.csv', index_col=0)
                            for rsv in self.set_rsv}
        self.vol_range = {rsv: pd.read_csv(f'{self.root_path}forebay_level_to_volume_{rsv}.csv', dtype=float)
                          for rsv in self.set_rsv}

        self.set_seg = [f'seg_{i_seg}'
                        for i_seg in range(self.num_seg)]

        self.num_unit = {rsv: int(self.rate_to_power[rsv].shape[1] / 2)
                         for rsv in self.set_rsv}

        self.set_unit = [(rsv, f'unit_{i_unit}')
                         for rsv in self.set_rsv
                         for i_unit in range(self.num_unit[rsv])]

        self.sys_vol_req_daily = pd.read_csv(f'{self.root_path}system_volume_req.csv', index_col=0, dtype=float)

        # Read prediction data
        self.pre_wi_natural = pd.read_csv(f'{self.root_path}day_ahead_inflow.csv', index_col=0, parse_dates=True)
        self.pre_wi_natural_rate = self.pre_wi_natural.drop(columns='Temperature/C')
        self.pre_wi_natural_temp = self.pre_wi_natural.drop(columns='Discharge/cfs')

        self.pre_lmp = pd.read_csv(f'{self.root_path}day_ahead_lmp.csv', index_col=0, parse_dates=True)

        self.pre_p_wind = pd.read_csv(f'{self.root_path}day_ahead_ver_wind.csv', index_col=0, parse_dates=True)
        self.pre_p_wind *= self.wind_scaler
        self.pre_p_wind = self.pre_p_wind.clip(upper=self.wind_ub, lower=self.wind_lb)

        self.pre_p_solar = pd.read_csv(f'{self.root_path}day_ahead_ver_solar.csv', index_col=0, parse_dates=True)
        self.pre_p_solar *= self.solar_scaler
        self.pre_p_solar = self.pre_p_solar.clip(upper=self.solar_ub, lower=self.solar_lb)

        self.pre_p_ver = pd.concat([self.pre_p_wind, self.pre_p_solar], axis=1)
        self.pre_p_ver.columns = self.set_ver
        # Read realization data
        self.act_wi_natural = pd.read_csv(f'{self.root_path}real_time_inflow.csv', index_col=0, parse_dates=True)
        self.act_wi_natural_rate = self.act_wi_natural.drop(columns='Temperature/C')
        self.act_wi_natural_temp = self.act_wi_natural.drop(columns='Discharge/cfs')
        self.wi_natural_rate_ub = self.act_wi_natural_rate.values.max()
        self.wi_natural_rate_lb = self.act_wi_natural_rate.values.min()

        self.act_lmp = pd.read_csv(f'{self.root_path}real_time_lmp.csv', index_col=0, parse_dates=True)
        self.lmp_ub = self.act_lmp.values.max()
        self.lmp_lb = self.act_lmp.values.min()

        self.act_p_wind = pd.read_csv(f'{self.root_path}real_time_ver_wind.csv', index_col=0, parse_dates=True)
        self.act_p_wind *= self.wind_scaler
        self.act_p_wind = self.act_p_wind.clip(upper=self.wind_ub, lower=self.wind_lb)

        self.act_p_solar = pd.read_csv(f'{self.root_path}real_time_ver_solar.csv', index_col=0, parse_dates=True)
        self.act_p_solar *= self.solar_scaler
        self.act_p_solar = self.act_p_solar.clip(upper=self.solar_ub, lower=self.solar_lb)

        self.act_p_ver = pd.concat([self.act_p_wind, self.act_p_solar], axis=1)
        self.act_p_ver.columns = self.set_ver

        # Read scenarios
        self.p_ver_scen_all = pd.read_csv(f'{self.root_path}scenarios_{str(self.ver_cap)}.csv', nrows=len(self.set_hour)*self.num_scen, parse_dates=True)
        self.p_ver_scen_all.index = [f'{scen} {hour}'
                                     for scen in self.set_scen
                                     for hour in self.set_hour]


        # Get the discharge rate-power curve Power = a*Rate^2 + b*Rate + c
        self.fitting_coefficient = {rsv: pd.DataFrame(np.full((self.num_unit[rsv], 3), None),
                                                      index=[(f'unit_{i_unit}') for i_unit in range(self.num_unit[rsv])],
                                                      columns=['a', 'b', 'c'])
                                    for rsv in self.set_rsv}

        self.vol_max = {}
        self.vol_min = {}
        self.unit_rate_max = {}
        self.unit_rate_min = {}
        self.p_hyd_max = {}
        self.p_hyd_min = {}
        self.pwl_slope = {}
        self.pwl_intercept = {}

        for rsv, unit in self.set_unit:
            # Get Vol range
            self.vol_max[rsv] = self.vol_range[rsv].loc[:, 'volume'].max()
            self.vol_min[rsv] = self.vol_range[rsv].loc[:, 'volume'].min()
            # Extract x and y values
            self.rate_sample = self.rate_to_power[rsv][f'rate_of_{unit}/cfs'].dropna().to_numpy()
            self.power_sample = self.rate_to_power[rsv][f'power_of_{unit}/MW'].dropna().to_numpy()
            # Fit quadratic function and save coefficient
            self.fitting_coefficient[rsv].loc[unit, :] = np.polyfit(self.rate_sample, self.power_sample, self.num_order)
            self.a = self.fitting_coefficient[rsv].loc[unit, 'a']
            self.b = self.fitting_coefficient[rsv].loc[unit, 'b']
            self.c = self.fitting_coefficient[rsv].loc[unit, 'c']
            # Get rate range
            self.unit_rate_max[(rsv, unit)] = round(np.roots([self.a, self.b, self.c - self.power_range[rsv].loc[unit, 'max_power']])[1], 3)
            self.unit_rate_min[(rsv, unit)] = round(np.roots([self.a, self.b, self.c - self.power_range[rsv].loc[unit, 'min_power']])[1], 3)
            # Get power range
            self.p_hyd_max[(rsv, unit)] = self.power_range[rsv].loc[unit, 'max_power']
            self.p_hyd_min[(rsv, unit)] = self.power_range[rsv].loc[unit, 'min_power']
            # Get breakpoints for linearizing
            self.rate_breakpoint = np.linspace(self.unit_rate_min[rsv, unit], self.unit_rate_max[rsv, unit], self.num_seg + 1)
            self.power_breakpoint = [self.a * rate ** 2 + self.b * rate + self.c for rate in self.rate_breakpoint]
            # Calculate slopes and intercepts for each of the three segments
            for seg_i, seg_name in enumerate(self.set_seg):
                self.x1, self.x2 = self.rate_breakpoint[seg_i], self.rate_breakpoint[seg_i + 1]
                self.y1, self.y2 = self.power_breakpoint[seg_i], self.power_breakpoint[seg_i + 1]
                self.slope = (self.y2 - self.y1) / (self.x2 - self.x1)
                self.intercept = self.y1 - self.slope * self.x1
                self.pwl_slope[rsv, unit, seg_name] = round(self.slope, 4)
                self.pwl_intercept[rsv, unit, seg_name] = round(self.intercept, 4)

    def get_data(self):
        return self.set_rsv.copy(), \
               self.set_unit.copy(), \
               self.set_ver.copy(), \
               self.set_seg.copy(), \
               self.set_scen.copy(), \
               self.num_unit, \
               self.num_scen, \
               self.minute_per_hour, \
               self.minute_per_subhour, \
               self.pwl_slope.copy(), \
               self.pwl_intercept.copy(), \
               self.vol_max.copy(), \
               self.vol_min.copy(), \
               self.unit_rate_max.copy(), \
               self.unit_rate_min.copy(), \
               self.p_hyd_max.copy(), \
               self.p_hyd_min.copy(), \
               self.pre_wi_natural.copy(), \
               self.pre_wi_natural_rate.copy(), \
               self.pre_wi_natural_temp.copy(), \
               self.pre_lmp.copy(), \
               self.pre_p_wind.copy(), \
               self.pre_p_solar.copy(), \
               self.pre_p_ver.copy(), \
               self.act_wi_natural.copy(), \
               self.act_wi_natural_rate.copy(), \
               self.act_wi_natural_temp.copy(), \
               self.act_lmp.copy(), \
               self.act_p_wind.copy(), \
               self.act_p_solar.copy(), \
               self.act_p_ver.copy(), \
               self.penalty_slack, \
               self.penalty_diff, \
               self.water_value, \
               self.rate_to_vol_DA, \
               self.rate_to_vol_RT, \
               self.p_ver_scen_all.copy(), \
               self.sys_vol_req_daily.copy()