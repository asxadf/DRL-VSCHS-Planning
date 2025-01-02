import numpy as np
import pandas as pd
import pyomo.environ as pyo
import scipy.io
from datetime import datetime, timedelta
from operation_model.lib_orgnize_result import retrieve_result_DA_x, organize_result_DA_x, retrieve_result_RT_z, organize_result_RT_z

dic = globals()

########################################################################################################################
def water_balance_DA_y(model, scen, rsv, time):  # in cfs-day
    if time != model.set_time.last():  # Volume NOT at the end
        return model.var_y_vol_t[scen, rsv, model.set_time.next(time)] \
               == model.var_y_vol_t[scen, rsv, time] + model.var_y_wf_in[scen, rsv, time] - model.var_y_wf_out[scen, rsv, time]
    elif time == model.set_time.last():  # Volume at the end
        return model.var_y_vol_end[scen, rsv] \
               == model.var_y_vol_t[scen, rsv, time] + model.var_y_wf_in[scen, rsv, time] - model.var_y_wf_out[scen, rsv, time]

def wf_in_DA_y(model, scen, rsv, time):  # in cfs-day
    base_inflow = model.para_rate_to_vol * model.para_pre_wi_natural_rate[rsv, time]  # cfs-day
    if rsv == model.set_rsv.first():
        # If it's the first reservoir, only natural inflow affects it
        return model.var_y_wf_in[scen, rsv, time] == base_inflow

    elif rsv != model.set_rsv.first() and time == model.set_time.first():
        # For downstream reservoirs in the first time period, consider yesterday's delayed inflow
        delayed_inflow = model.para_wf_in_delay_from_yesterday[rsv]  # cfs-day
        return model.var_y_wf_in[scen, rsv, time] == base_inflow + delayed_inflow

    elif rsv != model.set_rsv.first() and time != model.set_time.first():
        # For downstream reservoirs and not the first time period
        upstream_rsv = model.set_rsv.prev(rsv)
        prev_time = model.set_time.prev(time)
        delayed_inflow = (model.var_y_wf_spill[scen, upstream_rsv, prev_time]
                          + model.para_rate_to_vol * sum(model.var_y_rate[scen, upstream_rsv, :, prev_time]))
        return model.var_y_wf_in[scen, rsv, time] == base_inflow + delayed_inflow

def operation_DA_mode(DatabaseSpec, date, input_vol_ini, input_mid_term_planning, input_wf_in_delay_yesterday):
    mode = DatabaseSpec.mode
    # Initialize
    input_set_rsv, \
    input_set_unit, \
    input_set_ver, \
    input_set_seg, \
    input_set_scen, \
    input_num_unit, \
    input_num_scen, \
    input_minute_per_hour, \
    input_minute_per_subhour, \
    input_pwl_slope, \
    input_pwl_intercept, \
    input_vol_max, \
    input_vol_min, \
    input_unit_rate_max, \
    input_unit_rate_min, \
    input_p_hyd_max, \
    input_p_hyd_min, \
    input_pre_wi_natural_all, \
    input_pre_wi_natural_rate_all, \
    input_pre_wi_natural_temp_all, \
    input_pre_lmp_all, \
    input_pre_p_wind_all, \
    input_pre_p_solar_all, \
    input_pre_p_ver_all, \
    input_act_wi_natural_all, \
    input_act_wi_natural_rate_all, \
    input_act_wi_natural_temp_all, \
    input_act_lmp_all, \
    input_act_p_wind_all, \
    input_act_p_solar_all, \
    input_act_p_ver_all, \
    input_penalty_slack, \
    input_penalty_diff, \
    input_water_value, \
    input_rate_to_vol_DA, \
    input_rate_to_vol_RT, \
    input_p_ver_scen_all, \
    input_sys_vol_req_daily = DatabaseSpec.get_data()

    # Set of time in hourly resolution
    datetime_1st = pd.to_datetime(f"{date} 00:00:00")
    datetime_end = pd.to_datetime(f"{date} 23:00:00")
    input_set_time_with_date = list(pd.date_range(start=datetime_1st, end=datetime_end, freq='1h').strftime('%Y-%m-%d %H:00:00'))
    input_set_time = list(pd.date_range(start=datetime_1st, end=datetime_end, freq='1h').strftime('%H:00:00'))

    # Read scenarios
    input_p_ver_scen = {(scen, time): input_p_ver_scen_all.at[f'{scen} {time}', date]
                        for scen in input_set_scen
                        for time in input_set_time}

    # Get LMP predictions
    input_pre_lmp = dict(zip(input_set_time, input_pre_lmp_all.loc[input_set_time_with_date, 'LMP']))

    # Get water inflow predictions
    input_pre_wi_natural_rate_df = pd.DataFrame(0, index=input_set_time, columns=input_set_rsv)
    input_pre_wi_natural_rate_df.iloc[:, 0] = input_pre_wi_natural_rate_all.loc[input_set_time_with_date, :].iloc[:,
                                              0].values
    input_pre_wi_natural_rate = {(rsv, time): input_pre_wi_natural_rate_df.at[time, rsv]
                                 for time in input_set_time
                                 for rsv in input_set_rsv}

    # Get VER predictions
    input_pre_p_ver_df = input_pre_p_ver_all.loc[input_set_time_with_date, :]
    input_pre_p_ver_df.index = input_set_time
    input_pre_p_ver = {(ver, time): input_pre_p_ver_df.at[time, ver]
                       for time in input_set_time
                       for ver in input_set_ver}

    model = pyo.ConcreteModel(name='day_ahead_model')

    # Sets
    model.set_rsv = pyo.Set(initialize=input_set_rsv)
    model.set_unit = pyo.Set(initialize=input_set_unit)
    model.set_ver = pyo.Set(initialize=input_set_ver)
    model.set_time = pyo.Set(initialize=input_set_time)
    model.set_seg = pyo.Set(initialize=input_set_seg)
    model.set_scen = pyo.Set(initialize=input_set_scen)
    # System parameters
    model.para_minute_per_time = pyo.Param(initialize=input_minute_per_hour)
    model.para_penalty_slack = pyo.Param(initialize=input_penalty_slack)
    model.para_penalty_diff = pyo.Param(initialize=input_penalty_diff)
    model.para_num_rsv = pyo.Param(initialize=len(input_set_rsv))
    model.para_num_seg = pyo.Param(initialize=len(input_set_seg))
    model.para_num_unit = pyo.Param(model.set_rsv, initialize=input_num_unit)
    model.para_rate_to_vol = pyo.Param(initialize=input_rate_to_vol_DA)  # [cfs-day/cfs]
    model.para_vol_max = pyo.Param(model.set_rsv, initialize=input_vol_max)
    model.para_vol_min = pyo.Param(model.set_rsv, initialize=input_vol_min)
    model.para_rate_max = pyo.Param(model.set_unit, initialize=input_unit_rate_max)
    model.para_rate_min = pyo.Param(model.set_unit, initialize=input_unit_rate_min)
    model.para_p_hyd_max = pyo.Param(model.set_unit, initialize=input_p_hyd_max)
    model.para_p_hyd_min = pyo.Param(model.set_unit, initialize=input_p_hyd_min)
    model.para_pwl_slope = pyo.Param(model.set_unit, model.set_seg, initialize=input_pwl_slope)
    model.para_pwl_intercept = pyo.Param(model.set_unit, model.set_seg, initialize=input_pwl_intercept)
    model.para_scen_prob = pyo.Param(initialize=round(1/input_num_scen, 2))
    # Parameters that change every model
    model.para_vol_ini = pyo.Param(model.set_rsv, within=pyo.NonNegativeReals,
                                   initialize=input_vol_ini, mutable=True)

    model.para_mid_term_planning = pyo.Param(model.set_rsv, within=pyo.NonNegativeReals,
                                             initialize=input_mid_term_planning, mutable=True)

    model.para_wf_in_delay_from_yesterday = pyo.Param(model.set_rsv, within=pyo.NonNegativeReals,
                                                      initialize=input_wf_in_delay_yesterday, mutable=True)

    model.para_pre_lmp = pyo.Param(model.set_time, within=pyo.Reals,
                                   initialize=input_pre_lmp, mutable=True)

    model.para_pre_p_ver = pyo.Param(model.set_ver, model.set_time, within=pyo.NonNegativeReals,
                                     initialize=input_pre_p_ver, mutable=True)

    model.para_pre_wi_natural_rate = pyo.Param(model.set_rsv, model.set_time, within=pyo.NonNegativeReals,
                                               initialize=input_pre_wi_natural_rate, mutable=True)

    model.para_pre_p_ver_scen = pyo.Param(model.set_scen, model.set_time, within=pyo.NonNegativeReals,
                                          initialize=input_p_ver_scen, mutable=True)
    # X Variables
    model.var_x_i = pyo.Var(model.set_unit, model.set_time, domain=pyo.Binary)
    model.var_x_p_cmit = pyo.Var(model.set_time, domain=pyo.NonNegativeReals)
    model.var_x_p_hyd = pyo.Var(model.set_unit, model.set_time, domain=pyo.NonNegativeReals)
    model.var_x_p_ver = pyo.Var(model.set_time, domain=pyo.NonNegativeReals)
    model.var_x_revenue_gross = pyo.Var(model.set_time, domain=pyo.Reals)

    # Y Variables
    model.var_y_charge = pyo.Var(model.set_scen, domain=pyo.Reals)
    model.var_y_i = pyo.Var(model.set_scen, model.set_unit, model.set_time, domain=pyo.Binary)
    model.var_y_p_act = pyo.Var(model.set_scen, model.set_time, domain=pyo.NonNegativeReals)
    model.var_y_p_hyd = pyo.Var(model.set_scen, model.set_unit, model.set_time, domain=pyo.NonNegativeReals)
    model.var_y_p_ver = pyo.Var(model.set_scen, model.set_time, domain=pyo.NonNegativeReals)
    model.var_y_rate = pyo.Var(model.set_scen, model.set_unit, model.set_time, domain=pyo.NonNegativeReals)
    model.var_y_vol_t = pyo.Var(model.set_scen, model.set_rsv, model.set_time, domain=pyo.NonNegativeReals)
    model.var_y_vol_end = pyo.Var(model.set_scen, model.set_rsv, domain=pyo.NonNegativeReals)
    model.var_y_vol_end_slack = pyo.Var(model.set_scen, model.set_rsv, domain=pyo.NonNegativeReals)
    model.var_y_wf_in = pyo.Var(model.set_scen, model.set_rsv, model.set_time, domain=pyo.NonNegativeReals)
    model.var_y_wf_out = pyo.Var(model.set_scen, model.set_rsv, model.set_time, domain=pyo.NonNegativeReals)
    model.var_y_wf_spill = pyo.Var(model.set_scen, model.set_rsv, model.set_time, domain=pyo.NonNegativeReals)
    model.var_y_future_value = pyo.Var(model.set_scen, model.set_rsv, domain=pyo.NonNegativeReals)

    # X Constraint: Committed power
    model.cons_x_p_cmit = pyo.Constraint(model.set_time,
                                         rule=lambda model, time:
                                         model.var_x_p_cmit[time] == sum(model.var_x_p_hyd[:, :, time]) +
                                         model.var_x_p_ver[time])

    # X Constraint: VER limit
    model.cons_x_p_ver_ub = pyo.Constraint(model.set_time,
                                           rule=lambda model, time:
                                           model.var_x_p_ver[time] <= sum(model.para_pre_p_ver[:, time]))

    # X Constraint: Hydro power limit
    model.cons_x_p_hyd_lb = pyo.Constraint(model.set_unit, model.set_time,
                                           rule=lambda model, rsv, unit, time:
                                           model.var_x_p_hyd[rsv, unit, time]
                                           >= model.var_x_i[rsv, unit, time] * model.para_p_hyd_min[rsv, unit])

    model.cons_x_p_hyd_ub = pyo.Constraint(model.set_unit, model.set_time,
                                           rule=lambda model, rsv, unit, time:
                                           model.var_x_p_hyd[rsv, unit, time]
                                           <= model.var_x_i[rsv, unit, time] * model.para_p_hyd_max[rsv, unit])

    # X Constraint: Gross revenue
    model.cons_x_revenue_gross = pyo.Constraint(model.set_time,
                                                rule=lambda model, time:
                                                model.var_x_revenue_gross[time]
                                                == model.para_pre_lmp[time] * model.var_x_p_cmit[time])

    # Y Constraint: Initial storage
    model.cons_y_vol_ini = pyo.Constraint(model.set_scen, model.set_rsv,
                                          rule=lambda model, scen, rsv:
                                          model.para_vol_ini[rsv] == model.var_y_vol_t[scen, rsv, model.set_time.first()])

    # Y Constraint: Charge
    model.cons_y_charge = pyo.Constraint(model.set_scen,
                                         rule=lambda model, scen:
                                         model.var_y_charge[scen]
                                         == sum(model.para_penalty_diff*(model.var_x_p_cmit[time] - model.var_y_p_act[scen, time])
                                                for time in model.set_time))

    # Y Constraint: Actual power ub
    model.cons_y_p_act_ub = pyo.Constraint(model.set_scen, model.set_time,
                                           rule=lambda model, scen, time:
                                           model.var_x_p_cmit[time] >= model.var_y_p_act[scen, time])

    # Y Constraint: Actual power
    model.cons_y_p_act = pyo.Constraint(model.set_scen, model.set_time,
                                        rule=lambda model, scen, time:
                                        model.var_y_p_act[scen, time] == sum(model.var_y_p_hyd[scen, :, :, time]) +
                                        model.var_y_p_ver[scen, time])

    # Y Constraint: VER limit
    model.cons_y_p_ver_ub = pyo.Constraint(model.set_scen, model.set_time,
                                           rule=lambda model, scen, time:
                                           model.var_y_p_ver[scen, time] <= model.para_pre_p_ver_scen[scen, time])

    # Y Constraint: Storage limit
    model.cons_y_vol_lb = pyo.Constraint(model.set_scen, model.set_rsv, model.set_time,
                                         rule=lambda model, scen, rsv, time:
                                         model.var_y_vol_t[scen, rsv, time] >= model.para_vol_min[rsv])

    model.cons_y_vol_ub = pyo.Constraint(model.set_scen, model.set_rsv, model.set_time,
                                         rule=lambda model, scen, rsv, time:
                                         model.var_y_vol_t[scen, rsv, time] <= model.para_vol_max[rsv])

    model.cons_y_vol_end_lb = pyo.Constraint(model.set_scen, model.set_rsv,
                                             rule=lambda model, scen, rsv:
                                             model.var_y_vol_end[scen, rsv] >= model.para_vol_min[rsv])

    model.cons_y_vol_end_ub = pyo.Constraint(model.set_scen, model.set_rsv,
                                             rule=lambda model, scen, rsv:
                                             model.var_y_vol_end[scen, rsv] <= model.para_vol_max[rsv])

    # Y Constraint: Water balance
    model.cons_y_water_balance = pyo.Constraint(model.set_scen, model.set_rsv, model.set_time,
                                                rule=water_balance_DA_y)

    # Y Constraint: Scheduled water flow
    model.cons_y_wf_in = pyo.Constraint(model.set_scen, model.set_rsv, model.set_time,
                                        rule=wf_in_DA_y)

    model.cons_y_wf_out = pyo.Constraint(model.set_scen, model.set_rsv, model.set_time,
                                         rule=lambda model, scen, rsv, time:
                                         model.var_y_wf_out[scen, rsv, time]
                                         == model.para_rate_to_vol * sum(model.var_y_rate[scen, rsv, :, time]) +
                                         model.var_y_wf_spill[scen, rsv, time])

    # Y Constraint: Piece-wise linearize hydropower
    model.cons_y_pwl = pyo.Constraint(model.set_scen, model.set_unit, model.set_seg, model.set_time,
                                      rule=lambda model, scen, rsv, unit, seg, time:
                                      model.var_y_p_hyd[scen, rsv, unit, time]
                                      == model.para_pwl_slope[rsv, unit, seg] * model.var_y_rate[scen, rsv, unit, time])

    # Y Constraint: Hydro power limit
    model.cons_y_p_hyd_lb = pyo.Constraint(model.set_scen, model.set_unit, model.set_time,
                                           rule=lambda model, scen, rsv, unit, time:
                                           model.var_y_p_hyd[scen, rsv, unit, time]
                                           >= model.var_y_i[scen, rsv, unit, time] * model.para_p_hyd_min[rsv, unit])

    model.cons_y_p_hyd_ub = pyo.Constraint(model.set_scen, model.set_unit, model.set_time,
                                           rule=lambda model, scen, rsv, unit, time:
                                           model.var_y_p_hyd[scen, rsv, unit, time]
                                           <= model.var_y_i[scen, rsv, unit, time] * model.para_p_hyd_max[rsv, unit])

    # Y Constraint: Rate limit
    model.cons_y_rate_lb = pyo.Constraint(model.set_scen, model.set_unit, model.set_time,
                                          rule=lambda model, scen, rsv, unit, time:
                                          model.var_y_rate[scen, rsv, unit, time]
                                          >= model.var_y_i[scen, rsv, unit, time] * model.para_rate_min[rsv, unit])

    model.cons_y_rate_ub = pyo.Constraint(model.set_scen, model.set_unit, model.set_time,
                                          rule=lambda model, scen, rsv, unit, time:
                                          model.var_y_rate[scen, rsv, unit, time]
                                          <= model.var_y_i[scen, rsv, unit, time] * model.para_rate_max[rsv, unit])


    if mode == 'EODS':
        model.cons_y_vol_end_req = pyo.Constraint(model.set_scen, model.set_rsv,
                                                  rule=lambda model, scen, rsv:
                                                  model.para_mid_term_planning[rsv]
                                                  == model.var_y_vol_end[scen, rsv] + model.var_y_vol_end_slack[scen, rsv])
        # Obj
        model.obj = pyo.Objective(expr=sum(model.var_x_revenue_gross[:])
                                       - model.para_scen_prob * sum(model.var_y_charge[:])
                                       - model.para_scen_prob * model.para_penalty_slack * sum(model.var_y_vol_end_slack[:, :]), sense=pyo.maximize)
    elif mode == 'LMWV':
        model.cons_y_future_value = pyo.Constraint(model.set_scen, model.set_rsv,
                                                   rule=lambda model, scen, rsv:
                                                   model.var_y_future_value[scen, rsv]
                                                   == model.para_mid_term_planning[rsv] * model.var_y_vol_end[scen, rsv])
        # Obj
        model.obj = pyo.Objective(expr=sum(model.var_x_revenue_gross[:])
                                       + model.para_scen_prob*sum(model.var_y_future_value[:, :])
                                       - model.para_scen_prob*sum(model.var_y_charge[:]), sense=pyo.maximize)

    # Solve it
    solver = pyo.SolverFactory('gurobi')
    solver.options['MipGap'] = 0.01
    optimization = solver.solve(model, tee=False)
    # Check feasibility
    solver_status = optimization.solver.status
    termination_condition = optimization.solver.termination_condition
    if solver_status == 'ok' and termination_condition == 'optimal':
        is_optimal = True
        print(f"{date}: The day-ahead model has been solved optimally!")
    else:
        is_optimal = False
        print(f"{date}: The day-ahead model cannot be solved optimally.")

    # Get results
    result = retrieve_result_DA_x(model, is_optimal)

    return model, result

########################################################################################################################
def wf_in_RT_z(model, rsv):  # in cfs-day
    base_inflow = model.para_rate_to_vol * model.para_act_wi_natural_rate[rsv]
    if rsv == model.set_rsv.first():
        # If it's the first reservoir or the first time period, only natural inflow affects it
        return model.var_z_wf_in[rsv] == base_inflow
    else:
        return model.var_z_wf_in[rsv] == base_inflow + model.para_wf_in_delay_from_lasthour[rsv]

def operation_RT(DatabaseSpec, date, subhour, input_vol_ini, input_p_cmit, input_wf_in_delay_from_lasthour):
    # Initialize
    input_set_rsv, \
    input_set_unit, \
    input_set_ver, \
    input_set_seg, \
    input_set_scen, \
    input_num_unit, \
    input_num_scen, \
    input_minute_per_hour, \
    input_minute_per_subhour, \
    input_pwl_slope, \
    input_pwl_intercept, \
    input_vol_max, \
    input_vol_min, \
    input_unit_rate_max, \
    input_unit_rate_min, \
    input_p_hyd_max, \
    input_p_hyd_min, \
    input_pre_wi_natural_all, \
    input_pre_wi_natural_rate_all, \
    input_pre_wi_natural_temp_all, \
    input_pre_lmp_all, \
    input_pre_p_wind_all, \
    input_pre_p_solar_all, \
    input_pre_p_ver_all, \
    input_act_wi_natural_all, \
    input_act_wi_natural_rate_all, \
    input_act_wi_natural_temp_all, \
    input_act_lmp_all, \
    input_act_p_wind_all, \
    input_act_p_solar_all, \
    input_act_p_ver_all, \
    input_penalty_slack, \
    input_penalty_diff, \
    input_water_value, \
    input_rate_to_vol_DA, \
    input_rate_to_vol_RT, \
    input_p_ver_scen_all, \
    input_sys_vol_req_daily= DatabaseSpec.get_data()

    # Set of time in subhourly resolution
    datetime_1st = pd.to_datetime(f"{date} 00:00:00")
    datetime_end = pd.to_datetime(f"{date} 23:55:00")
    input_set_time_with_date = list(
        pd.date_range(start=datetime_1st, end=datetime_end, freq='5min').strftime('%Y-%m-%d %H:%M:00'))
    input_set_time = list(pd.date_range(start=datetime_1st, end=datetime_end, freq='5min').strftime('%H:%M:00'))

    # Get LMP realizations
    input_act_lmp_daily = dict(zip(input_set_time, input_act_lmp_all.loc[input_set_time_with_date, 'LMP']))

    # Get water inflow realizations
    input_act_wi_natural_rate_daily_df = pd.DataFrame(0, index=input_set_time, columns=input_set_rsv)
    input_act_wi_natural_rate_daily_df.iloc[:, 0] = input_act_wi_natural_rate_all.loc[input_set_time_with_date, :].iloc[
                                                    :, 0].values
    input_act_wi_natural_rate_daily = {(rsv, time): input_act_wi_natural_rate_daily_df.at[time, rsv]
                                       for time in input_set_time
                                       for rsv in input_set_rsv}

    # Get RES realizations
    input_act_p_ver_daily_df = input_act_p_ver_all.loc[input_set_time_with_date, :]
    input_act_p_ver_daily_df.index = input_set_time
    input_act_p_ver_daily = {(ver, time): input_act_p_ver_daily_df.at[time, ver]
                             for time in input_set_time
                             for ver in input_set_ver}

    # Interval information
    input_act_lmp = input_act_lmp_daily[subhour]
    input_act_wi_natural_rate = {rsv: input_act_wi_natural_rate_daily[rsv, subhour] for rsv in input_set_rsv}
    input_act_p_ver = {ver: input_act_p_ver_daily[ver, subhour] for ver in input_set_ver}

    #
    model = pyo.ConcreteModel(name='real_time_model')

    # Set
    model.set_rsv = pyo.Set(initialize=input_set_rsv)
    model.set_unit = pyo.Set(initialize=input_set_unit)
    model.set_ver = pyo.Set(initialize=input_set_ver)
    model.set_seg = pyo.Set(initialize=input_set_seg)
    # System parameters
    model.para_minute_per_time = pyo.Param(initialize=input_minute_per_subhour)
    model.para_water_value = pyo.Param(initialize=input_water_value)
    model.para_num_rsv = pyo.Param(initialize=len(input_set_rsv))
    model.para_num_seg = pyo.Param(initialize=len(input_set_seg))
    model.para_num_unit = pyo.Param(model.set_rsv, initialize=input_num_unit)
    model.para_penalty_diff = pyo.Param(initialize=input_penalty_diff)
    model.para_rate_to_vol = pyo.Param(initialize=input_rate_to_vol_RT)  # [cfs-day/cfs]
    model.para_vol_max = pyo.Param(model.set_rsv, initialize=input_vol_max)
    model.para_vol_min = pyo.Param(model.set_rsv, initialize=input_vol_min)
    model.para_rate_max = pyo.Param(model.set_unit, initialize=input_unit_rate_max)
    model.para_rate_min = pyo.Param(model.set_unit, initialize=input_unit_rate_min)
    model.para_p_hyd_max = pyo.Param(model.set_unit, initialize=input_p_hyd_max)
    model.para_p_hyd_min = pyo.Param(model.set_unit, initialize=input_p_hyd_min)
    model.para_pwl_slope = pyo.Param(model.set_unit, model.set_seg, initialize=input_pwl_slope)
    model.para_pwl_intercept = pyo.Param(model.set_unit, model.set_seg, initialize=input_pwl_intercept)
    # Parameters that change every model
    model.para_p_cmit = pyo.Param(within=pyo.NonNegativeReals, initialize=input_p_cmit, mutable=True)
    model.para_act_lmp = pyo.Param(within=pyo.Reals, initialize=input_act_lmp, mutable=True)
    model.para_act_p_ver = pyo.Param(model.set_ver, within=pyo.NonNegativeReals, initialize=input_act_p_ver,
                                     mutable=True)
    model.para_vol_ini = pyo.Param(model.set_rsv, within=pyo.NonNegativeReals, initialize=input_vol_ini, mutable=True)
    model.para_wf_in_delay_from_lasthour = pyo.Param(model.set_rsv,
                                                     within=pyo.NonNegativeReals,
                                                     initialize=input_wf_in_delay_from_lasthour, mutable=True)
    model.para_act_wi_natural_rate = pyo.Param(model.set_rsv, within=pyo.NonNegativeReals,
                                               initialize=input_act_wi_natural_rate, mutable=True)
    # Variables
    model.var_z_i = pyo.Var(model.set_unit, domain=pyo.Binary)
    model.var_z_p_act = pyo.Var(domain=pyo.NonNegativeReals)  # in MW
    model.var_z_p_hyd = pyo.Var(model.set_unit, domain=pyo.NonNegativeReals)  # in MW
    model.var_z_p_ver = pyo.Var(domain=pyo.NonNegativeReals)  # in MW
    model.var_z_rate = pyo.Var(model.set_unit, domain=pyo.NonNegativeReals)  # in cfs
    model.var_z_vol_end = pyo.Var(model.set_rsv, domain=pyo.NonNegativeReals)  # in cfs-day
    model.var_z_wf_in = pyo.Var(model.set_rsv, domain=pyo.NonNegativeReals)  # in cfs-day
    model.var_z_wf_out = pyo.Var(model.set_rsv, domain=pyo.NonNegativeReals)  # in cfs-day
    model.var_z_wf_spill = pyo.Var(model.set_rsv, domain=pyo.NonNegativeReals)  # in cfs-day
    model.var_z_p_difference = pyo.Var(domain=pyo.NonNegativeReals)  # in MW

    # Z Constraint: Power difference
    model.cons_z_difference = pyo.ConstraintList()
    model.cons_z_difference.add(model.var_z_p_difference == model.para_p_cmit - model.var_z_p_act)
    model.cons_z_difference.add(model.para_p_cmit >= model.var_z_p_act)

    # Z Constraint: actual power
    model.cons_z_p_actual = pyo.ConstraintList()
    model.cons_z_p_actual.add(model.var_z_p_act == sum(model.var_z_p_hyd[:, :]) + model.var_z_p_ver)

    # Z Constraint: VER limit
    model.cons_z_p_ver_ub = pyo.Constraint(expr=model.var_z_p_ver <= sum(model.para_act_p_ver[:]))

    # Z Constraint: Storage limit
    model.cons_z_vol_end_lb = pyo.Constraint(model.set_rsv,
                                             rule=lambda model, rsv:
                                             model.var_z_vol_end[rsv] >= model.para_vol_min[rsv])

    model.cons_z_vol_end_ub = pyo.Constraint(model.set_rsv,
                                             rule=lambda model, rsv:
                                             model.var_z_vol_end[rsv] <= model.para_vol_max[rsv])

    # Z Constraint: Water balance
    model.cons_z_water_balance = pyo.Constraint(model.set_rsv,
                                                rule=lambda model, rsv:
                                                model.var_z_vol_end[rsv] == model.para_vol_ini[rsv] + model.var_z_wf_in[rsv] -
                                                model.var_z_wf_out[rsv])

    # Z Constraint: Water inflow scheduled to flow in
    model.cons_z_wi_sche_in = pyo.Constraint(model.set_rsv, rule=wf_in_RT_z)

    # Z Constraint: Water inflow scheduled to flow out
    model.cons_z_water_sche_out = pyo.Constraint(model.set_rsv,
                                                 rule=lambda model, rsv:
                                                 model.var_z_wf_out[rsv]
                                                 == model.var_z_wf_spill[rsv] + model.para_rate_to_vol * sum(model.var_z_rate[rsv, :]))

    # Z Constraint: Piece-wise linearize hydropower
    model.cons_z_pwl = pyo.Constraint(model.set_unit, model.set_seg,
                                      rule=lambda model, rsv, unit, seg:
                                      model.var_z_p_hyd[rsv, unit]
                                      == model.para_pwl_slope[rsv, unit, seg] * model.var_z_rate[rsv, unit])

    # Z Constraint: Hydro power limit
    model.cons_z_p_hyd_lb = pyo.Constraint(model.set_unit,
                                           rule=lambda model, rsv, unit:
                                           model.var_z_p_hyd[rsv, unit] >= model.var_z_i[rsv, unit] *
                                           model.para_p_hyd_min[rsv, unit])

    model.cons_z_p_hyd_ub = pyo.Constraint(model.set_unit,
                                           rule=lambda model, rsv, unit:
                                           model.var_z_p_hyd[rsv, unit] <= model.var_z_i[rsv, unit] *
                                           model.para_p_hyd_max[rsv, unit])

    # Constraint: Rate limit
    model.cons_z_rate_lb = pyo.Constraint(model.set_unit,
                                          rule=lambda model, rsv, unit:
                                          model.var_z_rate[rsv, unit]
                                          >= model.var_z_i[rsv, unit] * model.para_rate_min[rsv, unit])

    model.cons_z_rate_ub = pyo.Constraint(model.set_unit,
                                          rule=lambda model, rsv, unit:
                                          model.var_z_rate[rsv, unit]
                                          <= model.var_z_i[rsv, unit] * model.para_rate_max[rsv, unit])

    # Obj
    model.obj = pyo.Objective(expr=model.para_penalty_diff * model.var_z_p_difference - model.para_water_value * sum(model.var_z_vol_end[:]),
                              sense=pyo.minimize)

    solver = pyo.SolverFactory('gurobi')  # Using the Gurobi Persistent interface
    solver.options['MipGap'] = 0.00

    # Solve it
    optimization = solver.solve(model, tee=False)

    # Check feasibility
    solver_status = optimization.solver.status
    termination_condition = optimization.solver.termination_condition
    if solver_status == 'ok' and termination_condition == 'optimal':
        is_optimal = True
        print(f"{date} {subhour}: The real-time model has been solved optimally!")
    else:
        is_optimal = False
        print(f"{date} {subhour}: The real-time model cannot be solved optimally.")

    result = retrieve_result_RT_z(model, is_optimal)
    return model, result

def operation_RT_by_mp(DatabaseSpec, theta, cr_result, theta_act_lmp):
    """
    theta = {'para_p_cmit': theta_p_cmit,
             'para_vol_ini_R0': theta_vol_ini_R0,
             'para_vol_ini_R1': theta_vol_ini_R1,
             'para_wf_in_R0': theta_wf_in_R0,
             'para_wf_in_R1': theta_wf_in_R1,
             'para_p_act_ver': theta_p_act_ver}
    """
    theta['para_p_act_ver'] = float(round(theta['para_p_act_ver'], 0))

    theta_values = np.array(list(theta.values())).reshape((len(theta.keys()), 1))
    var = {'var_z_rate_R0U0': None,
           'var_z_rate_R0U1': None,
           'var_z_rate_R0U2': None,
           'var_z_rate_R1U0': None,
           'var_z_rate_R1U1': None,
           'var_z_rate_R1U2': None,
           'var_z_p_ver': None,
           'var_z_wf_spill_R0': None,
           'var_z_wf_spill_R1': None,
           'var_z_p_difference': None,
           'var_z_i_R0U0': None,
           'var_z_i_R0U1': None,
           'var_z_i_R0U2': None,
           'var_z_i_R1U0': None,
           'var_z_i_R1U1': None,
           'var_z_i_R1U2': None}

    # Recover para
    para_act_lmp = theta_act_lmp
    para_act_p_ver = theta['para_p_act_ver']
    para_act_wi_natural_rate = {DatabaseSpec.set_rsv[0]: theta['para_wf_in_R0'] / DatabaseSpec.rate_to_vol_RT,
                                DatabaseSpec.set_rsv[1]: 0}

    para_vol_ini = {DatabaseSpec.set_rsv[0]: theta['para_vol_ini_R0'],
                    DatabaseSpec.set_rsv[1]: theta['para_vol_ini_R1']}

    # Search the CR
    cr_fall = []
    for n in range(int(len(cr_result) / 4)):
        if (np.dot(cr_result[f'CR{n}_A'], theta_values) <= cr_result[f'CR{n}_b']).all():
            cr_fall.append(n)

    if not cr_fall:
        print(f'Not CR has been found. theta: {theta}.')

    # Get vars' names and values
    var_value_list = (np.dot(cr_result[f'CR{cr_fall[0]}_Slope'], theta_values)
                      + cr_result[f'CR{cr_fall[0]}_Constant']).flatten().tolist()

    for var_idx, var_name in enumerate(var.keys()):
        var[var_name] = var_value_list[var_idx]

    # Recover var
    var_z_i = {DatabaseSpec.set_unit[0]: var['var_z_i_R0U0'],
               DatabaseSpec.set_unit[1]: var['var_z_i_R0U1'],
               DatabaseSpec.set_unit[2]: var['var_z_i_R0U2'],
               DatabaseSpec.set_unit[3]: var['var_z_i_R1U0'],
               DatabaseSpec.set_unit[4]: var['var_z_i_R1U1'],
               DatabaseSpec.set_unit[5]: var['var_z_i_R1U2']}

    var_z_p_hyd = {DatabaseSpec.set_unit[0]: list(DatabaseSpec.pwl_slope.values())[0] * var['var_z_rate_R0U0'],
                   DatabaseSpec.set_unit[1]: list(DatabaseSpec.pwl_slope.values())[1] * var['var_z_rate_R0U1'],
                   DatabaseSpec.set_unit[2]: list(DatabaseSpec.pwl_slope.values())[2] * var['var_z_rate_R0U2'],
                   DatabaseSpec.set_unit[3]: list(DatabaseSpec.pwl_slope.values())[3] * var['var_z_rate_R1U0'],
                   DatabaseSpec.set_unit[4]: list(DatabaseSpec.pwl_slope.values())[4] * var['var_z_rate_R1U1'],
                   DatabaseSpec.set_unit[5]: list(DatabaseSpec.pwl_slope.values())[5] * var['var_z_rate_R1U2']}

    var_z_rate = {DatabaseSpec.set_unit[0]: var['var_z_rate_R0U0'],
                  DatabaseSpec.set_unit[1]: var['var_z_rate_R0U1'],
                  DatabaseSpec.set_unit[2]: var['var_z_rate_R0U2'],
                  DatabaseSpec.set_unit[3]: var['var_z_rate_R1U0'],
                  DatabaseSpec.set_unit[4]: var['var_z_rate_R1U1'],
                  DatabaseSpec.set_unit[5]: var['var_z_rate_R1U2']}

    var_z_vol_end = {DatabaseSpec.set_rsv[0]:
                         round(theta['para_vol_ini_R0'] + theta['para_wf_in_R0'] \
                               - DatabaseSpec.rate_to_vol_RT * (var['var_z_rate_R0U0']
                                                                + var['var_z_rate_R0U1']
                                                                + var['var_z_rate_R0U2']) - var['var_z_wf_spill_R0'], 2),

                     DatabaseSpec.set_rsv[1]:
                         round(theta['para_vol_ini_R1'] + theta['para_wf_in_R1'] \
                               - DatabaseSpec.rate_to_vol_RT * (var['var_z_rate_R1U0']
                                                                + var['var_z_rate_R1U1']
                                                                + var['var_z_rate_R1U2']) - var['var_z_wf_spill_R1'], 2)}

    var_z_wf_in = {DatabaseSpec.set_rsv[0]: theta['para_wf_in_R0'],
                   DatabaseSpec.set_rsv[1]: theta['para_wf_in_R1']}

    var_z_wf_out = {DatabaseSpec.set_rsv[0]:
                        DatabaseSpec.rate_to_vol_RT * (var['var_z_rate_R0U0']
                                                       + var['var_z_rate_R0U1']
                                                       + var['var_z_rate_R0U2']) + var['var_z_wf_spill_R0'],

                    DatabaseSpec.set_rsv[1]:
                        DatabaseSpec.rate_to_vol_RT * (var['var_z_rate_R1U0']
                                                       + var['var_z_rate_R1U1']
                                                       + var['var_z_rate_R1U2']) + var['var_z_wf_spill_R1']}

    var_z_wf_spill = {DatabaseSpec.set_rsv[0]: var['var_z_wf_spill_R0'],
                      DatabaseSpec.set_rsv[1]: var['var_z_wf_spill_R1']}

    var_z_p_ver = var['var_z_p_ver']

    var_z_p_act = var_z_p_ver + sum(var_z_p_hyd.values())

    var_z_p_difference = var['var_z_p_difference']

    result_RT = {'para_act_lmp': para_act_lmp,
                 'para_act_p_ver': para_act_p_ver,
                 'para_act_wi_natural_rate': para_act_wi_natural_rate,
                 'para_vol_ini': para_vol_ini,
                 'var_z_i': var_z_i,
                 'var_z_p_hyd': var_z_p_hyd,
                 'var_z_rate': var_z_rate,
                 'var_z_vol_end': var_z_vol_end,
                 'var_z_wf_in': var_z_wf_in,
                 'var_z_wf_out': var_z_wf_out,
                 'var_z_wf_spill': var_z_wf_spill,
                 'var_z_p_ver': var_z_p_ver,
                 'var_z_p_act': var_z_p_act,
                 'var_z_p_difference': var_z_p_difference}

    return result_RT

########################################################################################################################
def auto_DA_RT_multiple_day(DatabaseSpec, date_1st, date_end):
    mode = DatabaseSpec.mode

    # Initialize solver setting
    solver = pyo.SolverFactory('gurobi')
    solver.options['MipGap'] = 0.00

    # Initialize date and time
    tf_date = DatabaseSpec.tf_date
    tf_hour = DatabaseSpec.tf_hour
    tf_subhour = DatabaseSpec.tf_subhour

    # Initialize set
    set_rsv = DatabaseSpec.set_rsv
    set_ver = DatabaseSpec.set_ver
    set_date = pd.date_range(start=date_1st, end=date_end).strftime(tf_date).tolist()
    set_hour = DatabaseSpec.set_hour
    set_subhour = DatabaseSpec.set_subhour
    set_subhour_last_hour = DatabaseSpec.set_subhour_last_hour
    set_scen = DatabaseSpec.set_scen

    # Initialize recorder of DA
    if mode == 'EODS':
        mid_term_planning_by_date = {set_date[0]: {set_rsv[0]: DatabaseSpec.vol_max['rsv_0'],
                                                   set_rsv[1]: DatabaseSpec.vol_max['rsv_1']}}
    elif mode == 'LMWV':
        mid_term_planning_by_date = {set_date[0]: {set_rsv[0]: 10,
                                                   set_rsv[1]: 5}}

    vol_ini_DA_by_date = {set_date[0]: {set_rsv[0]: DatabaseSpec.vol_max['rsv_0'],
                                        set_rsv[1]: DatabaseSpec.vol_max['rsv_1']}}

    wf_in_delay_from_yesterday_DA_by_date = {date: {rsv: 0 for rsv in set_rsv} for date in set_date}

    # Initialize recorder of RT
    vol_ini_RT_by_datesubhour = {(set_date[0], set_subhour[0]): vol_ini_DA_by_date[set_date[0]]}
    p_cmit_by_datesubhour = {(set_date[0], set_subhour[0]): 100}
    wf_in_delay_from_lasthour_RT_by_datesubhour = {(date, subhour): {rsv: 0 for rsv in set_rsv}
                                                   for date in set_date
                                                   for subhour in set_subhour}

    # Initialize recorder of model and result
    result_DA_by_date = {}
    result_RT_by_datesubhour = {}

    # Get LP files
    model_DA_public, _ = operation_DA_mode(DatabaseSpec,
                                           date_1st,
                                           vol_ini_DA_by_date[date_1st],
                                           mid_term_planning_by_date[date_1st],
                                           wf_in_delay_from_yesterday_DA_by_date[date_1st])

    model_RT_public, _ = operation_RT(DatabaseSpec,
                                      date_1st,
                                      set_subhour[0],
                                      vol_ini_RT_by_datesubhour[date_1st, set_subhour[0]],
                                      p_cmit_by_datesubhour[date_1st, set_subhour[0]],
                                      wf_in_delay_from_lasthour_RT_by_datesubhour[date_1st, set_subhour[0]])

    # Rolling begins
    for date in set_date:
        date_next = (datetime.strptime(date, tf_date) + timedelta(days=1)).strftime(tf_date)
        # Update RHS parameters of DA operations
        for rsv in set_rsv:
            model_DA_public.para_vol_ini[rsv] = vol_ini_DA_by_date[date][rsv]
            model_DA_public.para_mid_term_planning[rsv] = mid_term_planning_by_date[date][rsv]
            model_DA_public.para_wf_in_delay_from_yesterday[rsv] = wf_in_delay_from_yesterday_DA_by_date[date][rsv]
            for time in set_hour:
                if rsv == set_rsv[0]:
                    model_DA_public.para_pre_wi_natural_rate[rsv, time]\
                        = DatabaseSpec.pre_wi_natural_rate.at[f'{date} {time}', 'Discharge/cfs']

        for ver in set_ver:
            for time in set_hour:
                model_DA_public.para_pre_p_ver[ver, time] = DatabaseSpec.pre_p_ver.at[f'{date} {time}', ver]

        for time in set_hour:
            model_DA_public.para_pre_lmp[time] = DatabaseSpec.pre_lmp.at[f'{date} {time}', 'LMP']

        for scen in set_scen:
            for time in set_hour:
                model_DA_public.para_pre_p_ver_scen[scen, time] = DatabaseSpec.p_ver_scen_all.at[f'{scen} {time}', date]

        # Solve DA model
        optimization = solver.solve(model_DA_public, tee=False)
        if optimization.solver.status == 'ok' and optimization.solver.termination_condition == 'optimal':
            print(f'Day-ahead model of {date} has been solved.')
            result_DA_by_date[date] = retrieve_result_DA_x(model_DA_public, is_optimal=True)
        else:
            print(f'Day-ahead model of {date} cannot be solved.')

        # Begin RT intervals
        for subhour in set_subhour:
            hour = datetime.strptime(subhour, tf_subhour).strftime(tf_hour)
            subhour_1h_later = (datetime.strptime(subhour, tf_subhour) + timedelta(hours=1)).strftime(tf_subhour)
            subhour_next = (datetime.strptime(subhour, tf_subhour) + timedelta(minutes=5)).strftime(tf_subhour)
            # Update RHS parameters of RT operations
            for rsv in set_rsv:
                model_RT_public.para_vol_ini[rsv] = vol_ini_RT_by_datesubhour[(date, subhour)][rsv]
                model_RT_public.para_wf_in_delay_from_lasthour[rsv]\
                    = wf_in_delay_from_lasthour_RT_by_datesubhour[(date, subhour)][rsv]
                if rsv == set_rsv[0]:
                    model_RT_public.para_act_wi_natural_rate[rsv]\
                        = DatabaseSpec.act_wi_natural_rate.at[f'{date} {subhour}', 'Discharge/cfs']

            for ver in set_ver:
                model_RT_public.para_act_p_ver[ver] = DatabaseSpec.act_p_ver.at[f'{date} {subhour}', ver]

            model_RT_public.para_p_cmit = result_DA_by_date[date]['var_x_p_cmit'].at[hour, 'MW']
            model_RT_public.para_act_lmp = DatabaseSpec.act_lmp.at[f'{date} {subhour}', 'LMP']
            # Solve RT model
            optimization = solver.solve(model_RT_public, tee=False)
            if optimization.solver.status == 'ok' and optimization.solver.termination_condition == 'optimal':
                print(f'Real-time model of {date} {subhour} has been solved.')
                result_RT_by_datesubhour[(date, subhour)] = retrieve_result_RT_z(model_RT_public, is_optimal=True)
            else:
                print(f'Real-time model of {date} {subhour} cannot be solved.')

            # Update recorder of RT
            if hour != '23:00:00':
                wf_in_delay_from_lasthour_RT_by_datesubhour[(date, subhour_1h_later)][set_rsv[1]]\
                    = result_RT_by_datesubhour[(date, subhour)]['var_z_wf_out'][set_rsv[0]]

                vol_ini_RT_by_datesubhour[(date, subhour_next)]\
                    = result_RT_by_datesubhour[(date, subhour)]['var_z_vol_end']

            elif date != date_end and hour == '23:00:00':
                wf_in_delay_from_lasthour_RT_by_datesubhour[(date_next, subhour_1h_later)][set_rsv[1]]\
                    = result_RT_by_datesubhour[(date, subhour)]['var_z_wf_out'][set_rsv[0]]
                if subhour != '23:55:00':
                    vol_ini_RT_by_datesubhour[(date, subhour_next)]\
                        = result_RT_by_datesubhour[(date, subhour)]['var_z_vol_end']
                elif subhour == '23:55:00':
                    vol_ini_RT_by_datesubhour[(date_next, subhour_next)]\
                        = result_RT_by_datesubhour[(date, subhour)]['var_z_vol_end']

            elif date == date_end and hour == '23:00:00':
                if subhour != '23:55:00':
                    vol_ini_RT_by_datesubhour[(date, subhour_next)]\
                        = result_RT_by_datesubhour[(date, subhour)]['var_z_vol_end']

        # Update recorder of DA after last subhour is done
        if date != date_end and subhour == '23:55:00':
            vol_ini_DA_by_date[date_next] = result_RT_by_datesubhour[(date, subhour)]['var_z_vol_end']
            if mode == 'EODS':
                mid_term_planning_by_date[date_next] = vol_ini_DA_by_date[date_next] # Use vol_ini
            elif mode == 'LMWV':
                mid_term_planning_by_date[date_next] = mid_term_planning_by_date[date] # Use LMWV

            wf_out_from_yesterday_last_hour = [result_RT_by_datesubhour[(date, time)]['var_z_wf_out'][set_rsv[0]]
                                               for time in set_subhour_last_hour]
            wf_in_delay_from_yesterday_DA_by_date[date_next][set_rsv[1]] = sum(wf_out_from_yesterday_last_hour)

    # Organize results
    result_DA = organize_result_DA_x(DatabaseSpec,
                                     result_DA_by_date,
                                     wf_in_delay_from_yesterday_DA_by_date)

    result_RT = organize_result_RT_z(DatabaseSpec,
                                     result_RT_by_datesubhour,
                                     wf_in_delay_from_lasthour_RT_by_datesubhour)

    return model_DA_public, result_DA, model_RT_public, result_RT

########################################################################################################################
def auto_DA_RT_single_day(DatabaseSpec,
                          date_today,
                          model_DA_public,
                          model_RT_public,
                          vol_ini_DA,
                          mid_term_planning,
                          wf_in_delay_from_yesterday_DA_by_date,
                          wf_in_delay_from_lasthour_RT_by_datesubhour):

    # Initialize solver setting
    solver = pyo.SolverFactory('gurobi')
    solver.options['MipGap'] = 0.00

    # Initialize date and time
    tf_date = DatabaseSpec.tf_date
    tf_hour = DatabaseSpec.tf_hour
    tf_subhour = DatabaseSpec.tf_subhour

    # Initialize set
    set_rsv = DatabaseSpec.set_rsv
    set_ver = DatabaseSpec.set_ver
    set_date = pd.date_range(start=date_today, end=date_today).strftime(tf_date).tolist()
    set_hour = DatabaseSpec.set_hour
    set_subhour = DatabaseSpec.set_subhour
    set_scen = DatabaseSpec.set_scen

    # Initialize recorder of DA
    mid_term_planning_by_date = {set_date[0]: mid_term_planning}
    vol_ini_DA_by_date = {set_date[0]: vol_ini_DA}

    # Initialize recorder of RT
    vol_ini_RT_by_datesubhour = {(set_date[0], set_subhour[0]): vol_ini_DA}

    # Initialize recorder of model and result
    result_DA_by_date = {}
    result_RT_by_datesubhour = {}

    # Rolling begins
    for date in set_date:
        # Update RHS parameters of DA operations
        for rsv in set_rsv:
            model_DA_public.para_vol_ini[rsv] = vol_ini_DA_by_date[date][rsv]
            model_DA_public.para_mid_term_planning[rsv] = mid_term_planning_by_date[date][rsv]
            model_DA_public.para_wf_in_delay_from_yesterday[rsv] = wf_in_delay_from_yesterday_DA_by_date[date][rsv]
            for time in set_hour:
                if rsv == set_rsv[0]:
                    model_DA_public.para_pre_wi_natural_rate[rsv, time]\
                        = DatabaseSpec.pre_wi_natural_rate.at[f'{date} {time}', 'Discharge/cfs']

        for ver in set_ver:
            for time in set_hour:
                model_DA_public.para_pre_p_ver[ver, time] = DatabaseSpec.pre_p_ver.at[f'{date} {time}', ver]

        for time in set_hour:
            model_DA_public.para_pre_lmp[time] = DatabaseSpec.pre_lmp.at[f'{date} {time}', 'LMP']

        for scen in set_scen:
            for time in set_hour:
                model_DA_public.para_pre_p_ver_scen[scen, time] = DatabaseSpec.p_ver_scen_all.at[f'{scen} {time}', date]

        # Solve DA model
        optimization = solver.solve(model_DA_public, tee=False)
        if optimization.solver.status == 'ok' and optimization.solver.termination_condition == 'optimal':
            print(f'Day-ahead model of {date} has been solved.')
            result_DA_by_date[date] = retrieve_result_DA_x(model_DA_public, is_optimal=True)
        else:
            print(f'Day-ahead model of {date} cannot be solved.')

        # Begin RT intervals
        for subhour in set_subhour:
            hour = datetime.strptime(subhour, tf_subhour).strftime(tf_hour)
            subhour_1h_later = (datetime.strptime(subhour, tf_subhour) + timedelta(hours=1)).strftime(tf_subhour)
            subhour_next = (datetime.strptime(subhour, tf_subhour) + timedelta(minutes=5)).strftime(tf_subhour)
            # Update RHS parameters of RT operations
            for rsv in set_rsv:
                model_RT_public.para_vol_ini[rsv] = vol_ini_RT_by_datesubhour[(date, subhour)][rsv]
                model_RT_public.para_wf_in_delay_from_lasthour[rsv]\
                    = wf_in_delay_from_lasthour_RT_by_datesubhour[(date, subhour)][rsv]
                if rsv == set_rsv[0]:
                    model_RT_public.para_act_wi_natural_rate[rsv]\
                        = DatabaseSpec.act_wi_natural_rate.at[f'{date} {subhour}', 'Discharge/cfs']

            for ver in set_ver:
                model_RT_public.para_act_p_ver[ver] = DatabaseSpec.act_p_ver.at[f'{date} {subhour}', ver]

            model_RT_public.para_p_cmit = result_DA_by_date[date]['var_x_p_cmit'].at[hour, 'MW']
            model_RT_public.para_act_lmp = DatabaseSpec.act_lmp.at[f'{date} {subhour}', 'LMP']
            # Solve RT model
            optimization = solver.solve(model_RT_public, tee=False)
            if optimization.solver.status == 'ok' and optimization.solver.termination_condition == 'optimal':
                # print(f'Real-time model of {date} {subhour} has been solved.')
                result_RT_by_datesubhour[(date, subhour)] = retrieve_result_RT_z(model_RT_public, is_optimal=True)
            else:
                print(f'Real-time model of {date} {subhour} cannot be solved.')

            # Update recorder of RT
            if hour != '23:00:00':
                wf_in_delay_from_lasthour_RT_by_datesubhour[(date, subhour_1h_later)][set_rsv[1]]\
                    = result_RT_by_datesubhour[(date, subhour)]['var_z_wf_out'][set_rsv[0]]

                vol_ini_RT_by_datesubhour[(date, subhour_next)]\
                    = result_RT_by_datesubhour[(date, subhour)]['var_z_vol_end']

            elif hour == '23:00:00':
                if subhour != '23:55:00':
                    vol_ini_RT_by_datesubhour[(date, subhour_next)]\
                        = result_RT_by_datesubhour[(date, subhour)]['var_z_vol_end']

    # Organize results
    result_DA = organize_result_DA_x(DatabaseSpec,
                                     result_DA_by_date,
                                     wf_in_delay_from_yesterday_DA_by_date)

    result_RT = organize_result_RT_z(DatabaseSpec,
                                     result_RT_by_datesubhour,
                                     wf_in_delay_from_lasthour_RT_by_datesubhour)

    return result_DA, result_RT


########################################################################################################################
def auto_DA_RT_single_day_with_mp(DatabaseSpec,
                                  date_today,
                                  model_DA_public,
                                  vol_ini_DA,
                                  mid_term_planning,
                                  wf_in_delay_from_yesterday_DA_by_date,
                                  wf_in_delay_from_lasthour_RT_by_datesubhour,
                                  cr_result):
    # Initialize solver setting
    solver = pyo.SolverFactory('gurobi')
    solver.options['MipGap'] = 0.00

    # Initialize date and time
    tf_date = DatabaseSpec.tf_date
    tf_hour = DatabaseSpec.tf_hour
    tf_subhour = DatabaseSpec.tf_subhour

    # Initialize set
    set_rsv = DatabaseSpec.set_rsv
    set_ver = DatabaseSpec.set_ver
    set_date = pd.date_range(start=date_today, end=date_today).strftime(tf_date).tolist()
    set_hour = DatabaseSpec.set_hour
    set_subhour = DatabaseSpec.set_subhour
    set_scen = DatabaseSpec.set_scen

    # Initialize recorder of DA
    mid_term_planning_by_date = {set_date[0]: mid_term_planning}
    vol_ini_DA_by_date = {set_date[0]: vol_ini_DA}

    # Initialize recorder of RT
    vol_ini_RT_by_datesubhour = {(set_date[0], set_subhour[0]): vol_ini_DA}

    # Initialize recorder of model and result
    result_DA_by_date = {}
    result_RT_by_datesubhour = {}

    # Rolling begins
    for date in set_date:
        # Update RHS parameters of DA operations
        for rsv in set_rsv:
            model_DA_public.para_vol_ini[rsv] = vol_ini_DA_by_date[date][rsv]
            model_DA_public.para_mid_term_planning[rsv] = mid_term_planning_by_date[date][rsv]
            model_DA_public.para_wf_in_delay_from_yesterday[rsv] = round(wf_in_delay_from_yesterday_DA_by_date[date][rsv], 1)
            if rsv == set_rsv[0]:
                for time in set_hour:
                    model_DA_public.para_pre_wi_natural_rate[rsv, time]\
                        = DatabaseSpec.pre_wi_natural_rate.at[f'{date} {time}', 'Discharge/cfs']

        for ver in set_ver:
            for time in set_hour:
                model_DA_public.para_pre_p_ver[ver, time] = DatabaseSpec.pre_p_ver.at[f'{date} {time}', ver]

        for time in set_hour:
            model_DA_public.para_pre_lmp[time] = DatabaseSpec.pre_lmp.at[f'{date} {time}', 'LMP']

        for scen in set_scen:
            for time in set_hour:
                model_DA_public.para_pre_p_ver_scen[scen, time] = DatabaseSpec.p_ver_scen_all.at[f'{scen} {time}', date]


        # Solve DA model
        optimization = solver.solve(model_DA_public, tee=False)
        if optimization.solver.status == 'ok' and optimization.solver.termination_condition == 'optimal':
            print(f'Day-ahead model of {date} has been solved.')
            result_DA_by_date[date] = retrieve_result_DA_x(model_DA_public, is_optimal=True)
        else:
            print(f'Day-ahead model of {date} cannot be solved.')

        # Begin RT intervals
        for subhour in set_subhour:
            hour = datetime.strptime(subhour, tf_subhour).strftime(tf_hour)
            subhour_1h_later = (datetime.strptime(subhour, tf_subhour) + timedelta(hours=1)).strftime(tf_subhour)
            subhour_next = (datetime.strptime(subhour, tf_subhour) + timedelta(minutes=5)).strftime(tf_subhour)
            # Update RHS parameters of RT operations
            theta_p_cmit = round(result_DA_by_date[date]['var_x_p_cmit'].at[hour, 'MW'], 1)+0.01

            theta_vol_ini_R0 = round(vol_ini_RT_by_datesubhour[(date, subhour)]['rsv_0'], 1)

            theta_vol_ini_R1 = round(vol_ini_RT_by_datesubhour[(date, subhour)]['rsv_1'], 1)

            theta_wf_in_R0 = round(DatabaseSpec.rate_to_vol_RT
                                   * DatabaseSpec.act_wi_natural_rate.at[f'{date} {subhour}', 'Discharge/cfs'], 1)

            theta_wf_in_R1 = round(wf_in_delay_from_lasthour_RT_by_datesubhour[(date, subhour)]['rsv_1'], 1) + 0.1

            theta_p_act_ver = float(round(DatabaseSpec.act_p_ver.at[f'{date} {subhour}', 'wind']
                                    + DatabaseSpec.act_p_ver.at[f'{date} {subhour}', 'solar'], 1))+0.09

            theta_act_lmp = DatabaseSpec.act_lmp.at[f'{date} {subhour}', 'LMP']

            theta = {'para_p_cmit': theta_p_cmit,
                     'para_vol_ini_R0': theta_vol_ini_R0,
                     'para_vol_ini_R1': theta_vol_ini_R1,
                     'para_wf_in_R0': theta_wf_in_R0,
                     'para_wf_in_R1': theta_wf_in_R1,
                     'para_p_act_ver': theta_p_act_ver}

            result_RT_by_datesubhour[(date, subhour)] = operation_RT_by_mp(DatabaseSpec,
                                                                           theta,
                                                                           cr_result,
                                                                           theta_act_lmp)

            # print(f'Real-time model of {date} {subhour} has been solved.')

            # Update recorder of RT
            if hour != '23:00:00':
                wf_in_delay_from_lasthour_RT_by_datesubhour[(date, subhour_1h_later)][set_rsv[1]]\
                    = result_RT_by_datesubhour[(date, subhour)]['var_z_wf_out'][set_rsv[0]]

                vol_ini_RT_by_datesubhour[(date, subhour_next)]\
                    = result_RT_by_datesubhour[(date, subhour)]['var_z_vol_end']

            elif hour == '23:00:00':
                if subhour != '23:55:00':
                    vol_ini_RT_by_datesubhour[(date, subhour_next)]\
                        = result_RT_by_datesubhour[(date, subhour)]['var_z_vol_end']

    # Organize results
    result_DA = organize_result_DA_x(DatabaseSpec,
                                     result_DA_by_date,
                                     wf_in_delay_from_yesterday_DA_by_date)

    result_RT = organize_result_RT_z(DatabaseSpec,
                                     result_RT_by_datesubhour,
                                     wf_in_delay_from_lasthour_RT_by_datesubhour)

    return result_DA, result_RT


def get_cr():
    mat_data = scipy.io.loadmat('CR_Result.mat')['CR_Result']
    cr_result = {}
    for name in mat_data.dtype.names:
        cr_result[name] = mat_data[name][0, 0]

    return cr_result


########################################################################################################################
def generate_LP_file_DA(DatabaseSpec, date_1st):
    mode = DatabaseSpec.mode

    # Initialize solver setting
    solver = pyo.SolverFactory('gurobi')
    solver.options['MipGap'] = 0.00

    # Initialize date and time
    tf_date = DatabaseSpec.tf_date

    # Initialize set
    set_rsv = DatabaseSpec.set_rsv
    set_date = pd.date_range(start=date_1st, end=date_1st).strftime(tf_date).tolist()

    # Initialize recorder of DA
    if mode == 'EODS':
        mid_term_planning_by_date = {set_date[0]: {set_rsv[0]: DatabaseSpec.vol_max['rsv_0'],
                                                   set_rsv[1]: DatabaseSpec.vol_max['rsv_1']}}
    elif mode == 'LMWV':
        mid_term_planning_by_date = {set_date[0]: {set_rsv[0]: 10,
                                                   set_rsv[1]: 5}}

    vol_ini_DA_by_date = {set_date[0]: {set_rsv[0]: DatabaseSpec.vol_max['rsv_0'],
                                        set_rsv[1]: DatabaseSpec.vol_max['rsv_1']}}

    wf_in_delay_from_yesterday_DA_by_date = {date: {rsv: 0 for rsv in set_rsv}
                                             for date in set_date}

    # Initialize LP files
    model_DA_public, _ = operation_DA_mode(DatabaseSpec,
                                           date_1st,
                                           vol_ini_DA_by_date[date_1st],
                                           mid_term_planning_by_date[date_1st],
                                           wf_in_delay_from_yesterday_DA_by_date[date_1st])

    return model_DA_public

def generate_LP_file_RT(DatabaseSpec, date_1st):
    # Initialize solver setting
    solver = pyo.SolverFactory('gurobi')
    solver.options['MipGap'] = 0.00

    # Initialize date and time
    tf_date = DatabaseSpec.tf_date

    # Initialize set
    set_rsv = DatabaseSpec.set_rsv
    set_date = pd.date_range(start=date_1st, end=date_1st).strftime(tf_date).tolist()
    set_subhour = DatabaseSpec.set_subhour

    # Initialize recorder of DA
    vol_ini_DA_by_date = {set_date[0]: {set_rsv[0]: DatabaseSpec.vol_max['rsv_0'],
                                        set_rsv[1]: DatabaseSpec.vol_max['rsv_1']}}

    # Initialize recorder of RT
    vol_ini_RT_by_datesubhour = {(set_date[0], set_subhour[0]): vol_ini_DA_by_date[set_date[0]]}

    p_cmit_by_datesubhour = {(set_date[0], set_subhour[0]): 100}

    wf_in_delay_from_lasthour_RT_by_datesubhour = {(date, subhour): {rsv: 0 for rsv in set_rsv}
                                                   for date in set_date
                                                   for subhour in set_subhour}

    model_RT_public, _ = operation_RT(DatabaseSpec,
                                      date_1st,
                                      set_subhour[0],
                                      vol_ini_RT_by_datesubhour[date_1st, set_subhour[0]],
                                      p_cmit_by_datesubhour[date_1st, set_subhour[0]],
                                      wf_in_delay_from_lasthour_RT_by_datesubhour[date_1st, set_subhour[0]])

    return model_RT_public