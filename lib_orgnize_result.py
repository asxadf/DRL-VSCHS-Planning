import pandas as pd
import pyomo.environ as pyo

dic = globals()

def value_variable_DA_x(model, var_name, index_types):
    var = getattr(model, var_name)
    indices = {'rsv': model.set_rsv,
               'time': model.set_time,
               'unit': model.set_unit, }

    if index_types == ('rsv', 'time'):
        result = pd.DataFrame({rsv: [round(var[rsv, t].value, 2) for t in indices['time']]
                               for rsv in indices['rsv']}, index=indices['time'].ordered_data())

    elif index_types == ('unit', 'time'):
        result = pd.DataFrame({unit: [round(var[unit, t].value, 2) for t in indices['time']]
                               for unit in indices['unit']}, index=indices['time'].ordered_data())

    elif index_types == ('time',):
        result = pd.DataFrame({'MW': [round(var[t].value, 2) for t in indices['time']]
                               }, index=indices['time'].ordered_data())

    elif index_types == ('rsv',):
        result = {rsv: round(var[rsv].value, 2) for rsv in indices['rsv']}

    return result


def retrieve_result_DA_x(model, is_optimal):
    result = {'para_pre_lmp': pd.DataFrame.from_dict(model.para_pre_lmp.extract_values(), orient='index', dtype=float)}

    para_pre_p_res_dict = model.para_pre_p_ver.extract_values()
    result['para_pre_p_ver'] = pd.DataFrame(index=model.set_time, columns=model.set_ver, dtype=float)
    for (col, row), value in para_pre_p_res_dict.items():
        result['para_pre_p_ver'].at[row, col] = value

    para_pre_wi_natural_rate_dict = model.para_pre_wi_natural_rate.extract_values()
    result['para_pre_wi_natural_rate'] = pd.DataFrame(index=model.set_time, columns=model.set_rsv, dtype=float)
    for (col, row), value in para_pre_wi_natural_rate_dict.items():
        result['para_pre_wi_natural_rate'].at[row, col] = value

    if is_optimal:
        for var in model.component_objects(pyo.Var, active=True):
            var_name = var.name
            if var_name in ('var_x_i', 'var_x_p_hyd'):
                result[var_name] = value_variable_DA_x(model, var_name, ('unit', 'time'))

            elif var_name in ('var_x_p_cmit', 'var_x_p_ver', 'var_x_revenue_gross'):
                result[var_name] = value_variable_DA_x(model, var_name, ('time',))
    return result


def organize_result_DA_x(DatabaseSpec, result_DA_by_date, wf_in_delay_from_yesterday_DA_by_date):
    retrieve_list = ['para_pre_lmp', 'para_pre_p_ver', 'para_pre_wi_natural_rate', 'para_wf_in_delay_from_yesterday',
                     'var_x_i', 'var_x_p_hyd',
                     'var_x_p_ver', 'var_x_p_cmit', 'var_x_revenue_gross']

    set_date = list(result_DA_by_date.keys())
    set_hour = pd.date_range(start="00:00:00", end="23:00:00", freq='1h').strftime('%H:00:00').tolist()
    set_datehour = [f'{date} {hour}' for date in set_date for hour in set_hour]
    set_unit = DatabaseSpec.set_unit
    set_rsv = DatabaseSpec.set_rsv
    set_ver = DatabaseSpec.set_ver

    result_DA = {}
    for retrieve_name in retrieve_list:
        result_DA[retrieve_name] = pd.DataFrame(dtype=float)

        if retrieve_name == 'para_pre_lmp':
            for date in set_date:
                result_DA[retrieve_name] = pd.concat([result_DA[retrieve_name], result_DA_by_date[date][retrieve_name]])

            result_DA[retrieve_name].index = set_datehour

        elif retrieve_name == 'para_pre_p_ver':
            for date in set_date:
                result_DA[retrieve_name] = pd.concat([result_DA[retrieve_name], result_DA_by_date[date][retrieve_name]])

            result_DA[retrieve_name].index = set_datehour
            result_DA[retrieve_name].columns = set_ver

        elif retrieve_name == 'para_pre_wi_natural_rate':
            for date in set_date:
                result_DA[retrieve_name] = pd.concat([result_DA[retrieve_name], result_DA_by_date[date][retrieve_name]])

            result_DA[retrieve_name].index = set_datehour
            result_DA[retrieve_name].columns = set_rsv

        elif retrieve_name == 'para_wf_in_delay_from_yesterday':
            for date in set_date:
                for rsv in set_rsv:
                    result_DA[retrieve_name].at[date, rsv] = wf_in_delay_from_yesterday_DA_by_date[date][rsv]

        elif retrieve_name in ('var_x_i', 'var_x_p_hyd'):
            for date in set_date:
                result_DA[retrieve_name] = pd.concat([result_DA[retrieve_name], result_DA_by_date[date][retrieve_name]])

            result_DA[retrieve_name].index = set_datehour
            result_DA[retrieve_name].columns = set_unit

        elif retrieve_name in ('var_x_p_ver', 'var_x_p_cmit', 'var_x_revenue_gross'):
            for date in set_date:
                result_DA[retrieve_name] = pd.concat([result_DA[retrieve_name], result_DA_by_date[date][retrieve_name]])

            result_DA[retrieve_name].index = set_datehour

    return result_DA


def value_variable_RT_z(model, var_name, index_types):
    var = getattr(model, var_name)
    indices = {'rsv': model.set_rsv,
               'unit': model.set_unit}

    # Initialize dictionary or DataFrame to store results
    if index_types == ('rsv',):
        result = {rsv: float(round(var[rsv].value, 4)) for rsv in indices['rsv']}

    elif index_types == ('unit',):
        result = {unit: float(round(var[unit].value, 4)) for unit in indices['unit']}

    elif index_types == ('-',):
        result = float(round(var.value, 4))

    return result


def retrieve_result_RT_z(model, is_optimal):
    result = {'para_act_p_ver': model.para_act_p_ver.extract_values(),
              'para_act_lmp': float(model.para_act_lmp.value),
              'para_act_wi_natural_rate': model.para_act_wi_natural_rate.extract_values(),
              'para_vol_ini': model.para_vol_ini.extract_values()}

    if is_optimal:
        for var in model.component_objects(pyo.Var, active=True):
            var_name = var.name
            if var_name in ('var_z_wf_in', 'var_z_wf_out', 'var_z_wf_spill'):
                result[var_name] = value_variable_RT_z(model, var_name, ('rsv',))

            elif var_name in ('var_z_i', 'var_z_p_hyd', 'var_z_rate'):
                result[var_name] = value_variable_RT_z(model, var_name, ('unit',))

            elif var_name == 'var_z_vol_end':
                result[var_name] = value_variable_RT_z(model, var_name, ('rsv',))

            elif var_name in ('var_z_p_ver', 'var_z_p_act', 'var_z_p_difference'):
                result[var_name] = value_variable_RT_z(model, var_name, ('-',))
    return result


def organize_result_RT_z(DatabaseSpec, result_RT_by_datesubhour, wf_in_delay_from_lasthour_RT_by_datesubhour):
    retrieve_list = ['para_act_lmp', 'para_act_p_ver', 'para_act_wi_natural_rate',
                     'para_wf_in_delay_from_lasthour', 'para_vol_ini',
                     'var_z_i', 'var_z_p_hyd', 'var_z_rate',
                     'var_z_vol_end', 'var_z_wf_in', 'var_z_wf_out', 'var_z_wf_spill',
                     'var_z_p_ver', 'var_z_p_act', 'var_z_p_difference']

    set_datesubhour = [(date, subhour) for date, subhour in result_RT_by_datesubhour.keys()]
    set_unit = DatabaseSpec.set_unit
    set_rsv = DatabaseSpec.set_rsv
    set_ver = DatabaseSpec.set_ver

    result_RT = {}
    for retrieve_name in retrieve_list:
        if retrieve_name == 'para_act_lmp':
            result_RT[retrieve_name] = pd.DataFrame(index=set_datesubhour, columns=['LMP'], dtype=float)
            for datesubhour in set_datesubhour:
                result_RT[retrieve_name].at[datesubhour, 'LMP'] = result_RT_by_datesubhour[datesubhour][retrieve_name]

        elif retrieve_name == 'para_act_p_ver':
            if isinstance(result_RT_by_datesubhour[datesubhour][retrieve_name], dict):
                result_RT[retrieve_name] = pd.DataFrame(index=set_datesubhour, columns=set_ver, dtype=float)
                for datesubhour in set_datesubhour:
                    for res in set_ver:
                        result_RT[retrieve_name].at[datesubhour, res] = \
                            result_RT_by_datesubhour[datesubhour][retrieve_name][res]
            elif isinstance(result_RT_by_datesubhour[datesubhour][retrieve_name], float):
                result_RT[retrieve_name] = pd.DataFrame(index=set_datesubhour, columns=['wind+solar'], dtype=float)
                for datesubhour in set_datesubhour:
                    result_RT[retrieve_name].at[datesubhour, 'wind+solar'] = result_RT_by_datesubhour[datesubhour][
                        retrieve_name]

        elif retrieve_name == 'para_act_wi_natural_rate':
            result_RT[retrieve_name] = pd.DataFrame(index=set_datesubhour, columns=set_rsv, dtype=float)
            for datesubhour in set_datesubhour:
                for rsv in set_rsv:
                    result_RT[retrieve_name].at[datesubhour, rsv] = \
                        result_RT_by_datesubhour[datesubhour][retrieve_name][rsv]

        elif retrieve_name == 'para_wf_in_delay_from_lasthour':
            result_RT[retrieve_name] = pd.DataFrame(index=set_datesubhour, columns=set_rsv, dtype=float)
            for datesubhour in set_datesubhour:
                for rsv in set_rsv:
                    result_RT[retrieve_name].at[datesubhour, rsv] = \
                        wf_in_delay_from_lasthour_RT_by_datesubhour[datesubhour][rsv]

        elif retrieve_name == 'para_vol_ini':
            result_RT[retrieve_name] = pd.DataFrame(index=set_datesubhour, columns=set_rsv, dtype=float)
            for datesubhour in set_datesubhour:
                for rsv in set_rsv:
                    result_RT[retrieve_name].at[datesubhour, rsv] = \
                        result_RT_by_datesubhour[datesubhour][retrieve_name][rsv]

        elif retrieve_name in ('var_z_i', 'var_z_p_hyd', 'var_z_rate'):
            result_RT[retrieve_name] = pd.DataFrame(index=set_datesubhour, columns=set_unit, dtype=float)
            for datesubhour in set_datesubhour:
                for unit in set_unit:
                    result_RT[retrieve_name].at[datesubhour, unit] = \
                        result_RT_by_datesubhour[datesubhour][retrieve_name][unit]

        elif retrieve_name == 'var_z_vol_end':
            result_RT[retrieve_name] = pd.DataFrame(index=set_datesubhour, columns=set_rsv, dtype=float)
            for datesubhour in set_datesubhour:
                for rsv in set_rsv:
                    result_RT[retrieve_name].at[datesubhour, rsv] = \
                        result_RT_by_datesubhour[datesubhour][retrieve_name][rsv]

        elif retrieve_name in ('var_z_wf_in', 'var_z_wf_out', 'var_z_wf_spill'):
            result_RT[retrieve_name] = pd.DataFrame(index=set_datesubhour, columns=set_rsv, dtype=float)
            for datesubhour in set_datesubhour:
                for rsv in set_rsv:
                    result_RT[retrieve_name].at[datesubhour, rsv] = \
                        result_RT_by_datesubhour[datesubhour][retrieve_name][rsv]

        elif retrieve_name in ('var_z_p_ver', 'var_z_p_act', 'var_z_p_difference'):
            result_RT[retrieve_name] = pd.DataFrame(index=set_datesubhour, columns=['MW'], dtype=float)
            for datesubhour in set_datesubhour:
                result_RT[retrieve_name].at[datesubhour, 'MW'] = result_RT_by_datesubhour[datesubhour][retrieve_name]

    return result_RT