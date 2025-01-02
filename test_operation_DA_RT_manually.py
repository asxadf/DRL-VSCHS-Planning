from data.database import DatabaseClass
from lib_operation_model import operation_DA_mode, operation_RT, organize_result_RT_z, organize_result_DA_x
from datetime import datetime, timedelta
import pandas as pd

# Initialize DatabaseSpec
DatabaseSpec = DatabaseClass()
tf_date = DatabaseSpec.tf_date
tf_hour = DatabaseSpec.tf_hour
tf_subhour = DatabaseSpec.tf_subhour
set_rsv = DatabaseSpec.set_rsv
set_subhour = DatabaseSpec.set_subhour
set_subhour_last_hour = DatabaseSpec.set_subhour_last_hour

date_1st = '2022-02-01'
date_end = '2022-02-03'
set_date = pd.date_range(start=date_1st, end=date_end).strftime(tf_date).tolist()

# Input of DA
mid_term_planning_by_date = {set_date[0]: {'rsv_0': 138260, 'rsv_1': 1494}} # EODS
# mid_term_planning_by_date = {set_date[0]: {'rsv_0': 5, 'rsv_1': 1}} # LMWV
vol_ini_DA_by_date = {set_date[0]: {'rsv_0': 138260, 'rsv_1': 1494}}
wf_in_delay_from_yesterday_DA_by_date = {date: {rsv: 0 for rsv in set_rsv} for date in set_date}

# Input of RT
vol_ini_RT_by_datesubhour = {(set_date[0], set_subhour[0]): vol_ini_DA_by_date[set_date[0]]}
p_cmit_by_datesubhour = {}
wf_in_delay_from_lasthour_RT_by_datesubhour = {(date, subhour): {rsv: 0 for rsv in set_rsv} for date in set_date for subhour in set_subhour}

# Recorder of model and result
model_DA_by_date = {}
result_DA_by_date = {}
model_RT_by_datesubhour = {}
result_RT_by_datesubhour = {}

for date in set_date:
    date_next = (datetime.strptime(date, tf_date) + timedelta(days=1)).strftime(tf_date)
    model_DA_by_date[date], result_DA_by_date[date] = operation_DA_mode(DatabaseSpec,
                                                                        date,
                                                                        vol_ini_DA_by_date[date],
                                                                        mid_term_planning_by_date[date],
                                                                        wf_in_delay_from_yesterday_DA_by_date[date])
    for subhour in set_subhour:
        hour = datetime.strptime(subhour, tf_subhour).strftime(tf_hour)
        subhour_1h_later = (datetime.strptime(subhour, tf_subhour) + timedelta(hours=1)).strftime(tf_subhour)
        subhour_next =  (datetime.strptime(subhour, tf_subhour) + timedelta(minutes=5)).strftime(tf_subhour)
        p_cmit_by_datesubhour[(date, subhour)] = result_DA_by_date[date]['var_x_p_cmit'].at[hour, 'MW']

        model_RT_by_datesubhour[(date, subhour)], result_RT_by_datesubhour[(date, subhour)]\
            = operation_RT(DatabaseSpec,
                           date,
                           subhour,
                           vol_ini_RT_by_datesubhour[(date, subhour)],
                           p_cmit_by_datesubhour[(date, subhour)],
                           wf_in_delay_from_lasthour_RT_by_datesubhour[(date, subhour)])

        # Update recorder of RT
        if hour != '23:00:00':
            wf_in_delay_from_lasthour_RT_by_datesubhour[(date, subhour_1h_later)][set_rsv[1]] = \
                result_RT_by_datesubhour[(date, subhour)]['var_z_wf_out'][set_rsv[0]]
            vol_ini_RT_by_datesubhour[(date, subhour_next)] = result_RT_by_datesubhour[(date, subhour)]['var_z_vol_end']

        elif date != date_end and hour == '23:00:00':
            wf_in_delay_from_lasthour_RT_by_datesubhour[(date_next, subhour_1h_later)][set_rsv[1]] = \
                result_RT_by_datesubhour[(date, subhour)]['var_z_wf_out'][set_rsv[0]]
            if subhour != '23:55:00':
                vol_ini_RT_by_datesubhour[(date, subhour_next)] = result_RT_by_datesubhour[(date, subhour)]['var_z_vol_end']
            elif subhour == '23:55:00':
                vol_ini_RT_by_datesubhour[(date_next, subhour_next)] = result_RT_by_datesubhour[(date, subhour)]['var_z_vol_end']

        elif date == date_end and hour == '23:00:00':
            if subhour != '23:55:00':
                vol_ini_RT_by_datesubhour[(date, subhour_next)] = result_RT_by_datesubhour[(date, subhour)]['var_z_vol_end']

    # Update recorder of DA after last subhour is done
    if date != date_end and subhour == '23:55:00':
        vol_ini_DA_by_date[date_next] = result_RT_by_datesubhour[(date, subhour)]['var_z_vol_end']
        mid_term_planning_by_date[date_next] = mid_term_planning_by_date[date]

        wf_out_from_yesterday_last_hour = [result_RT_by_datesubhour[(date, time)]['var_z_wf_out'][set_rsv[0]]
                                           for time in set_subhour_last_hour]
        wf_in_delay_from_yesterday_DA_by_date[date_next][set_rsv[1]] = sum(wf_out_from_yesterday_last_hour)

# Organize results
result_DA = organize_result_DA_x(DatabaseSpec, result_DA_by_date, wf_in_delay_from_yesterday_DA_by_date)
result_RT = organize_result_RT_z(DatabaseSpec, result_RT_by_datesubhour, wf_in_delay_from_lasthour_RT_by_datesubhour)