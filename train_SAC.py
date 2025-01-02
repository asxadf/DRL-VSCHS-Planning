import numpy as np
import pandas as pd
from data.database import DatabaseClass
from learning.env import OperationEnvClass
from operation_model.lib_operation_model import generate_LP_file_DA, generate_LP_file_RT, auto_DA_RT_single_day
from stable_baselines3 import SAC
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from stable_baselines3.common.callbacks import BaseCallback

DRL = 'SAC'
LR = 0.001
date_1st_train = '2021-10-01'
date_end_train = '2021-10-05'
num_round = 100

DatabaseSpec = DatabaseClass()
Mode = DatabaseSpec.mode
set_date_train = pd.date_range(start=date_1st_train, end=date_end_train).strftime(DatabaseSpec.tf_date).tolist()
max_episode_length = len(set_date_train)
env = OperationEnvClass(DatabaseSpec, date_1st_train, date_end_train)

# Train
# tensorboard --logdir=C:\Users\10481\PycharmProjects\DOE_CHS_ST_Operation\version_1\learning\training_info\EODS
# tensorboard --logdir=C:\Users\10481\PycharmProjects\DOE_CHS_ST_Operation\version_1\learning\training_info\LMWV
model = SAC(policy="MultiInputPolicy",
            env=env,
            learning_rate=LR,
            buffer_size=max_episode_length*num_round*10,
            learning_starts=len(set_date_train),
            batch_size=64,
            train_freq=(1, "episode"),
            ent_coef='auto',
            tensorboard_log=f"./training_info/{Mode}")

model.learn(total_timesteps=len(set_date_train)*num_round,
            progress_bar=True)

model.save(f"{DRL}_{Mode}_{str(LR).replace('0.', '')}_{date_1st_train}_{date_end_train}.zip")
#model = SAC.load(f"{DRL}_{Mode}_{str(LR).replace('0.', '')}.zip")



# Define testing period
# Add one year using relativedelta
date_1st_train_plus_year = datetime.strptime(date_1st_train, '%Y-%m-%d') + relativedelta(years=1)
date_end_train_plus_year = datetime.strptime(date_end_train, '%Y-%m-%d') + relativedelta(years=1)

# Convert back to string if needed
date_1st_test = date_1st_train_plus_year.strftime('%Y-%m-%d')
date_end_test = date_end_train_plus_year.strftime('%Y-%m-%d')
set_date_test = pd.date_range(start=date_1st_test, end=date_end_test).strftime(DatabaseSpec.tf_date).tolist()
#
# # Initialize
model_DA_public = generate_LP_file_DA(DatabaseSpec, set_date_test[0])
model_RT_public = generate_LP_file_RT(DatabaseSpec, set_date_test[0])
rec_wf_in_delay_from_yesterday_DA_by_date = {date: {rsv: 0 for rsv in DatabaseSpec.set_rsv}
                                             for date in set_date_test}
#
rec_wf_in_delay_from_lasthour_RT_by_datesubhour = {(date, subhour): {rsv: 0 for rsv in DatabaseSpec.set_rsv}
                                                   for date in set_date_test
                                                   for subhour in DatabaseSpec.set_subhour}

# Reset the environment for testing
env = OperationEnvClass(DatabaseSpec, date_1st_test, date_end_test)
rec_state = {}
rec_state[set_date_test[0]], info = env.reset()

rec_vol_ini_DA = {}
mid_term_planning = {}
rec_result_DA = {}
rec_result_RT = {}
#
rec_revenue_gross = {}
rec_charge_imb = {}
rec_revenue_net = {}

for date in set_date_test:
    day_in_year_org = datetime.strptime(date, DatabaseSpec.tf_date).timetuple().tm_yday
    print('Testing...')
    # Update information
    if date is not set_date_test[0]:
        rec_wf_in_delay_from_yesterday_DA_by_date[date][DatabaseSpec.set_rsv[1]]\
            = temp_wf_in_delay_from_yesterday_DA_by_date.at[0, DatabaseSpec.set_rsv[1]]

        for subhour_first_hour in DatabaseSpec.set_subhour_first_hour:
            rec_wf_in_delay_from_lasthour_RT_by_datesubhour[date, subhour_first_hour][DatabaseSpec.set_rsv[1]]\
                = temp_wf_in_delay_from_lasthour_RT_by_datesubhour.at[subhour_first_hour, DatabaseSpec.set_rsv[1]]

        rec_state[date] = {}
        rec_state[date]['vol_ini_DA'] = env.normalize(temp_vol_ini_DA.to_numpy().flatten(), 'vol')
        rec_state[date]['pre_wi_natural_rate_avg'] = env.normalize(np.array([np.mean(DatabaseSpec.pre_wi_natural_rate.loc[date])]), 'rate')
        rec_state[date]['pre_p_ver_avg'] = env.normalize(np.array(np.mean(DatabaseSpec.pre_p_ver.loc[date], axis=0)), 'ver')
        rec_state[date]['pre_lmp_avg'] = env.normalize(np.array([np.mean(DatabaseSpec.pre_lmp.loc[date])]), 'lmp')
        rec_state[date]['day_in_year'] = env.normalize(np.array([day_in_year_org]), 'year')
        rec_state[date]['sys_vol_req_daily_ub'] = np.array([DatabaseSpec.sys_vol_req_daily.loc[day_in_year_org, 'UB']])
        rec_state[date]['sys_vol_req_daily_lb'] = np.array([DatabaseSpec.sys_vol_req_daily.loc[day_in_year_org, 'LB']])

        # Prescribe the end-of-day storage
    action, _ = model.predict(rec_state[date])

    temp_vol_ini_DA = env.normalize_reverse(rec_state[date]['vol_ini_DA'], 'vol')
    rec_vol_ini_DA[date] = {rsv: temp_vol_ini_DA[i] for i, rsv in enumerate(DatabaseSpec.set_rsv)}

    if DatabaseSpec.mode == 'EODS':
        # Manually modify action
        rsv_at_V_max = temp_vol_ini_DA == env.vol_max_array
        rsv_at_V_min = temp_vol_ini_DA == env.vol_min_array
        temp_vol_end_exp_DA = env.de_normalize_V_end(action, temp_vol_ini_DA)
        mid_term_planning[date] = {rsv: temp_vol_end_exp_DA[i] for i, rsv in enumerate(DatabaseSpec.set_rsv)}
    elif DatabaseSpec.mode == 'LMWV':
        mid_term_planning[date] = {rsv: action[i] for i, rsv in enumerate(DatabaseSpec.set_rsv)}

    # One-day operation
    rec_result_DA[date], rec_result_RT[date] = auto_DA_RT_single_day(DatabaseSpec,
                                                                     date,
                                                                     model_DA_public,
                                                                     model_RT_public, rec_vol_ini_DA[date],
                                                                     mid_term_planning[date],
                                                                     rec_wf_in_delay_from_yesterday_DA_by_date,
                                                                     rec_wf_in_delay_from_lasthour_RT_by_datesubhour)
                                                                    # Pass the address and update internally

    vol_end_act = (rec_result_RT[date]['var_z_vol_end'].iloc[-1]).values

    print(f'vol_ini is {np.round(temp_vol_ini_DA)}; '
          f'mid_term_planning ({DatabaseSpec.mode}) is {np.round(np.array(list(mid_term_planning[date].values())), 2)}; '
          f'vol_end_act is {np.round(vol_end_act)}; ')

    temp_vol_ini_DA = rec_result_RT[date]['var_z_vol_end'].iloc[[-1]].reset_index(drop=True)

    temp_array = np.array([0, sum(rec_result_RT[date]['var_z_wf_out'].loc[(date, DatabaseSpec.set_subhour_last_hour[0]):
                                                                          (date, DatabaseSpec.set_subhour_last_hour[-1]),
                                  DatabaseSpec.set_rsv[0]])])
    temp_wf_in_delay_from_yesterday_DA_by_date = pd.DataFrame([temp_array],
                                                              columns=DatabaseSpec.set_rsv)

    temp_wf_in_delay_from_lasthour_RT_by_datesubhour = pd.DataFrame(np.zeros((len(DatabaseSpec.set_subhour_first_hour),
                                                                              len(DatabaseSpec.set_rsv))),
                                                                    index=DatabaseSpec.set_subhour_first_hour,
                                                                    columns=DatabaseSpec.set_rsv)

    for t, subhour_first_hour in enumerate(DatabaseSpec.set_subhour_first_hour):
        subhour_last_hour = DatabaseSpec.set_subhour_last_hour[t]
        temp_wf_in_delay_from_lasthour_RT_by_datesubhour.loc[subhour_first_hour, DatabaseSpec.set_rsv[1]]\
            = rec_result_RT[date]['var_z_wf_out'].at[(date, subhour_last_hour), DatabaseSpec.set_rsv[0]]

    # Calculate the reward
    lmp_DA = rec_result_DA[date]['para_pre_lmp']
    x_p_cmit = rec_result_DA[date]['var_x_p_cmit']

    lmp_RT = rec_result_RT[date]['para_act_lmp']
    z_p_act = rec_result_RT[date]['var_z_p_act']
    z_p_difference = rec_result_RT[date]['var_z_p_difference']

    rec_revenue_gross[date] = (lmp_DA.iloc[:, 0] * x_p_cmit.iloc[:, 0]).sum()
    rec_charge_imb[date] = 10*(lmp_RT.iloc[:, 0] * z_p_difference.iloc[:, 0]).sum() * (1 / 12)

    rec_revenue_net[date] = rec_revenue_gross[date] - rec_charge_imb[date]
    print(
        f'Gross revenue: {np.round(rec_revenue_gross[date])}; '
        f'Imbalance charge: {np.round(rec_charge_imb[date])}; '
        f'Net revenue: {np.round(rec_revenue_net[date])}')


#rec_revenue_net = pd.DataFrame(rec_revenue_net.values(), index=rec_revenue_net.keys())
#rec_revenue_net.to_csv('rec_revenue_net.csv')

# vol_ini_act = pd.DataFrame(rec_vol_ini_DA.values(), index=rec_vol_ini_DA.keys())
# vol_ini_act.to_csv('vol_ini_act.csv')

#mid_term_planning = pd.DataFrame(mid_term_planning.values(), index=mid_term_planning.keys())
#mid_term_planning.to_csv('mid_term_planning.csv')
# sum(rec_revenue_net.values())
##########################################################################################

import shap
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Helvetica'

feature_cols = [
    'day_in_year',
    'pre_lmp_avg',
    'pre_p_ver_avg_wind',
    'pre_p_ver_avg_soalr',
    'pre_wi_natural_rate_avg',
    'sys_vol_req_daily_lb',
    'sys_vol_req_daily_ub',
    'vol_ini_DA_RB',
    'vol_ini_DA_PT', ]

data_rows = []
# Process each item in rec_state
for key, values in rec_state.items():
    # Flatten 'pre_p_ver_avg' and 'vol_ini_DA' to individual columns
    row = {
        'day_in_year': values['day_in_year'][0],
        'pre_lmp_avg': values['pre_lmp_avg'][0],
        'pre_p_ver_avg_wind': values['pre_p_ver_avg'][0],
        'pre_p_ver_avg_solar': values['pre_p_ver_avg'][1],
        'pre_wi_natural_rate_avg': values['pre_wi_natural_rate_avg'][0],
        'sys_vol_req_daily_lb': values['sys_vol_req_daily_lb'][0],
        'sys_vol_req_daily_ub': values['sys_vol_req_daily_ub'][0],
        'vol_ini_DA_RB': values['vol_ini_DA'][0],
        'vol_ini_DA_PT': values['vol_ini_DA'][1]
    }
    data_rows.append(row)

# Convert list of rows to DataFrame
rec_state_df = pd.DataFrame(data_rows)
rec_state_array = rec_state_df.values

def sac_predict(states_array):
    states_list = [
        {
            'day_in_year': np.array([row[0]]),
            'pre_lmp_avg': np.array([row[1]]),
            'pre_p_ver_avg': np.array([row[2], row[3]]),
            'pre_wi_natural_rate_avg': np.array([row[4]]),
            'sys_vol_req_daily_lb': np.array([row[5]]),
            'sys_vol_req_daily_ub': np.array([row[6]]),
            'vol_ini_DA': np.array([row[7], row[8]])
        }
        for row in states_array
    ]

    # Loop over each state in states_dict
    action_list = []
    for state in states_list:
        # Use model.predict() to get the action for the current state
        action, _ = model.predict(state)
        # Append the action as a list (flattened) to keep a consistent structure for the DataFrame
        action_list.append(action.flatten())

    # Convert the list of actions to a DataFrame with appropriate column names
    action_df = pd.DataFrame(action_list, columns=['vol_end_exp_RB', 'vol_end_exp_PT'])
    return action_df

explainer = shap.KernelExplainer(sac_predict, rec_state_df, feature_cols)
shap_values = explainer.shap_values(rec_state_df, nsamples=5000)

for i, action in enumerate(['vol_end_exp_RB', 'vol_end_exp_PT']):
    shap.summary_plot(shap_values[i], rec_state_df, feature_names=feature_cols, show=False)
    plt.title(f"SHAP Summary for {action}")
    plt.savefig("example_plot.pdf", format="pdf", dpi=300)  # Specify format explicitly
    plt.show()

