from data.database import DatabaseClass
from lib_operation_model import auto_DA_RT_multiple_day

# Initialize
DatabaseSpec = DatabaseClass()

# Initialize date and time
date_1st = '2022-01-01'
date_end = '2022-01-04'

model_DA_public, result_DA, model_RT_public, result_RT = auto_DA_RT_multiple_day(DatabaseSpec, date_1st, date_end)