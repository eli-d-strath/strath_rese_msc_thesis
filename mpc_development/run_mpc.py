"""
This is intended to execute the MPC control script if selected. 

It is written distinctly from the main.py script so as to avoid breaking the operation
of the RBC controls. 
"""

from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic('reset','-sf')

#! update path for specific project
import sys
sys.path.append('C:/Users/user/msc_dissertation')
import numpy as np
import pandas as pd
import math
import os
import warnings
from tqdm import tqdm
from tank_models import dhw_tank
from tank_models import sh_tank
from sim_tools import hp_cop
from sim_tools import temp_dhw
from sim_tools import temp_sh
from sim_tools import pv_check
from sim_tools import wind_check
from sim_tools import temp_day_counter
from sim_tools import sh_activation_status
from sim_tools import night_setback
from sim_tools import carbon_emissions
from sim_tools import electricity_price
from result_tools import store_timestep_results
from result_tools import store_node_temps
from result_tools import calculate_kpis
from result_tools import export_results
from advanced_rbc import AdvancedRBC
from baseline_rbc import BaselineRBC
from opp_pv_rbc import OppPVRBC
from mpc import MPCScheduler
import logging
from gekko import GEKKO
from progressbar import Bar, ETA, Percentage, ProgressBar, RotatingMarker
import matplotlib.pyplot as plt
import time

LOG = logging.getLogger(__name__)

"""Simulation Characteristics and Dataset Import & Processing"""
#%% Simulation Input Settings
tariff_options = ["standard","daynight","windtariff"]
control_options = ["baseline_rbc","opp_pv_rbc","advanced_rbc","mpc"]  
tariff = tariff_options[0] 
control = control_options[3]
month = 1                 # start month
doy = 1               # start day of year 
doyt = np.array([0,31,59,90,120,151,181,212,243,273,304,334,366])
days_simulated = 1    # days to simulate - normally 365
ts_length = 0.5         # timestep length [h]
horizon = 10          # [timesteps] prediction horizon for MPC - normally 336 for a week

# start, end, length timesteps calculated
if doy+days_simulated > 366:
    days_simulated = 366-doy
    warnings.warn("Specified simulation period goes beyond available data. \nSimulation period adjusted to stay within available data.")

ts_start = math.floor(doy*(24/ts_length)-(24/ts_length))
ts_end = math.floor(ts_start + days_simulated*24/ts_length)
ts_num = ts_end - ts_start

#%% Possible Control Setpoints (assigned to timesteps later)
#! set max_temp to always be at the high limit 
#! set min_temp to always be at low limit (45)
#! eliminate lim_T
dhw_min_T_set = 46      # Minimum DHW tank node[3] temperature before forcing a charge
# dhw_lim_T_set = 50      # Minimum DHW tank top sensor temperature before allowing a charge
dhw_max_T_set = 50.5    # DHW tank 4th node sensor temperature at which charging stops
dhw_max_T_ren = 52      # DHW new max top node temperature when renewably charged

sh_min_T_set = 50       # Minimum SH tank top sensor temperature before forcing a charge
# sh_lim_T_set = 55       # Minimum SH tank top sensor temperature before allowing a charge
sh_max_T_set = 56       # SH tank top sensor temperature at which charging stops
sh_max_T_ren = 57.      # SH new max top node temperature when renewably charged

night_offset_dhw = 3    # decrease in allowable minimum temperature during setback period [C]
night_offset_sh = 5     # decrease in allowable minimum temperature during setback period [C]


pt = 2.  # [kW]
wt = 5. # [kW]
pv_thresholds = np.array([pt,pt,pt,pt,pt,pt,pt,pt,pt,pt,pt,pt])  # PV forced charging threshold for every month [kW]
wind_thresholds = np.array([wt,wt,wt,wt,wt,wt,wt,wt,wt,wt,wt,wt])   # Wind surplus charging threshold for every month [kW]

cp = 4.181 #[kJ/kg-K]

cyc_dhw = 0 
cyc_sh = 0

#%% Import Available Datasets
init_temps_dhw = np.array([50.5,50.4,50.2,50.,22.]) # Starting DHW tank temperature
init_temps_sh = np.array([54.,53.,52.,51.,50.]) # Starting SH tank temperature
water_T_data = np.array([9.3,8.7,8.4,9.1,10.2,11.8,13.5,14.1,13.9,13.1,12.0,10.6]) # Heat Pump Source Temperature (Water)

# Import half-hourly datasets from Emoncms
vol_dhw_data = np.loadtxt('c:/Users/user/msc_dissertation/site_data/WS_DWH_Act_Flow_Liters_half-hour_2023.csv', delimiter=',')    # [L] HW heat supply to WS 
vol_sh_data = np.loadtxt('c:/Users/user/msc_dissertation/site_data/WS_SH-Flow_Act__Liters_half-hour_2023.csv', delimiter=',')    # [L] SH heat supply to WS
pv_E_data = np.loadtxt('c:/Users/user/msc_dissertation/site_data/WS_PV_Act_Energy-kWh_half-hour_2023.csv', delimiter=',')  # [kWh] PV energy generation at WS
# pv_P_data = np.loadtxt('c:/Users/user/msc_dissertation/site_data/WS_PV_Act_Power-Average_kW_half-hour_2023.csv', delimiter=',')  # [kW] PV power generation at WS
CB001_P_data = np.loadtxt('c:/Users/user/msc_dissertation/site_data/FH_CB001_Act_Power_kW_half-hour_2023_With-missing-Data-from-2024.csv', delimiter=',')  # [kW] wind power generation
CB002_P_data = np.loadtxt('c:/Users/user/msc_dissertation/site_data/FH_CB002_Act_Power_kW_half-hour_2023_With-missing-Data-from-2024.csv', delimiter=',')  # [kW] power consumption at main Findhorn site
WW_SS_P_data = np.loadtxt('c:/Users/user/msc_dissertation/site_data/FH_WW-SS_Power_Act_kW_half-hour_2023_Heavily_Sythesized.csv', delimiter=',')  # [kW] power consumption at WW-SS
amb_T_data = np.loadtxt('c:/Users/user/msc_dissertation/site_data/kinloss_daily_av_temp_2023.csv', delimiter=',')    # [C] daily average ambient temperature

# # state of charge = maximum energy that can be discharged by each tank
# soc_dhw = np.zeros(ts_num+1)
# soc_dhw[0] = dhw_tank.HotWaterTank().max_energy_in_out( # DHW State of charge [kWh] based on 45C minimum
#     'discharging', init_temps_dhw,
#     0., dhw_flow_T, dhw_return_T, 0, 30)

# soc_sh = np.zeros(ts_num+1)
# soc_sh[0] = sh_tank.HotWaterTank().max_energy_in_out( # SH State of charge [kWh] based on 45C minimum
#     'discharging', init_temps_sh,
#     0., sh_flow_T, sh_return_T, 0, 30)

# soc_dhw[t+1] = dhw_tank.HotWaterTank().max_energy_in_out(
#         'discharging', dhw_tank_T,
#         0., 45., 10., 0, 30)

#%% Initialise Output Variables and Arrays
tot_elec_ch = 0.                # total electrical charge [kWhe]
tot_therm_ch = 0.               # total thermal charge [kWhth]
tot_therm_disch = 0.            # total thermal discharge to meet demands [kWhth]
tot_dhw_elec_ch = 0.            # total DHW charge [kWhe]
tot_dhw_therm_ch = 0.           # total DHW charge [kWhth]
tot_dhw_therm_disch = 0.        # total DHW discharge to meet demands [kWhth]
tot_sh_elec_ch = 0.             # total SH charge [kWhe]
tot_sh_therm_ch = 0.            # total SH charge [kWhth]
tot_sh_therm_disch = 0.         # total SH discharge to meet demands [kWhth]
tot_pv = 0.                     # total PV electricity used for charging [kWhe]
tot_wind = 0.                   # total wind electricity used for charging [kWhe]
tot_grid = 0.                   # total grid electricity used for charging [kWhe]
tot_cost = 0.                   # total electricity cost for charging [£]

TimestepRes = np.zeros((ts_num,18))
Temps = np.zeros((ts_num,18))
KPIs = np.zeros((1,20))

# create results folder if needed
results_path = 'C:/Users/user/msc_dissertation/simulation_results'
os.makedirs(results_path,exist_ok=True)
outdir = '../simulation_results/'

# create titles for output results files
base_file_name = control + "_" + tariff + "_"
results_file = base_file_name + "results.csv"
kpis_file = base_file_name + "kpis.csv"
temps_file = base_file_name + "node_temps.csv"


#%% New MPC Run Area
#! first, need to meet discharging needs..
#! then need to run MPC prediction and optimization for this timestep
#! then to pass on to next timestep!

t0 = time.time()
# LOG.info('Running model predictive controller...')

print("Initializing MPCScheduler...")
myMPC = MPCScheduler(horizon, ts_num, ts_length, 
                     tariff, doyt,
                     pv_thresholds, wind_thresholds,
                     init_temps_dhw, init_temps_sh,
                     vol_dhw_data, vol_sh_data,
                     pv_E_data,
                     CB001_P_data, CB002_P_data,
                     WW_SS_P_data, amb_T_data,
                     water_T_data)

print("Running Precalculation...")
pre_calc = myMPC.pre_calculation(doy, month, ts_start, ts_num)

print("Simulating with moving horizon optimization...\n")
solutions = myMPC.moving_horizon(pre_calc, ts_start, ts_num)

t1 = time.time()
tot_time = (t1 - t0) /60
print(f'Simulation complete. Time taken: {round(tot_time, 2)} minutes\n')

# print("Printing Solutions")
# print(solutions)
# LOG.info(f'Simulation complete. Time taken: {round(tot_time, 2)} minutes')

#! need to output the results here somehow... adapt from other script?

#%% Results Processing and Exporting

# print("Trying to access results...")
# print(solutions[0]['RES']['RES_total'])

#! these are the values that must be accounted for in the output
# elec_demand = {'elec_charge': 0.0,      # [kwhe] electricity to run HP
#                        'pv_charge': 0.0,        # [kWhe] PV electricity to run HP
#                        'wind_charge': 0.0,      # [kWhe] wind electricity to run HP
#                        'grid_charge': 0.0,      # [kWhe] grid electricity to run HP
#                        }

# heat_demand = {'heat_dem_dhw': 0.0,  # [kWhth] DHW discharge heat demand
#                        'heat_dem_sh': 0.0,   # [kWhth] SH discharge heat demand
#                        'vol_dem_dhw': 0.0,      # [L] DHW discharge volume demand
#                        'vol_dem_sh': 0.0,       # [L] SH discharge volume demand
#                        }

# RES = {'PV': 0.0,                       # [kWhe] PV generation
#                'wind': 0.0,                     # [kWhe] wind generation
#                'RES_total': 0.0,                # [kWhe] total RES generation
#                'HP': 0.0,                       # [kWhe] RES used by HP
#                }

# HP = {'cop_dhw': 0.0,                   # COP if charging DHW tank
#               'cop_sh': 0.0,                    # COP if charging SH tank
#               'duty_dhw': 0.0,                  # [kWhth] duty if charging DHW tank
#               'duty_sh': 0.0,                   # [kWhth] duty if charging SH tank
#               'heat_to_dhw': 0.0,               # [kWhth] heat energy charged to DHW
#               'heat_to_sh': 0.0,                # [kWhth] heat energy charged to SH
#               'elec_usage_dhw': 0.0,            # [kWhe] electricity used to charge DHW
#               'elec_usage_sh': 0.0,             # [kWhe] electricity used to charge SH
#               'pv_usage_dhw': 0.0,              # [kWhe] PV electricity used to charge DHW
#               'pv_usage_sh': 0.0,               # [kWhe] PV electricity used to charge SH
#               'wind_usage_dhw': 0.0,            # [kWhe] wind electricity used to charge DHW
#               'wind_usage_sh': 0.0,             # [kWhe] wind electricity used to charge SH
#               'grid_usage_dhw': 0.0,            # [kWhe] grid electricity used to charge DHW
#               'grid_usage_sh': 0.0,             # [kWhe] grid electricity used to charge SH
#               }

# TS = {'charging_en_dhw': 0.0,           # [kWhth] energy charged into DHW tank
#               'charging_en_sh': 0.0,            # [kWhth] energy charged into SH tank
#               'discharging_en_dhw': 0.0,        # [kWhth] energy discharged from DHW tank
#               'discharging_en_sh': 0.0,         # [kWhth] energy discharged from SH tank
#               'HP_to_dhw': 0.0,                 # [kWhth] energy from HP to DHW tank 
#               'HP_to_sh': 0.0,                  # [kWhth] energy from HP to SH tank
#               'final_nodes_temp_dhw': 0.0,      # [degC] final temps for DHW tank nodes
#               'final_nodes_temp_sh': 0.0,       # [degC] final temps for SH tank nodes
#               }
# outputs = {'elec_demand': elec_demand,
#                    'heat_demand': heat_demand,
#                    'RES': RES,
#                    'HP': HP,
#                    'TS': TS
#                    }







#%% Timestep Model
    
    


#     charges_dhw[ts] = charge_dhw
#     charges_sh[ts] = charge_sh

#     elec_used = (charge_dhw / cop_dhw[ts]) + (charge_sh / cop_sh[ts])   # [kWh]
#     pv_used = min(pv_en[ts],elec_used)                             # [kWh]
#     pv_unused = pv_en[ts] - pv_used                                # [kWh]
#     pv_cost = pv_prices[ts]*pv_used         # [£]
#     wind_used = min(wind_en[ts], elec_used - pv_used)              # [kWh]
#     # print(f"Wind energy in timestep {ts} is {wind_en[ts]} kWh")

#     wind_unused = wind_en[ts] - wind_used                          # [kWh]
#     wind_cost = wind_prices[ts]*wind_used   # [£]
#     grid_used = max(0,elec_used-(pv_used+wind_used))            # [kWh]
#     grid_cost = grid_prices[ts]*grid_used   # [£]

#     # overall variables incremented from timestep variables 
#     tot_elec_ch += elec_used                # total electrical charge [kWhe]
#     tot_therm_ch += charge_dhw + charge_sh  # total thermal charge [kWhth]
#     tot_therm_disch += en_dem_dhw[ts] + en_dem_sh[ts]     # total thermal discharge to meet demands [kWhth]
#     tot_dhw_elec_ch += charge_dhw/cop_dhw[ts]       # total DHW charge [kWhe]
#     tot_dhw_therm_ch += charge_dhw          # total DHW charge [kWhth]
#     tot_dhw_therm_disch += en_dem_dhw[ts]          # total DHW heat provision [kWhth]
#     tot_sh_elec_ch += charge_sh/cop_sh[ts]         # total SH charge [kWhe]
#     tot_sh_therm_ch += charge_sh            # total SH charge [kWhth]
#     tot_sh_therm_disch += en_dem_sh[ts]           # total SH heat provision [kWhth]
#     tot_pv += pv_used                       # total PV electricity used for charging [kWhe]
#     tot_wind += wind_used                   # total wind electricity used for charging [kWhe]
#     tot_grid += grid_used                   # total grid electricity used for charging [kWhe]
    
#     tot_cost += pv_cost + wind_cost + grid_cost # total charging cost during timestep [£]
    
    
#     """Store Values from Timestep"""
#     TimestepRes = store_timestep_results(TimestepRes, t, ts, elec_used, charge_dhw, charge_sh, cop_dhw[ts], cop_sh[ts], 
#                                          soc_dhw[t+1], soc_sh[t+1], pv_en[ts], wind_en[ts], pv_used, wind_used, grid_used, 
#                                          pv_cost, wind_cost, grid_cost)
    
#     Temps = store_node_temps(Temps, t, ts, dhw_tank_T, sh_tank_T, vol_dem_dhw[ts], vol_dem_sh[ts], 
#                              en_dem_dhw[ts], en_dem_sh[ts],  charge_dhw, charge_sh)

# """Calculate KPIs and Export Results"""
# KPIs = calculate_kpis(KPIs, tot_cost, tot_elec_ch, 
#                     tot_therm_ch, tot_therm_disch,
#                     tot_pv, tot_wind, tot_grid,
#                     tot_dhw_elec_ch, tot_dhw_therm_ch,
#                     tot_dhw_therm_disch, tot_sh_elec_ch,
#                     tot_sh_therm_ch, tot_sh_therm_disch)

# export_results(outdir, results_file, temps_file, kpis_file,
#                 TimestepRes, Temps, KPIs)