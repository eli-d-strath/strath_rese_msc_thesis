"""
This will execute the different control scripts depending on what is elected. 
It will also make importing easier.

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
from demand_led_rbc import DemandLedRBC
from opp_pv_rbc import OppPVRBC

"""Simulation Characteristics and Dataset Import & Processing"""
#%% Simulation Input Settings
mth = 1  #11 calibration               # start month
doy = 1    #305 calibration           # start day of year 
doyt = np.array([0,31,59,90,120,151,181,212,243,273,304,334,365])

tariff_options = ["standard","daynight","dwt1","dwt2","dwt3"]
control_options = ["demand_led_rbc","opp_pv_rbc","advanced_rbc"]  
tariff = tariff_options[0] 
control = control_options[0]
days_simulated = 365    # days to simulate - normally 365
ts_length = 0.5         # timestep length [h] 

#%% Calculated Simulation Values
# determine simulation period and if valid
if doy+days_simulated > 366:
    days_simulated = 366-doy
    warnings.warn("Specified simulation period goes beyond available data. \nSimulation period adjusted to stay within available data.")

ts_start = math.floor(doy*(24/ts_length)-(24/ts_length))
ts_end = math.floor(ts_start + days_simulated*24/ts_length)
ts_num = ts_end - ts_start


# dhw_tank_T = np.array([50.5,50.4,50.2,50.,22.]) # Starting DHW tank temperature
dhw_tank_T = np.array([49.8,49.6,49.6,46.8,22.]) # Starting DHW tank temperature
sh_tank_T = np.array([54.,53.,52.,51.,50.]) # Starting SH tank temperature

#%% Possible Control Setpoints (assigned to timesteps later)
dhw_min_T_set = 46      # Minimum DHW tank node[3] temperature before forcing a charge
dhw_lim_T_set = 50      # Minimum DHW tank top sensor temperature before allowing a charge
dhw_max_T_set = 50.5    # DHW tank 4th node sensor temperature at which charging stops
dhw_max_T_ren = 52      # DHW new max top node temperature when renewably charged

sh_min_T_set = 50       # Minimum SH tank top sensor temperature before forcing a charge
sh_lim_T_set = 55       # Minimum SH tank top sensor temperature before allowing a charge
sh_max_T_set = 56       # SH tank top sensor temperature at which charging stops
sh_max_T_ren = 57.      # SH new max top node temperature when renewably charged

dhw_source_T_set = 51       # HP outlet temp to DHW
sh_source_T_set = 57        # HP outlet temp to SH
dhw_source_T_ren = 53       # HP outlet temp to DHW when charging with renewables
sh_source_T_ren = 58        # HP outlet temp to SH when charging with renewables

dhw_duty_set = 7.           # max HP duty [kWhth] to DHW tank per timestep
sh_duty_set = 8.5           # max HP duty [kWth] to SH tank per timestep

dhw_night_offset = 3    # decrease in allowable minimum temperature during setback period [C]
sh_night_offset = 5     # decrease in allowable minimum temperature during setback period [C]

dhw_flow_T = 45.        # assumed DHW flow temp
dhw_return_T = 10.      # assumed DHW return temp
sh_flow_T = 45.         # assumed SH flow temp
sh_return_T = 35.       # assumed SH return temp

PV_thresholds = [4.,4.,4.,4.,4.,4.,4.,4.,4.,4.,4.,4.]  # PV forced charging threshold for every month [kW]
wind_thresholds = [50.,50.,50.,50.,50.,50.,50.,50.,50.,50.,50.,50.]   # Wind surplus charging threshold for every month [kW]

cp = 4.181 #[kJ/kg-K]

cyc_dhw = 0 
cyc_sh = 0

#%% Import Available Datasets
SrcT = np.array([9.3,8.7,8.4,9.1,10.2,11.8,13.5,14.1,13.9,13.1,12.0,10.6]) # Heat Pump Source Temperature (Water)

# Import half-hourly datasets from Emoncms
pv_en = np.loadtxt('c:/Users/user/msc_dissertation/site_data/WS_PV_Act_Energy-kWh_half-hour_2023.csv', delimiter=',')  # [kWh] PV energy generation at WS
pv_pow = np.loadtxt('c:/Users/user/msc_dissertation/site_data/WS_PV_Act_Power-Average_kW_half-hour_2023.csv', delimiter=',')  # [kW] PV power generation at WS
dhw_disch_vol = np.loadtxt('c:/Users/user/msc_dissertation/site_data/WS_DWH_Act_Flow_Liters_half-hour_2023.csv', delimiter=',')    # [L] HW heat supply to WS 
sh_disch_vol = np.loadtxt('c:/Users/user/msc_dissertation/site_data/WS_SH-Flow_Act__Liters_half-hour_2023.csv', delimiter=',')    # [L] SH heat supply to WS
CB001_Act_P = np.loadtxt('c:/Users/user/msc_dissertation/site_data/FH_CB001_Act_Power_kW_half-hour_2023_With-missing-Data-from-2024.csv', delimiter=',')  # [kW] wind power generation
CB002_Act_P = np.loadtxt('c:/Users/user/msc_dissertation/site_data/FH_CB002_Act_Power_kW_half-hour_2023_With-missing-Data-from-2024.csv', delimiter=',')  # [kW] power consumption at main Findhorn site
WW_SS_Act_P = np.loadtxt('c:/Users/user/msc_dissertation/site_data/FH_WW-SS_Power_Act_kW_half-hour_2023_Heavily_Sythesized.csv', delimiter=',')  # [kW] power consumption at WW-SS
amb_temps = np.loadtxt('c:/Users/user/msc_dissertation/site_data/kinloss_daily_av_temp_2023.csv', delimiter=',')    # [C] daily average ambient temperature
water_temps = np.zeros(ts_num)

#%% Clean Imported Datasets at each Timestep
dhw_en_dem = np.zeros(ts_num)     # DHW discharge heat demand
sh_en_dem = np.zeros(ts_num)      # SH discharge heat demand
wind_en = np.zeros(ts_num)          # available wind generation
wind_pow = np.zeros(ts_num)         # available wind power
water_temps = np.zeros(ts_num)      # water temperatures

month = mth
day_year = doy
for ts in range(ts_num):

    #increment time values
    if ts>0 and np.remainder(ts,48) == 0.:
        day_year = day_year + 1
    if day_year > doyt[month]:
        month = month + 1

    wind_pow[ts] = CB001_Act_P[ts] - CB002_Act_P[ts] - WW_SS_Act_P[ts]  # net wind [kW]
    wind_en[ts] = wind_pow[ts] * ts_length

    # data cleaning
    if dhw_disch_vol[ts] < 0 or dhw_disch_vol[ts] > 500:
        dhw_disch_vol[ts] = 0     # [L]
    if sh_disch_vol[ts] < 0 or sh_disch_vol[ts] > 500:
        sh_disch_vol[ts] = 0      # [L]
    if pv_en[ts] < 0 or pv_en[ts] > 5:
        pv_en[ts] = 0      # [kWhe]
    if wind_pow[ts] > 750:
        wind_pow[ts] = 750  # [kW]
        wind_en[ts] = wind_pow[ts] * ts_length  # [kWh]
    elif wind_pow[ts] < 0:
        wind_pow[ts] = 0    # [kW]
        wind_en[ts] = 0     # [kWh]

    # additional datasets calculated
    dhw_en_dem[ts] = (dhw_disch_vol[ts] * cp * (dhw_flow_T-dhw_return_T)) / 3600.        # [kWh]
    sh_en_dem[ts] = (sh_disch_vol[ts] * cp * (sh_flow_T-sh_return_T)) / 3600.          # [kWh]
    water_temps[ts] = SrcT[month-1]     # degrees C

# state of charge = maximum energy that can be discharged by each tank
soc_dhw = np.zeros(ts_num+1)
soc_dhw[0] = dhw_tank.HotWaterTank().max_energy_in_out( # DHW State of charge [kWh] based on 45C minimum
    'discharging', dhw_tank_T,
    0., dhw_flow_T, dhw_return_T, 0, 30)

soc_sh = np.zeros(ts_num+1)
soc_sh[0] = sh_tank.HotWaterTank().max_energy_in_out( # SH State of charge [kWh] based on 45C minimum
    'discharging', sh_tank_T,
    0., sh_flow_T, sh_return_T, 0, 30)

#%% Establish Control Values at each Timestep
day_counter = np.zeros(ts_num)      # day counter to determine SH status
# sh_enabled = np.zeros(ts_num)       # SH status [bool]
pv_enabled = np.zeros(ts_num)       # PV status [bool]
wind_enabled = np.zeros(ts_num)     # wind status [bool]
dhw_min_T = np.zeros(ts_num)        # DHW forced charging threshold temp [degC]
dhw_lim_T = np.zeros(ts_num)        # DHW renewablecharging threshold temp [degC]
dhw_max_T = np.zeros(ts_num)        # DHW stop charging threshold temp  [degC]
sh_min_T = np.zeros(ts_num)         # SH forcedcharging threshold temp [degC]
sh_lim_T = np.zeros(ts_num)         # SH renewable charging threshold temp [degC]
sh_max_T = np.zeros(ts_num)         # SH stop charging threshold temp [degC]

cop_dhw = np.zeros(ts_num)              # heat pump COP during DHW charge
cop_sh = np.zeros(ts_num)               # heat pump COP during SH charge
duty_dhw = np.full(ts_num, dhw_duty_set)      # maximum DHW duty [kWhth] for the heat pump during timestep
duty_sh = np.full(ts_num, sh_duty_set)        # maximum SH duty [kWhth] for the heat pump during timestep
dhw_source_T = np.zeros(ts_num)         # DHW HP outlet temp [degC]
sh_source_T = np.zeros(ts_num)          # SH HP outlet temp [degC]

charges_dhw = np.zeros(ts_num)    
charges_sh = np.zeros(ts_num)
pv_prices = np.zeros(ts_num)
wind_prices = np.zeros(ts_num)
grid_prices = np.zeros(ts_num)


month = mth
day_year = doy
# setting control values at each ts
for ts in range(ts_start,ts_end):
    #increment time values
    if ts>0 and np.remainder(ts,48) == 0.:
        day_year = day_year + 1
    if day_year > doyt[month]:
        month = month + 1
    
    t = ts-ts_start # in case modeling from a timestep other than 0
    # t = math.floor(np.remainder(ts,48) + (doy-1)*48)
    
    # values for first timestep
    if ts == 0:
        day_counter[t] = 0
    else:
        day_counter[t] = temp_day_counter(month,amb_temps[doy-1],day_counter[t-1],t)

    pv_enabled[t] = pv_check(month,pv_en[t],PV_thresholds)
    wind_enabled[t] = wind_check(month,wind_pow[t],wind_thresholds)

    pv_prices[t] = electricity_price("pv",tariff,t)
    wind_prices[t] = electricity_price("wind",tariff,t)
    grid_prices[t] = electricity_price("grid",tariff,t)

    # control values when demand_led_rbc
    if control == control_options[0]:
        dhw_min_T[t] = dhw_min_T_set
        dhw_lim_T[t] = dhw_lim_T_set
        dhw_max_T[t] = dhw_max_T_set
        sh_min_T[t] = sh_min_T_set
        sh_lim_T[t] = sh_lim_T_set
        sh_max_T[t] = sh_max_T_set
        dhw_source_T[t] = dhw_source_T_set
        sh_source_T[t] = sh_source_T_set
        cop_dhw[t] = hp_cop(water_temps[t],dhw_source_T[t])
        cop_sh[t] = hp_cop(water_temps[t],sh_source_T[t])

    # control values when opp_pv_rbc
    elif control == control_options[1]:
        if pv_enabled[t]:
            dhw_min_T[t] = dhw_min_T_set
            dhw_lim_T[t] = dhw_lim_T_set
            dhw_max_T[t] = dhw_max_T_ren
            sh_min_T[t] = dhw_min_T_set
            sh_lim_T[t] = sh_lim_T_set
            sh_max_T[t] = sh_max_T_ren
            dhw_source_T[t] = dhw_source_T_ren
            sh_source_T[t] = sh_source_T_ren
            cop_dhw[t] = hp_cop(water_temps[t],dhw_source_T[t])
            cop_sh[t] = hp_cop(water_temps[t],sh_source_T[t])

        else:
            dhw_min_T[t] = dhw_min_T_set
            dhw_lim_T[t] = dhw_lim_T_set
            dhw_max_T[t] = dhw_max_T_set
            sh_min_T[t] = sh_min_T_set
            sh_lim_T[t] = sh_lim_T_set
            sh_max_T[t] = sh_max_T_set
            dhw_source_T[t] = dhw_source_T_set
            sh_source_T[t] = sh_source_T_set
            cop_dhw[t] = hp_cop(water_temps[t],dhw_source_T[t])
            cop_sh[t] = hp_cop(water_temps[t],sh_source_T[t])

    # control values when advanced_rbc
    elif control == control_options[2]:
        if pv_enabled[t] or wind_enabled[t]:
            dhw_min_T[t] = night_setback(t, dhw_min_T_set, dhw_night_offset, control)
            dhw_lim_T[t] = dhw_lim_T_set
            dhw_max_T[t] = dhw_max_T_ren
            sh_min_T[t] = night_setback(t, sh_min_T_set, sh_night_offset, control)
            sh_lim_T[t] = sh_lim_T_set
            sh_max_T[t] = sh_max_T_ren
            dhw_source_T[t] = dhw_source_T_ren
            sh_source_T[t] = sh_source_T_ren
            cop_dhw[t] = hp_cop(water_temps[t],dhw_source_T[t])
            cop_sh[t] = hp_cop(water_temps[t],sh_source_T[t])

        else:
            dhw_min_T[t] = night_setback(t, dhw_min_T_set, dhw_night_offset, control)
            dhw_lim_T[t] = dhw_lim_T_set
            dhw_max_T[t] = dhw_max_T_set
            sh_min_T[t] = night_setback(t, sh_min_T_set, sh_night_offset, control)
            sh_lim_T[t] = sh_lim_T_set
            sh_max_T[t] = sh_max_T_set
            dhw_source_T[t] = dhw_source_T_set
            sh_source_T[t] = sh_source_T_set
            cop_dhw[t] = hp_cop(water_temps[t],dhw_source_T[t])
            cop_sh[t] = hp_cop(water_temps[t],sh_source_T[t])
    
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

# create titles for output results files
base_file_name = "CALIBRATION" + control + tariff + "_"
results_path = '../simulation_results/'
results_file = base_file_name + "results.csv"
kpis_file = base_file_name + "kpis.csv"
temps_file = base_file_name + "node_temps.csv"

# create results folder if needed
path = 'C:/Users/user/msc_dissertation/simulation_results'
os.makedirs(path,exist_ok=True)

#%% Timestep Model
for t in tqdm(range(ts_start,ts_end)): #iterates through all timesteps of interest
    """Simulation Step Control"""
    
    # increment day and month when necessary
    if t>0 and np.remainder(t,48) == 0.:
        doy = doy + 1
    if doy > doyt[mth]:
        mth = mth + 1
    
    #to call right data in case starting midway through year - same as t if starting on doy=1
    # ts = math.floor(np.remainder(t,48) + (doy-1)*48)
    ts = t-ts_start

    """Reset Control Values"""
    charge_dhw = 0
    charge_sh = 0

    """Run Appropriate Actions for Control"""
    if control == "demand_led_rbc":

        """Discharging and Heat Loss"""
        prev_temp_dhw = np.copy(dhw_tank_T)
        prev_temp_sh = np.copy(sh_tank_T)

        dhw_tank_T = temp_dhw(dhw_en_dem[ts], prev_temp_dhw, dhw_disch_vol[ts], dhw_source_T[ts], dhw_flow_T)
        sh_tank_T = temp_sh(sh_en_dem[ts], prev_temp_sh, sh_disch_vol[ts], sh_source_T[ts], sh_flow_T)
        
        """Charging"""
        demand_led_rbc = DemandLedRBC(pv_enabled[ts], wind_enabled[ts],
                                    dhw_min_T[ts], dhw_lim_T[ts], dhw_max_T[ts],
                                    sh_min_T[ts], sh_lim_T[ts], sh_max_T[ts],
                                    dhw_source_T[ts], sh_source_T[ts], water_temps[ts],
                                    dhw_flow_T,sh_flow_T)

        # DHW - 1st Priority
        cop, charge_dhw, dhw_tank_T, cyc_dhw = demand_led_rbc.dhw_charging(dhw_tank_T, cyc_dhw)
        # SH - 2nd Priority
        cop, charge_sh, sh_tank_T, cyc_sh = demand_led_rbc.sh_charging(sh_tank_T, cyc_sh, charge_dhw)

    elif control == "opp_pv_rbc":
        
        """Discharging and Heat Loss"""
        prev_temp_dhw = np.copy(dhw_tank_T)
        prev_temp_sh = np.copy(sh_tank_T)

        dhw_tank_T = temp_dhw(dhw_en_dem[ts], prev_temp_dhw, dhw_disch_vol[ts], dhw_source_T[ts], dhw_flow_T)
        sh_tank_T = temp_sh(sh_en_dem[ts], prev_temp_sh, sh_disch_vol[ts], sh_source_T[ts], sh_flow_T)
        
        """Charging"""
        pv_rbc = OppPVRBC(pv_enabled[ts], wind_enabled[ts],
                        dhw_min_T[ts], dhw_lim_T[ts], dhw_max_T[ts],
                        sh_min_T[ts], sh_lim_T[ts], sh_max_T[ts],
                        dhw_source_T[ts], sh_source_T[ts], water_temps[ts],
                        dhw_flow_T,sh_flow_T)

        # DHW - 1st Priority
        cop, charge_dhw, dhw_tank_T, cyc_dhw  = pv_rbc.dhw_charging(dhw_tank_T, cyc_dhw)
        # SH - 2nd Priority
        cop, charge_sh, sh_tank_T, cyc_sh = pv_rbc.sh_charging(sh_tank_T, cyc_sh, charge_dhw)

    elif control == "advanced_rbc":
        
        """Discharging and Heat Loss"""
        prev_temp_dhw = np.copy(dhw_tank_T)
        prev_temp_sh = np.copy(sh_tank_T)

        dhw_tank_T = temp_dhw(dhw_en_dem[ts], prev_temp_dhw, dhw_disch_vol[ts], dhw_source_T[ts], dhw_flow_T)
        sh_tank_T = temp_sh(sh_en_dem[ts], prev_temp_sh, sh_disch_vol[ts], sh_source_T[ts], sh_flow_T)

        """Charging"""
        # initialize controller object with current timestep information
        advanced_rbc = AdvancedRBC(ts, doy, mth, 
                                pv_enabled[ts], wind_enabled[ts], day_counter[ts],
                                dhw_min_T[ts], dhw_lim_T[ts], dhw_max_T[ts],
                                sh_min_T[ts], sh_lim_T[ts], sh_max_T[ts],
                                dhw_source_T[ts], sh_source_T[ts], water_temps[ts],
                                dhw_flow_T,sh_flow_T)


        # DHW charging - 1st Priority
        cop, charge_dhw, dhw_tank_T, cyc_dhw  = advanced_rbc.dhw_charging(dhw_tank_T, cyc_dhw)
        # SH charging - 2nd Priority
        cop, charge_sh, sh_tank_T, cyc_sh = advanced_rbc.sh_charging(sh_tank_T, cyc_sh, charge_dhw)

    """Calculate Results"""
    soc_dhw[ts+1] = dhw_tank.HotWaterTank().max_energy_in_out(
        'discharging', dhw_tank_T,
        0., 45., 10., 0, 30)
    
    soc_sh[ts+1] = sh_tank.HotWaterTank().max_energy_in_out(
        'discharging', sh_tank_T,
        0., 45., 35., 0, 30)


    # Timestep Variables
    charges_dhw[ts] = charge_dhw
    charges_sh[ts] = charge_sh

    elec_used = (charge_dhw / cop_dhw[ts]) + (charge_sh / cop_sh[ts])   # [kWh]
    pv_used = min(pv_en[ts],elec_used)                             # [kWh]
    pv_unused = pv_en[ts] - pv_used                                # [kWh]
    pv_cost = pv_prices[ts]*pv_used         # [£]
    wind_used = min(wind_en[ts], elec_used - pv_used)              # [kWh]

    wind_unused = wind_en[ts] - wind_used                          # [kWh]
    wind_cost = wind_prices[ts]*wind_used   # [£]
    grid_used = max(0,elec_used-(pv_used+wind_used))            # [kWh]
    grid_cost = grid_prices[ts]*grid_used   # [£]

    # overall variables incremented from timestep variables 
    tot_elec_ch += elec_used                # total electrical charge [kWhe]
    tot_therm_ch += charge_dhw + charge_sh  # total thermal charge [kWhth]
    tot_therm_disch += dhw_en_dem[ts] + sh_en_dem[ts]     # total thermal discharge to meet demands [kWhth]
    tot_dhw_elec_ch += charge_dhw/cop_dhw[ts]       # total DHW charge [kWhe]
    tot_dhw_therm_ch += charge_dhw          # total DHW charge [kWhth]
    tot_dhw_therm_disch += dhw_en_dem[ts]          # total DHW heat provision [kWhth]
    tot_sh_elec_ch += charge_sh/cop_sh[ts]         # total SH charge [kWhe]
    tot_sh_therm_ch += charge_sh            # total SH charge [kWhth]
    tot_sh_therm_disch += sh_en_dem[ts]           # total SH heat provision [kWhth]
    tot_pv += pv_used                       # total PV electricity used for charging [kWhe]
    tot_wind += wind_used                   # total wind electricity used for charging [kWhe]
    tot_grid += grid_used                   # total grid electricity used for charging [kWhe]
    
    tot_cost += pv_cost + wind_cost + grid_cost # total charging cost during timestep [£]
    
    
    """Store Values from Timestep"""
    TimestepRes = store_timestep_results(TimestepRes, ts, t, elec_used, charge_dhw, charge_sh, cop_dhw[ts], cop_sh[ts], 
                                         soc_dhw[ts+1], soc_sh[ts+1], pv_en[ts], wind_en[ts], pv_used, wind_used, grid_used, 
                                         pv_cost, wind_cost, grid_cost)
    
    Temps = store_node_temps(Temps, ts, t, dhw_tank_T, sh_tank_T, dhw_disch_vol[ts], sh_disch_vol[ts], 
                             dhw_en_dem[ts], sh_en_dem[ts],  charge_dhw, charge_sh)

"""Calculate KPIs and Export Results"""
KPIs = calculate_kpis(KPIs, tot_cost, tot_elec_ch, 
                    tot_therm_ch, tot_therm_disch,
                    tot_pv, tot_wind, tot_grid,
                    tot_dhw_elec_ch, tot_dhw_therm_ch,
                    tot_dhw_therm_disch, tot_sh_elec_ch,
                    tot_sh_therm_ch, tot_sh_therm_disch)

export_results(results_path, results_file, temps_file, kpis_file,
                TimestepRes, Temps, KPIs)