"""
Functions called during simulation in control classes and main.py. 

Based on part of the find_ws_dt_0 script, with additional functions added
for use in the pseudocode-based RBC controls and MPC controls. 

This script defines the functions used to simulate the system
and run controls.
"""

from IPython import get_ipython
ipython = get_ipython()
# ipython.magic('reset -f')
ipython.run_line_magic('reset','-sf')

import os
import math
import numpy as np
import pandas as pd
from scipy.integrate import odeint

from tank_models import dhw_tank
from tank_models import sh_tank


#%% Functions
def hp_cop(water_T: float, source_T:float, method:str="regression"):
    """ Calculates the coefficient of performance of a water-source 
        heat pump based on a generic regression model.
    
        Arguments:
            water_temp {float} -- source water temperature for WSHP [C]
            source_T {float} -- HP output temperature to water tank
            action {str} -- method of calculating COP

        Returns:
            cop {float} -- WSHP COP
    """
    # use the stanard regression model from PyLESA
    if method.lower() == "regression":
        emp_fact = 0.7      # empirical factor to scale performance

        cop = emp_fact * (8.77 - 
                        0.15 * (source_T - water_T) +
                        0.000734 * (source_T - water_T)**2)

        return round(cop,2)
    
    # use known delta_T and COP values from the manufacturer of Ochsner Aqua 22
    elif method == "interpolation":
        cop_data = [(25, 5.9), (40, 3.9), (50, 2.9)]    # [(delta_T, COP)] in order of ascending delta_T
        delta_T_known, cop_known = zip(*cop_data)       # splits tuples
        delta_T_known = np.array(delta_T_known)         # converts to np.array
        cop_known = np.array(cop_known)                 # converts to np.array
        
        delta_T_actual = source_T-water_T               # current delta_T
        points = len(delta_T_known)
        emp_fact = 0.9

        if delta_T_actual <= delta_T_known[0]:
            cop = emp_fact*cop_known[0]
        elif delta_T_actual >= delta_T_known[points-1]:
            cop = emp_fact*cop_known[points-1]
        elif delta_T_known[0] < delta_T_actual < delta_T_known[1]:
            cop = emp_fact*np.interp(delta_T_actual,delta_T_known,cop_known)

        return round(cop,2)

def pv_check(month:int, pv_energy:float,PV_thresholds:list):
    """ checks if PV supply breaks threshold.
    
        Arguments:
            month {int} -- the current simulation month
            pv_energy {float} -- PV generated in this timestep [kWh]
            PV_thresholds {list} -- PV threshold for each month in the year

        Returns:
            pv_priority {bool} -- if PV exceeds specified threshold
    """
    if pv_energy >= (PV_thresholds[month-1]/2.):
        pv_priority = True
    else:
        pv_priority = False
    
    return pv_priority

def wind_check(month:int, wind_power:float,wind_thresholds:float):
    """ checks if wind net supply breaks threshold.
    
        Arguments:
            month {int} -- the current simulation month
            wind_power {float} -- net wind power generated in this timestep [kWh]
            wind_thresholds {list} -- wind threshold for each month in the year

        Returns:
            wind_priority {bool} -- if net wind exceeds specified threshold
    """

    if wind_power >= wind_thresholds[month-1]:
        wind_priority = True
    else:
        wind_priority = False

    return wind_priority

def electricity_price(source:str, tariff:str, timestep:int):
    """ Calculates the price of electricity for the given source
        under the given tariff setup in £/kWh

        Source for some prices: https://findhornwind.co.uk/electricity-prices-for-the-year-from-1st-november-2023/
    
        Arguments:
            source {str} -- the source of electricity (pv, wind, or grid)
            tariff {str} -- the tariff scheme (standard, wind_tariff)

        Returns:
            price {float} -- cost of electricity in £ per kWh
    """
    hour = math.floor((timestep % 48)/2)    # calculate hour of day from timestep
    
    if tariff.lower() == "standard":
        if source.lower() == "pv":
            price = 0.      # [£ per kWh]
        elif source.lower() == "wind":
            price = 0.3407   # [£ per kWh]
        elif source.lower() == "grid":
            price = 0.3407   # [£ per kWh]
    
    elif tariff.lower() == "daynight":
        #day prices
        if 7 <= hour < 12:
            if source.lower() == "pv":
                price = 0.      # [£ per kWh]
            elif source.lower() == "wind":
                price = 0.3607   # [£ per kWh]
            elif source.lower() == "grid":
                price = 0.3607   # [£ per kWh]
        #night prices
        else: 
            if source.lower() == "pv":
                price = 0.      # [£ per kWh]
            elif source.lower() == "wind":
                price = 0.3171   # [£ per kWh]
            elif source.lower() == "grid":
                price = 0.3171   # [£ per kWh]

    elif tariff.lower() == "windtariff":
        if source.lower() == "pv":
            price = 0.      # [£ per kWh]
        elif source.lower() == "wind":
            price = 0.18   # [£ per kWh]
        elif source.lower() == "grid":
            price = 0.45    # [£ per kWh]

    return price

def temp_day_counter(month: int, daily_amb_temp: float, day_counter: int, timestep: int):
    """ Adjusts the day count for determining SH activation status.
    
        Arguments:
            month {int} -- current simulation month
            daily_amb_temp {float} -- daily_average_ambient temperature
            day_counter {int} -- current day count
            timestep {int} -- current simulation timestep

        Returns:
            day_counter {int} -- adjusted day count
    """
    temp_off = 14 
    temp_on = 16
    temp_low = 11   # automatic 

    # only increment at end of each day
    if (timestep % 48 == 0) and timestep != 0:
        # when not shoulder season, set day count to 0
        if month != 5 and month != 9:
            day_counter = 0
        
        # when May, increment day count if daily average temp is above the off-setpoint
        elif month == 5 and daily_amb_temp >= temp_off:
            day_counter += 1
        
        # when September, increment day count if temps below low setpoint
        elif month == 9 and daily_amb_temp <= temp_on:
            day_counter += 1 

        # when September, set day_count to max_days to trigger SH if temp below health low temp threshold
        elif month == 9 and daily_amb_temp < temp_low:
            day_counter = 6 
        
    return day_counter

def sh_activation_status(month: int, day_counter: int):
    """ Determines whether SH is active/inactive based on day_counter 
    
        Arguments:
            month {int} -- the current simulation month
            daily_amb_temp {float} -- daily_average_ambient temperature
            day_counter {int} -- current day count

        Returns:
            status {bool} -- SH activity status 
    """
    max_days = 6 

    # SH always on in fall, winter, spring
    if (month < 5) or (month > 9):
        sh_enabled = True
    
    # SH always off in summer
    elif 5 < month < 9:
        sh_enabled = False
    
    # Turn SH off in May when temp is warm enough for max days
    elif month == 5:
        if day_counter >= max_days:
            sh_enabled = False
        elif day_counter < max_days:
            sh_enabled = True
    
    # Turn SH on in September when temp is cold enough for max days
    elif month == 9:
        if day_counter >= max_days:
            sh_enabled = True
        elif day_counter < max_days:
            sh_enabled = False

    return sh_enabled 

def night_setback(timestep: int, min_T: float, offset: float, control: str='rbc_baseline', start: int=19, end: int=9):
    """ Calculates the T_min to use during the setback period.
        Current instantiation assumes this is used between 7pm and 9am
    
        Arguments:
            timestep {int} -- the current simulation timestep
            min_T {float} -- typical SH or HW T_min [C]
            start {int} -- start timestep for setback (default of 7:00pm)
            end {int} -- ending timestep for setback (default of 9:00am)
            offset {float} -- temperature offset for night period [C]

        Returns:
            setback_min_T {float} -- setback value for T_min [C]
    """
    hour = math.floor((timestep % 48)/2)    # calculate hour of day from timestep

    if hour >= start or hour < end:  
        setback_min_T = min_T - offset
        return setback_min_T
    else:
        return min_T

def carbon_emissions(pv_charge: float, wind_charge: float, grid_charge:float):
    """ Calculates total carbon emissions based on electricity consumption.
    
        Arguments:
            pv_charge {float} -- PV electricity used for charging [kWh]
            wind_charge {float} -- wind electricity used for charging [kWh]
            grid_charge {float} -- grid electricity used for charging [kWh]

        Returns:
            emissions {float} -- total CO2eq emissions [kg]
    """
    pv_intensity = 43           # [gCO2eq/kWh]
    wind_intensity = 11.8       # [gCO2eq/kWh]
    grid_intensity = 254        # [gCO2eq/kWh]
    emissions = (pv_intensity*pv_charge + wind_intensity*wind_charge + grid_intensity*grid_charge)/1000 # [kg]

    return emissions

#TODO create a method to store and output results that takes up less space
def store_results():
    return 1



# methods developed but not used in this analysis.
'''
# def weather_comp_curve(ambient_temp: float): 
    """ Calculates SH flow temp using weather compensation.
    
        Arguments:
            ambient_temp {float} -- ambient outdoor air temperature [C]

        Returns:
            comp_flow_temp {float} -- required SH flow temperature [C]
    """
    min_temp = -3.
    max_temp = 23.
    max_flow_temp = 45.
    min_flow_temp = 35.

    if ambient_temp <= min_temp:
        return max_flow_temp
    elif ambient_temp >= max_temp:
        return min_flow_temp
    else:
        wc_factor = (max_flow_temp-min_flow_temp)/(max_temp-min_temp)
        comp_flow_temp = min_flow_temp + (max_temp-ambient_temp)*wc_factor
        return comp_flow_temp

# def DC_check(t, DC_counter, dhw_tank_T, pv_enabled, wind_enabled, DC_limit, HW_DC_T):
    """ checks if DC should be prioritized depending on renewables availability
        or days since last cycle. Also, increments counter since last cycle. 
    
        Arguments:
            t {int} -- the current simulationtimestep
            DC_counter {int} -- number of days since last DC
            pv_priority {bool} -- if PV exceeds specified threshold
            wind_priority {bool} -- if net wind exceeds specified threshold
            hw_tank_T {array} -- hot water tank temperatures
            DC_limit {int} -- maximum days between DCs
            HW_DC_T {float} -- target temperature for DC [C]


        Returns:
            DC_priority {bool} -- SH activity status
            DC_counter {int} -- number of days since last DC
    """
    # increment DC counter every 48 timesteps 
    if t % 48 == 0 and t != 0:  
        DC_counter += 1        
    
    #TODO issue: the DC counter immediately turns off...
    #TODO check if this level of surety is actually necessary...
    # if the whole tank reaches 60C, stop DC and restart DC_counter
    ave_temp = (dhw_tank_T[0]+dhw_tank_T[1]+dhw_tank_T[2]+dhw_tank_T[4])/4
    if ave_temp >= 60.:
        DC_priority = False
        DC_counter = 0

    # when PV is available, and 6+ days since last cycle, run DC
    elif pv_enabled == True and DC_counter >= DC_limit-4:
        DC_priority = True
        #DC_counter = 0

    # when PV is available, and 9+ days since last cycle, run DC
    elif wind_enabled == True and DC_counter >= DC_limit-1:
        DC_priority = True
        #DC_counter = 0

    # when 10+ days since last cycle, run DC
    elif DC_counter > DC_limit:
        DC_priority = True
        #DC_counter = 0

    # for all other conditions, do not run DC
    else:
        DC_priority = False   
        
    return DC_priority, DC_counter
'''

#%% DHW Tank Temperature Model
def temp_dhw(delta, prev_temp, vol, source_temp, flow_temp):
    
    if delta > 0.:
        new_temp = dhw_tank.HotWaterTank().new_nodes_temp(
            state='discharging', nodes_temp=prev_temp, source_temp=source_temp, 
            source_delta_t=40, flow_temp=flow_temp, #! changed - set flow_temp to current top temp instead of 45C
            return_temp=10., thermal_output=0., demand=delta, 
            timestep=0., MTS=30, vol=vol)

    elif delta < 0.:
        new_temp = dhw_tank.HotWaterTank().new_nodes_temp(
            state='charging', nodes_temp=prev_temp, source_temp=source_temp, 
            source_delta_t=prev_temp[0] - prev_temp[4], flow_temp=45., 
            return_temp=10., thermal_output=-delta, demand=0., 
            timestep=0., MTS=30, vol=vol)
    
    else:
        new_temp = dhw_tank.HotWaterTank().new_nodes_temp(   # Heat loss only
            state='standby', nodes_temp=prev_temp, source_temp=source_temp, 
            source_delta_t=40, flow_temp=50.,
            return_temp=5., thermal_output=0., demand=0., 
            timestep=0, MTS=30, vol=0)
    
    return new_temp

#%% SH Tank Temperature Model
def temp_sh(delta, prev_temp, vol, source_temp, flow_temp):
    if delta > 0.:
        new_temp = sh_tank.HotWaterTank().new_nodes_temp(
            state='discharging', nodes_temp=prev_temp, source_temp=source_temp, 
            source_delta_t=prev_temp[0]-prev_temp[4], flow_temp=flow_temp,
            return_temp=35, thermal_output=0., demand=delta, 
            timestep=0., MTS=30, vol=vol)
        
    elif delta < 0.:
        new_temp = sh_tank.HotWaterTank().new_nodes_temp(
            state='charging', nodes_temp=prev_temp, source_temp=source_temp, 
            source_delta_t=prev_temp[0] - prev_temp[4], flow_temp=flow_temp, 
            return_temp=35, thermal_output=-delta, demand=0., 
            timestep=0., MTS=30, vol=0)
    else:
        new_temp = sh_tank.HotWaterTank().new_nodes_temp(   # Heat loss only
            state='standby', nodes_temp=prev_temp, source_temp=source_temp, 
            source_delta_t=prev_temp[0] - prev_temp[4], flow_temp=flow_temp,
            return_temp=35, thermal_output=0., demand=0., 
            timestep=0, MTS=30, vol=0) 
    
    return new_temp