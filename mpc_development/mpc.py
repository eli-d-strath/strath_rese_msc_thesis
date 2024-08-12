"""
Model Predictive Control (MPC) for Woodside.

This control is based on the MPC controller included in PyLESA,
a tool developed by Andrew Lyden.  
"""

from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic('reset','-sf')

import sys
import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
from scipy.integrate import odeint
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
from sim_tools import carbon_emissions
from sim_tools import electricity_price

import logging
from gekko import GEKKO
from progressbar import Bar, ETA, Percentage, ProgressBar, RotatingMarker
import pickle

LOG = logging.getLogger(__name__)
np.set_printoptions(legacy='1.25')

class MPCScheduler(object):

    def __init__(self, horizon:int, ts_num: int, ts_length: float,
                 tariff:str, doyt: np.ndarray,
                 pv_thresholds: np.ndarray, wind_thresholds: np.ndarray,
                 init_temps_dhw: np.ndarray, init_temps_sh: np.ndarray,
                 vol_dhw_data: np.ndarray, vol_sh_data: np.ndarray,
                 pv_E_data: np.ndarray, 
                 CB001_P_data: np.ndarray, CB002_P_data: np.ndarray,
                 WW_SS_P_data: np.ndarray, amb_T_data: np.ndarray,
                 water_T_data: np.ndarray):
        
        #! all values are passed in successfully (debugged)
        self.myDHWTank = dhw_tank.HotWaterTank()# HotWaterTank object
        self.mySHTank = sh_tank.HotWaterTank()  # HotWaterTank object
        self.horizon = horizon #336             # [timesteps] prediction horizon
        self.tariff = tariff                    # standard, daynight, or windtariff
        self.ts_num = ts_num
        self.ts_length = ts_length  # [hours]
        self.doyt = doyt
        self.cp = 4.181     # [kJ/kg-K]

        # hard-coded system variables for Woodside
        self.st_set_dhw = 51.           # [degC] {float}
        self.st_set_sh = 57.            # [degC] {float}
        self.st_ren_dhw = 53.           # [degC] {float}
        self.st_ren_sh = 58.            # [degC] {float}

        self.rt_dhw = 10.               # [degC] {float}
        self.rt_sh = 35.                # [degC] {float}
        self.ft_dhw = 45.               # [degC] {float}
        self.ft_sh = 45.                # [degC] {float}
        self.duty_dhw = 7.              # [kWhth] {float}
        self.duty_sh = 8.5              # [kWhth] {float}

        self.pv_thresholds = pv_thresholds 
        self.wind_thresholds = wind_thresholds 
        self.init_temps_dhw = init_temps_dhw    # [degC] {ndarray} - 5
        self.init_temps_sh = init_temps_sh      # [degC] {ndarray} - 5
        self.vol_dhw_data = vol_dhw_data        # [L] {ndarray} - ts
        self.vol_sh_data = vol_sh_data          # [L] {ndarray} - ts
        self.pv_E_data = pv_E_data              # [kWh] {ndarray} - ts
        self.CB001_P_data = CB001_P_data        # [kW] {ndarray} - ts
        self.CB002_P_data = CB002_P_data        # [kW] {ndarray} - ts
        self.WW_SS_P_data = WW_SS_P_data        # [kW] {ndarray} - ts
        self.amb_T_data = amb_T_data            # [degC] {ndarray} - ts
        self.water_T_data = water_T_data        # [degC] {ndarray} - months+1
    
    def pre_calculation(self, doy, month, first_timestep, total_timesteps):
        """ Pre-calculating parameters for the MPC simulation over the year.

            HP performance (COP)
            renewable generation (wind and PV)
            heat demands (SH and DHW)
            tariff prices (different rates depending) 
        """

        # sorting out timestep range
        first_timestep = first_timestep
        total_timesteps = total_timesteps + self.horizon

        # Creating Empty Variables to Fill
        vol_dem_dhw = np.zeros(total_timesteps)         # [L] {ndarray} of DHW volume demands  
        vol_dem_sh = np.zeros(total_timesteps)         # [L] {ndarray} of SH volume demands
        en_dem_dhw = np.zeros(total_timesteps)          # [kWhth] {ndarray} of DHW heat demands
        en_dem_sh = np.zeros(total_timesteps)           # [kWhth] {ndarray} of SH heat demands
        pv_pow = np.zeros(total_timesteps)
        pv_en = np.zeros(total_timesteps)               # [kWhe] ndarray of PV generation over the year
        wind_pow = np.zeros(total_timesteps)
        wind_en = np.zeros(total_timesteps)             # [kWhe] ndarray of wind generation over the year

        cop_dhw = np.zeros(total_timesteps)             # [] {ndarray} COP over the year
        cop_sh = np.zeros(total_timesteps)              # [] {ndarray} COP over the year
        st_dhw = np.zeros(total_timesteps)              # [degC] {ndarray} DHW HP outlet/source temp
        st_sh = np.zeros(total_timesteps)               # [degC] {ndarray} SH HP outlet/source temp

        pv_prices = np.zeros(total_timesteps)           # [£/kWh] {ndarray} pv_prices under given tariff
        wind_prices = np.zeros(total_timesteps)         # [£/kWh] {ndarray} wind_prices under given tariff
        grid_prices = np.zeros(total_timesteps)         # [£/kWh] {ndarray} grid_prices under given tariff

        day_counter = np.zeros(total_timesteps)         # day counter to determine SH status
        sh_enabled = np.zeros(total_timesteps)          # SH status as 1 (active) or 0 (inactive) [int]

        max_capacity_dhw = np.zeros(total_timesteps)
        max_capacity_sh = np.zeros(total_timesteps)

        final_timestep = first_timestep + total_timesteps
        if final_timestep >= 17519 - self.horizon:
            final_timestep = 17519

        # iterate over timesteps of interest to assign values
        for timestep in range(first_timestep, final_timestep):
            ts = timestep - first_timestep   # gives a relative value - computes to 0 if starting in first timestep of year, so don't worry about it now

            """Iterating Month and Day of Year"""
            #increment time values
            if ts>0 and np.remainder(timestep,48) == 0.:
                doy = doy + 1
            if doy > self.doyt[month]:
                month = month + 1

            """Passing Input Data"""
            vol_dem_dhw[ts] = self.vol_dhw_data[timestep]
            vol_dem_sh[ts] = self.vol_sh_data[timestep]         # [L] SH volume demands
            wind_pow[ts] = self.CB001_P_data[timestep] - self.CB002_P_data[timestep] - self.WW_SS_P_data[timestep]  # net wind [kW]
            wind_en[ts] = wind_pow[ts] * self.ts_length
            pv_en[ts] = self.pv_E_data[timestep]
            pv_pow[ts] = pv_en[ts] / self.ts_length

            """Cleaning Input Data"""
            if vol_dem_dhw[ts] < 0 or vol_dem_dhw[ts] > 500:
                vol_dem_dhw[ts] = 0     # [L]
            if vol_dem_sh[ts] < 0 or vol_dem_sh[ts] > 500:
                vol_dem_sh[ts] = 0      # [L]
            if pv_en[ts] < 0 or pv_en[ts] > 5:
                pv_en[ts] = 0       # [kWhe]
                pv_pow[ts] = 0      # [kW]
            if wind_pow[ts] > 750:
                wind_pow[ts] = 750  # [kW]
                wind_en[ts] = wind_pow[ts] * self.ts_length  # [kWhe]
            elif wind_pow[ts] < 0:
                wind_pow[ts] = 0    # [kW]
                wind_en[ts] = 0     # [kWhe]

            """Calculating New Datasets"""
            water_temp = self.water_T_data[month-1]     # [degC]

            # heat demands and temperatures
            en_dem_dhw[ts] = (vol_dem_dhw[ts] * self.cp * (self.ft_dhw-self.rt_dhw)) / 3600.       # [kWh]
            en_dem_sh[ts] = (vol_dem_sh[ts] * self.cp * (self.ft_sh-self.rt_sh)) / 3600.          # [kWh]

            # electricity price by source
            pv_prices[ts] = electricity_price("pv",self.tariff,ts)
            wind_prices[ts] = electricity_price("wind",self.tariff,ts)
            grid_prices[ts] = electricity_price("grid",self.tariff,ts)

            # determining if SH enabled
            air_temp = self.amb_T_data[doy-1]
            if ts == 0:
                day_counter[ts] = 0
                sh_enabled[ts] = 1 #True
            else:
                day_counter[ts] = temp_day_counter(month,air_temp,day_counter[ts-1],ts)

                if sh_activation_status(month,day_counter[ts]) == True:
                    sh_enabled[ts] = 1
                else:
                    sh_enabled[ts] = 0

            # set HP charge temps based on renewable availability
            pv_enabled = pv_check(month,pv_en[ts],self.pv_thresholds)
            wind_enabled = wind_check(month,wind_pow[ts],self.wind_thresholds)

            if pv_enabled or wind_enabled:
                st_dhw[ts] = self.st_ren_dhw
                st_sh[ts] = self.st_ren_sh
                cop_dhw[ts] = hp_cop(water_temp,st_dhw[ts])
                cop_sh[ts] = hp_cop(water_temp,st_sh[ts])

            else:
                st_dhw[ts] = self.st_set_dhw
                st_sh[ts] = self.st_set_sh
                cop_dhw[ts] = hp_cop(water_temp,st_dhw[ts])
                cop_sh[ts] = hp_cop(water_temp,st_sh[ts])

            """Calculate Thermal storage max capacity in each timestep"""
            dhw_return_temp_nodes = []
            sh_return_temp_nodes = []
            for n in range(5): #5 water tank nodes
                    dhw_return_temp_nodes.append(self.rt_dhw)
                    sh_return_temp_nodes.append(self.rt_sh)
                
            # maximum possible charging energy is bringing nodes from return temp to source temp...
            # differs over timesteps because the source temp changes depending on renewables availability
            max_capacity_dhw[ts] = self.myDHWTank.max_energy_in_out(
                    'charging', dhw_return_temp_nodes,
                    st_dhw[ts], self.ft_dhw, self.rt_dhw, ts, 30)
            max_capacity_sh[ts] = self.mySHTank.max_energy_in_out(
                    'charging', sh_return_temp_nodes,
                    st_sh[ts], self.ft_sh, self.rt_sh, ts, 30)

        # create a dictionary of labels:ndarrays
        pre_calc = {
            'vol_dem_dhw': vol_dem_dhw,
            'vol_dem_sh': vol_dem_sh,
            'heat_dem_dhw': en_dem_dhw,
            'heat_dem_sh': en_dem_sh,
            'st_dhw': st_dhw,
            'st_sh': st_sh,
            'cop_dhw': cop_dhw,
            'cop_sh': cop_sh,
            'sh_enabled': sh_enabled,
            'max_capacity_dhw': max_capacity_dhw,
            'max_capacity_sh': max_capacity_sh,
            'pv_gen': pv_en,
            'wind_gen': wind_en,
            'pv_prices':pv_prices,
            'wind_prices':wind_prices,
            'grid_prices':grid_prices,
            }
        
        return pre_calc

    def solve(self, pre_calc, timestep, first_timestep, final_horizon_timestep, prev_result):

        # Extract ndarrays from pre_calc
        vol_dem_dhw = pre_calc['vol_dem_dhw']
        vol_dem_sh = pre_calc['vol_dem_sh']
        heat_dem_dhw = pre_calc['heat_dem_dhw']
        heat_dem_sh = pre_calc['heat_dem_sh']
        st_dhw = pre_calc['st_dhw']
        st_sh = pre_calc['st_sh']
        cop_dhw = pre_calc['cop_dhw']
        cop_sh = pre_calc['cop_sh']
        sh_enabled = pre_calc['sh_enabled']
        max_capacity_dhw = pre_calc['max_capacity_dhw']
        max_capacity_sh = pre_calc['max_capacity_sh']
        pv_gen = pre_calc['pv_gen']
        wind_gen = pre_calc['wind_gen']
        pv_prices = pre_calc['pv_prices']
        wind_prices = pre_calc['wind_prices']
        grid_prices = pre_calc['grid_prices']

        # Temperatures Passed
        rt_dhw = self.rt_dhw  
        rt_sh = self.rt_sh 
        ft_dhw = self.ft_dhw
        ft_sh = self.ft_sh

        #! Notes on the Thermal Storage model
        # a simple thermal storage model is used here, as GEKKO can't implicitly calculate node temperatures into the future.
        # doing this with a more advanced model would be ideal and can be scoped into future work

        """Creating GEKKO Model"""
        # configure timesteps for indexing and discretizing
        number_timesteps = final_horizon_timestep - timestep
        t1 = timestep - first_timestep  
        t2 = final_horizon_timestep - first_timestep 

        m = GEKKO(remote=False) #False for faster runtime, True if issues running   
        m.time = np.linspace(0, number_timesteps-1, number_timesteps) 

        """Variable Initialization""" #intermediates must be explicitly (not implicitly) calculated, so can't call external functions
        ### Cost Coefficients for PV, wind, grid electricity
        PVC_p = m.Param(value=list(pv_prices[t1:t2]))     # [£/kWh]
        WC_p = m.Param(value=list(wind_prices[t1:t2]))    # [£/kWh]
        GC_p = m.Param(value=list(grid_prices[t1:t2]))    # [£/kWh]

        ### Renewable Generation 
        pv_p = m.Param(value=list(pv_gen[t1:t2]))         # [kWhe]
        wind_p = m.Param(value=list(wind_gen[t1:t2]))     # [kWhe]
        
        ### Heat and Volume Demand
        hd_dhw_p = m.Param(value=list(heat_dem_dhw[t1:t2]))
        hd_sh_p = m.Param(value=list(heat_dem_sh[t1:t2]))
        vd_dhw_p = m.Param(value=list(vol_dem_dhw[t1:t2]))
        vd_sh_p = m.Param(value=list(vol_dem_sh[t1:t2]))

        ### Heat Pump 
        # duty
        duty_dhw_c = m.Const(self.duty_dhw)
        duty_sh_c = m.Const(self.duty_sh)
        # cop
        cop_dhw_p = m.Param(value=list(cop_dhw[t1:t2]))
        cop_sh_p = m.Param(value=list(cop_sh[t1:t2]))
        # charging with renewables 
        HPtpvs_dhw_v = m.Var(value=prev_result['HPtpvs_dhw'], lb=0)   # HP pv --> storage DHW [kWhth]
        HPtpvs_sh_v = m.Var(value=prev_result['HPtpvs_sh'], lb=0)     # HP pv --> storage SH  [kWhth]
        HPtws_dhw_v = m.Var(value=prev_result['HPtws_dhw'], lb=0)     # HP wind --> storage DHW [kWhth]
        HPtws_sh_v = m.Var(value=prev_result['HPtws_sh'], lb=0)       # HP wind --> storage SH [kWhth]
        HPtis_dhw_v = m.Var(value=prev_result['HPtis_dhw'], lb=0)     # HP grid import --> storage DHW [kWhth]
        HPtis_sh_v = m.Var(value=prev_result['HPtis_sh'], lb=0)       # HP grid import --> storage SH [kWhth]
        # on/off status #TODO issue - HP_status not staying at integer values...
        sh_enabled_p = m.Param(value=list(sh_enabled[t1:t2]))
        HP_status_dhw_v = m.MV(value=prev_result['HP_status_dhw'], lb=0, ub=1, integer=True)
        HP_status_sh_v = m.MV(value=prev_result['HP_status_sh'], lb=0, ub=1, integer=True)
        # thermal output, depending on status
        HPto_dhw_i = m.Intermediate(HP_status_dhw_v * duty_dhw_c)
        HPto_sh_i = m.Intermediate(HP_status_sh_v * duty_sh_c)

        ### Thermal Storage 
        # maximum charging capacity
        max_cap_dhw_p = m.Param(value=list(max_capacity_dhw[t1:t2]))
        max_cap_sh_p = m.Param(value=list(max_capacity_sh[t1:t2])) 
        #initial state of charge
        init_soc_dhw = self.myDHWTank.max_energy_in_out(
            'discharging', prev_result['final_nodes_temp_dhw'],
            st_dhw[timestep-first_timestep], ft_dhw, rt_dhw, timestep, 30)
        init_soc_sh = self.mySHTank.max_energy_in_out(
            'discharging', prev_result['final_nodes_temp_sh'],
            st_sh[timestep-first_timestep], ft_sh, rt_sh, timestep, 30)
        # storage charge/discharge amount
        TSc_dhw_v = m.Var(value=prev_result['TSc_dhw'], lb=0)     
        TSc_sh_v = m.Var(value=prev_result['TSc_sh'], lb=0)       
        TSd_dhw_v = m.Var(value=prev_result['TSd_dhw'], lb=0)     
        TSd_sh_v = m.Var(value=prev_result['TSd_sh'], lb=0)
        #for clarity max discharge is simply the soc
        soc_dhw_v = m.Var(value=init_soc_dhw, lb=0)               
        soc_sh_v = m.Var(value=init_soc_sh, lb=0)
        max_charge_dhw_i = m.Intermediate(max_cap_dhw_p - soc_dhw_v)   
        max_charge_sh_i = m.Intermediate(max_cap_sh_p - soc_sh_v)
        loss_dhw_i = m.Intermediate(0.01 * soc_dhw_v)
        loss_sh_i = m.Intermediate(0.01 * soc_sh_v)

        
        
        

        #! REALIZATION: when the top node temperature is below the flow temperature, the SOC is assigned to be zero within max_energy_in_out
        #! Therefore, we can simply set a constraint where SOC must be greater than 0, right?

        """Equality Constraints"""
        m.Equations([# heat demand must be met, but can be exceeded to store more
                    hd_dhw_p == TSd_dhw_v,
                    hd_sh_p == TSd_sh_v,
                    soc_dhw_v.dt() == TSc_dhw_v - TSd_dhw_v - loss_dhw_i,
                    soc_sh_v.dt() == TSc_sh_v - TSd_sh_v - loss_sh_i,
                    HP_status_dhw_v * duty_dhw_c == HPtpvs_dhw_v + HPtws_dhw_v + HPtis_dhw_v,    # energy charged to DHW must sum to energy from PV, wind, grid
                    HP_status_sh_v * duty_sh_c == HPtpvs_sh_v + HPtws_sh_v + HPtis_sh_v,         # energy charged to DHW must sum to energy from PV, wind, grid
                    TSc_dhw_v == HP_status_dhw_v * duty_dhw_c,    #! see if redundant and can eliminate a variable
                    TSc_sh_v == HP_status_sh_v * duty_sh_c,      #! see if redundant and can eliminate a variable
                    ])
        
        #TODO multiply sh_enabled_p by something so constraints aren't violated in summer?
        # I understand - HPt_var was meant to be bounded between duty and HP_min, so it operates in that range, only 
        """Inequality Constraints"""
        m.Equations([soc_dhw_v <= max_cap_dhw_p,
                     soc_sh_v <= max_cap_sh_p,
                     TSc_dhw_v <= max_charge_dhw_i,
                     TSc_sh_v <= max_charge_sh_i,
                     TSd_dhw_v <= soc_dhw_v,
                     TSd_sh_v <= soc_sh_v,
                    #  HP_status_dhw_v + HP_status_sh_v <= 1,   #! so DHW and SH can't operate simultaneously
                     HPtpvs_dhw_v/cop_dhw_p + HPtpvs_sh_v/cop_sh_p <= pv_p,
                     HPtws_dhw_v/cop_dhw_p + HPtws_sh_v/cop_sh_p <= wind_p,
                     #HP_status_sh_v <= sh_enabled_p
                     ])     # SH can't charge if deactivated seasonally
                    
                    # TSd_dhw_v >= hd_dhw_p,        # discharged energy must exceed heat demand DHW
                    # TSd_sh_v >= hd_sh_p,           # discharged energy must exceed heat demand SH
                    # soc_dhw_v > 0,         # can't discharge more energy than there is inside
                    # soc_sh_v > 0,           # can't discharge more energy than there is inside

        """Objective Function"""
        m.Minimize(PVC_p * (HPtpvs_dhw_v/cop_dhw_p + HPtpvs_sh_v/cop_sh_p) + 
                    WC_p * (HPtws_dhw_v/cop_dhw_p + HPtws_sh_v/cop_sh_p) +
                    GC_p * (HPtis_dhw_v/cop_dhw_p + HPtis_sh_v/cop_sh_p)
                    )

        """Solver Options and Execution"""
        m.options.IMODE = 6  # MPC mode
        m.options.SOLVER = 1  # APOPT for solving MINLP problems

        # if self.myHeatPump.minimum_output == 0:
        #     i = 'minlp_as_nlp 1'
        # else:
        #     i = 'minlp_as_nlp 0'
        i = 'minlp_as_nlp 1'
        m.solver_options = ['minlp_maximum_iterations 500', \
                            # minlp iterations with integer solution
                            'minlp_max_iter_with_int_sol 500', \
                            # treat minlp as nlp
                            'minlp_as_nlp 1', \
                            # nlp sub-problem max iterations
                            'nlp_maximum_iterations 500', \
                            # 1 = depth first, 2 = breadth first
                            'minlp_branch_method 1', \
                            # maximum deviation from whole number
                            'minlp_integer_tol 0.05', \
                            # covergence tolerance
                            'minlp_gap_tol 0.05']

        print("Going to solve...")
        m.solve(disp=True, debug=True)
        print("SOLVED")

        """Determining State and Nodes Temp"""
        h = 1

        TSc_dhw_ = round(TSc_dhw_v[h] - TSd_dhw_v[h], 2)
        TSc_sh_ = round(TSc_sh_v[h] - TSd_sh_v[h], 2)
        TSd_dhw_ = round(TSd_dhw_v[h] - TSc_dhw_v[h], 2)
        TSd_sh_ = round(TSd_sh_v[h] - TSc_sh_v[h], 2)

        # calculate state for DHW tank
        if TSc_dhw_ > TSd_dhw_:
            max_c = self.myDHWTank.max_energy_in_out(
                'charging',
                prev_result['final_nodes_temp_dhw'],
                st_dhw[h], ft_dhw, rt_dhw, h, 30)
            TSc_dhw_v[h] = min(max_c, TSc_dhw_)
            TSd_dhw_v[h] = 0
            if TSc_dhw_v[h] == 0.0:
                state_dhw = 'standby'
            else:
                state_dhw = 'charging'
        elif TSd_dhw_ > TSc_dhw_:
            max_d = self.myDHWTank.max_energy_in_out(
                'discharging',
                prev_result['final_nodes_temp_dhw'],
                st_dhw[h], ft_dhw, rt_dhw, h, 30)
            TSd_dhw_v[h] = min(max_d, TSd_dhw_)
            TSc_dhw_v[h] = 0.0
            if TSd_dhw_v[h] == 0.0:
                state_dhw = 'standby'
            else:
                state_dhw = 'discharging'
        else:
            state_dhw = 'standby'

        # calculate state for SH tank
        if TSc_sh_ > TSd_sh_:
            max_c = self.mySHTank.max_energy_in_out(
                'charging',
                prev_result['final_nodes_temp_sh'],
                st_sh[h], ft_sh, rt_sh, h, 30)
            TSc_sh_v[h] = min(max_c, TSc_sh_)
            TSd_sh_v[h] = 0
            if TSc_sh_v[h] == 0.0:
                state_sh = 'standby'
            else:
                state_sh = 'charging'
        elif TSd_sh_ > TSc_sh_:
            max_d = self.mySHTank.max_energy_in_out(
                'discharging',
                prev_result['final_nodes_temp_sh'],
                st_sh[h], ft_sh, rt_sh, h, 30)
            TSd_sh_v[h] = min(max_d, TSd_sh_)
            TSc_sh_v[h] = 0.0
            if TSd_sh_v[h] == 0.0:
                state_sh = 'standby'
            else:
                state_sh = 'discharging'
        else:
            state_sh = 'standby'

        # use outputs as final set of node temps; the new version of new_nodes_temp 
        # has been modified to only output the final set of interest
        next_nodes_temp_dhw = self.myDHWTank.new_nodes_temp(
            state=state_dhw, nodes_temp = prev_result['final_nodes_temp_dhw'],
            source_temp=st_dhw[h], source_delta_t=0, flow_temp=ft_dhw,
            return_temp=rt_dhw, thermal_output=HPto_dhw_i[h], demand=hd_dhw_p[h],
            timestep = timestep+h, MTS=30, vol=vd_dhw_p[h])

        next_nodes_temp_sh = self.mySHTank.new_nodes_temp(
            state=state_sh, nodes_temp = prev_result['final_nodes_temp_sh'],
            source_temp=st_sh[h], source_delta_t=0, flow_temp=ft_sh,
            return_temp=rt_sh, thermal_output=HPto_sh_i[h], demand=hd_sh_p[h],
            timestep = timestep+h, MTS=30, vol=vd_sh_p[h]) 

        # when storing results, index to "h" for parameters/variables, and to "ts" for values from pre_calc
        """Results"""
        # results are for the second timestep
        ts = timestep + 1
        results = self.set_of_results()
 
        # Elec Demand for HP Operation Results   
        results['elec_demand']['elec_charge'] = round((HPto_dhw_i[h]/cop_dhw_p[h]),4) + round((HPto_sh_i[h]/cop_sh_p[h]),4)
        results['elec_demand']['pv_charge'] = round(HPtpvs_dhw_v[h]/cop_dhw_p[h],4) + round(HPtpvs_sh_v[h]/cop_sh_p[h],4)
        results['elec_demand']['wind_charge'] = round(HPtws_dhw_v[h]/cop_dhw_p[h],4) + round(HPtws_sh_v[h]/cop_sh_p[h],4)
        results['elec_demand']['grid_charge'] = round(HPtis_dhw_v[h]/cop_dhw_p[h],3) + round(HPtis_sh_v[h]/cop_sh_p[h],3)

        # Heat Demand results (pass from pre_calc) 
        results['heat_demand']['heat_dem_dhw'] = round(heat_dem_dhw[ts],2)
        results['heat_demand']['heat_dem_sh'] = round(heat_dem_sh[ts],2)
        results['heat_demand']['vol_dem_dhw'] = round(vol_dem_dhw[ts],2)
        results['heat_demand']['vol_dem_sh'] = round(vol_dem_sh[ts],2)

        # RES Generation Results (pass from pre_calc) 
        results['RES']['PV'] = round(pv_p[h],4)
        results['RES']['wind'] = round(wind_p[h],4)   # wind generation [kWh]
        results['RES']['RES_total'] = round(pv_p[h] + wind_p[h],4)
        results['RES']['HP'] = round(((HPtpvs_dhw_v[h]+HPtws_dhw_v[h])/cop_dhw_p[h] + (HPtpvs_sh_v[h]+HPtws_sh_v[h])/cop_sh_p[h]),2)     # RES generation used by heat pump [kWh]

        # Heat Pump Results (pass from pre_calc and some params)
        results['HP']['cop_dhw'] = round(cop_dhw[ts],2) # maybe instead do cop_dhw_[h]
        results['HP']['cop_sh'] = round(cop_sh[ts],2)   # maybe instead do cop_sh_[h]
        results['HP']['duty_dhw'] = self.duty_dhw   
        results['HP']['duty_sh'] = self.duty_sh     
        results['HP']['heat_to_dhw'] = round(HPto_dhw_i[h],3)
        results['HP']['heat_to_sh'] = round(HPto_sh_i[h],3)
        results['HP']['elec_usage_dhw'] = round((HPto_dhw_i[h]/cop_dhw_p[h]),4)
        results['HP']['elec_usage_sh'] = round((HPto_sh_i[h]/cop_sh_p[h]),4)
        results['HP']['pv_usage_dhw'] = round(HPtpvs_dhw_v[h]/cop_dhw_p[h],4)
        results['HP']['pv_usage_sh'] = round(HPtpvs_sh_v[h]/cop_sh_p[h],4)
        results['HP']['wind_usage_dhw'] = round(HPtws_dhw_v[h]/cop_dhw_p[h],4)
        results['HP']['wind_usage_sh'] = round(HPtws_sh_v[h]/cop_sh_p[h],4)
        results['HP']['grid_usage_dhw'] = round(HPtis_dhw_v[h]/cop_dhw_p[h],3)
        results['HP']['grid_usage_sh'] = round(HPtis_sh_v[h]/cop_sh_p[h],3)

        # TS results (pass from params) 
        results['TS']['charging_en_dhw'] = round(TSc_dhw_v[h],3)
        results['TS']['charging_en_sh'] = round(TSc_sh_v[h],3)
        results['TS']['discharging_en_dhw'] = round(TSd_dhw_v[h],3) 
        results['TS']['discharging_en_sh'] = round(TSd_sh_v[h],3)
        results['TS']['HP_to_dhw'] = (
            results['HP']['heat_to_dhw'])
        results['TS']['HP_to_sh'] = (
            results['HP']['heat_to_sh'])
        results['TS']['final_nodes_temp_dhw'] = np.round(next_nodes_temp_dhw,2) #! edited
        results['TS']['final_nodes_temp_sh'] = np.round(next_nodes_temp_sh,2)
        results['TS']['soc_dhw'] = np.round(soc_dhw_v[h],2)
        results['TS']['soc_sh'] = np.round(soc_sh_v[h],2)

        next_results = {
            'HPto_dhw': round(HPto_dhw_i[h],3),        'HPto_sh': round(HPto_sh_i[h],3),
            'HPtpvs_dhw': round(HPtpvs_dhw_v[h],3),    'HPtpvs_sh': round(HPtpvs_sh_v[h],3),
            'HPtws_dhw': round(HPtws_dhw_v[h],3),      'HPtws_sh': round(HPtws_sh_v[h],3),
            'HPtis_dhw': round(HPtis_dhw_v[h],3),      'HPtis_sh': round(HPtis_sh_v[h],3),
            'HP_status_dhw': HP_status_dhw_v[h],       'HP_status_sh': HP_status_sh_v[h],
            # 'HPt_var_dhw': round(HPt_var_dhw[h],2),  'HPt_var_sh': round(HPt_var_sh[h],3),
            'TSc_dhw': round(TSc_dhw_v[h],2),          'TSc_sh': round(TSc_sh_v[h],2),
            'TSd_dhw': round(TSd_dhw_v[h], 2),         'TSd_sh': round(TSd_sh_v[h],2),
            'state_dhw': state_dhw,         'state_sh': state_sh,
            'pv_cost': PVC_p[h],           
            'wind_cost': WC_p[h],
            'grid_cost': GC_p[h],
            'final_nodes_temp_dhw': np.round(next_nodes_temp_dhw,2),
            'final_nodes_temp_sh': np.round(next_nodes_temp_sh,2),
            }

        return {'results': results, 'next_results': next_results}

    def moving_horizon(self, pre_calc, first_ts, timesteps):
        
        final_ts = first_ts + timesteps

        # includes a progress bar, purely for aesthetics
        widgets = ['Running: MPC' + ' ', Percentage(), ' ',
                   Bar(marker=RotatingMarker(), left='[', right=']'),
                   ' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=timesteps)
        pbar.start()

        if final_ts > 17520:
            msg = f'The final timestep is {final_ts} which is beyond the end of the year (17520)'
            LOG.error(msg)
            raise ValueError(msg)

        # results will hold 
        results = []
        next_results = []

        # iterate through range of timesteps
        for timestep in tqdm(range(first_ts, final_ts-1)):
            ts = timestep - first_ts # in case not starting simulation in timestep 0 of year
            
            final_horizon_timestep = timestep + self.horizon
            if final_horizon_timestep > 17519:
                final_horizon_timestep = 17519

            # if first timestep, previous results are zero
            if ts == 0 or timestep == final_horizon_timestep - 1:   # if ts == 0 or timestep == 17518:
            
                #! removed PyLESA chunk of code here..
                prev_result = {
                    'HPto_dhw': 0,          'HPto_sh': 0,
                    'HPtpvs_dhw': 0,        'HPtpvs_sh': 0,
                    'HPtws_dhw': 0,         'HPtws_sh': 0,
                    'HPtis_dhw': 0,         'HPtis_sh': 0, 
                    'HP_status_dhw': 0,     'HP_status_sh': 0,
                    'TSc_dhw': 0,           'TSc_sh': 0,
                    'TSd_dhw': 0,           'TSd_sh': 0,
                    'final_nodes_temp_dhw': self.init_temps_dhw,
                    'final_nodes_temp_sh': self.init_temps_sh,
                    'state_dhw': 'standby', 'state_sh': 'standby',
                    'pv_cost': pre_calc['pv_prices'][ts],          
                    'wind_cost': pre_calc['wind_prices'][ts],
                    'grid_cost': pre_calc['grid_prices'][ts]}

                # generate an empty set of results and fill some
                res = self.set_of_results()
                res['TS']['final_nodes_temp_dhw'] = self.init_temps_dhw
                res['TS']['final_nodes_temp_sh'] = self.init_temps_sh
                res['HP']['heat_to_dhw'] = 0 
                res['HP']['heat_to_sh'] = 0 
                res['HP']['cop_dhw'] = pre_calc['cop_dhw'][ts]
                res['HP']['cop_sh'] = pre_calc['cop_sh'][ts]
                res['HP']['duty_dhw'] = self.duty_dhw
                res['HP']['duty_sh'] = self.duty_sh
                res['heat_demand']['heat_dem_dhw'] = pre_calc['heat_dem_dhw'][ts]
                res['heat_demand']['heat_dem_sh'] = pre_calc['heat_dem_sh'][ts]
                res['heat_demand']['vol_dem_dhw'] = pre_calc['vol_dem_dhw'][ts]
                res['heat_demand']['vol_dem_dhw'] = pre_calc['vol_dem_sh'][ts]

                # want to add an initial "buffer" to results for ts - this is ok
                results.append(res)
                next_results.append(prev_result) 

                # if first timestep, solve over the prediction horizon and store results
                if ts == 0:
                    r = self.solve(pre_calc, timestep, first_ts, final_horizon_timestep, prev_result)
                    results.append(r['results'])
                    next_results.append(r['next_results'])

            # for all timestep values except first
            else:
                # try pulling the previous result from next_results
                try:
                    prev_result = next_results[timestep - first_ts]
                    r = self.solve(pre_calc, timestep, first_ts, final_horizon_timestep, prev_result)
                    results.append(r['results'])
                    next_results.append(r['next_results'])

                    # print(f"\nSolar: {r['results']['RES']['PV']} kWh")
                    # print(f"Wind: {r['results']['RES']['wind']} kWh")
                    # print(f"HP Used: {r['results']['RES']['HP']}")
                    # print(f"kWhth to DHW: {r['results']['HP']['heat_to_dhw']}")
                    # print(f"kWhth to SH: {r['results']['HP']['heat_to_sh']}")
                    print(f"DHW node temperatures are: {r['results']['TS']['final_nodes_temp_dhw']}")
                    print(f"SH node temperatures are: {r['results']['TS']['final_nodes_temp_sh']}")
                    print(f"DHW SOC is: {r['results']['TS']['soc_dhw']}")
                    print(f"SH SOC is: {r['results']['TS']['soc_sh']}")

                    # print(r['results'])
                
                # if there isn't anything in next_results, assign empty prev_result
                except:
                    prev_result = {
                            'HPto_dhw': 0,          'HPto_sh': 0,
                            'HPtpvs_dhw': 0,        'HPtpvs_sh': 0,
                            'HPtws_dhw': 0,         'HPtws_sh': 0,
                            'HPtis_dhw': 0,         'HPtis_sh': 0, 
                            'HP_status_dhw': 0,     'HP_status_sh': 0,
                            'TSc_dhw': 0,           'TSc_sh': 0,
                            'TSd_dhw': 0,           'TSd_sh': 0,
                            'final_nodes_temp_dhw': self.init_temps_dhw,
                            'final_nodes_temp_sh': self.init_temps_sh,
                            'state_dhw': 'standby', 'state_sh': 'standby',
                            'pv_cost': pre_calc['pv_prices'][first_ts],      #are these the right call?    
                            'wind_cost': pre_calc['wind_prices'][first_ts],
                            'grid_cost': pre_calc['grid_prices'][first_ts]}
                    
                    r = self.solve(pre_calc, timestep, first_ts, final_horizon_timestep, prev_result)
                    results.append(r['results'])
                    next_results.append(r['next_results'])
                
            # update for progress bar
            pbar.update(timestep - first_ts + 1)
        # stop progress bar
        pbar.finish()

        # write the outputs to a pickle
        # file = self.root / OUTDIR / self.subname / 'outputs.pkl'
        # with open(file, 'wb') as output_file:
        #     pickle.dump(results, output_file,
        #                 protocol=pickle.HIGHEST_PROTOCOL)

        return results


    def set_of_results(self):
        """Creates and returns a nested dictionary for results.
        
        elec_demand -- electricity sources for heat pump
        heat_demand -- demands for heat/vol from thermal stores
        RES -- renewables generation
        HP -- heat pump performance, output, and usage of electricity sources
        TS -- thermal storage charging, discharging, 
        """

        #TODO remove redundant variables (low priority, get it to work first)
        #need to instead take values from variables in the solver?
        elec_demand = {'elec_charge': 0.0,      # [kwhe] electricity to run HP
                       'pv_charge': 0.0,        # [kWhe] PV electricity to run HP
                       'wind_charge': 0.0,      # [kWhe] wind electricity to run HP
                       'grid_charge': 0.0,      # [kWhe] grid electricity to run HP
                       }

        heat_demand = {'heat_dem_dhw': 0.0,     # [kWhth] DHW discharge heat demand
                       'heat_dem_sh': 0.0,      # [kWhth] SH discharge heat demand
                       'vol_dem_dhw': 0.0,      # [L] DHW discharge volume demand
                       'vol_dem_sh': 0.0,       # [L] SH discharge volume demand
                       }

        RES = {'PV': 0.0,                       # [kWhe] PV generation
               'wind': 0.0,                     # [kWhe] wind generation
               'RES_total': 0.0,                # [kWhe] total RES generation
               'HP': 0.0,                       # [kWhe] RES used by HP
               }

        HP = {'cop_dhw': 0.0,                   # COP if charging DHW tank
              'cop_sh': 0.0,                    # COP if charging SH tank
              'duty_dhw': 0.0,                  # [kWhth] duty if charging DHW tank
              'duty_sh': 0.0,                   # [kWhth] duty if charging SH tank
              'heat_to_dhw': 0.0,               # [kWhth] heat energy charged to DHW
              'heat_to_sh': 0.0,                # [kWhth] heat energy charged to SH
              'elec_usage_dhw': 0.0,            # [kWhe] electricity used to charge DHW
              'elec_usage_sh': 0.0,             # [kWhe] electricity used to charge SH
              'pv_usage_dhw': 0.0,              # [kWhe] PV electricity used to charge DHW
              'pv_usage_sh': 0.0,               # [kWhe] PV electricity used to charge SH
              'wind_usage_dhw': 0.0,            # [kWhe] wind electricity used to charge DHW
              'wind_usage_sh': 0.0,             # [kWhe] wind electricity used to charge SH
              'grid_usage_dhw': 0.0,            # [kWhe] grid electricity used to charge DHW
              'grid_usage_sh': 0.0,             # [kWhe] grid electricity used to charge SH
              }

        TS = {'charging_en_dhw': 0.0,           # [kWhth] energy charged into DHW tank
              'charging_en_sh': 0.0,            # [kWhth] energy charged into SH tank
              'discharging_en_dhw': 0.0,        # [kWhth] energy discharged from DHW tank
              'discharging_en_sh': 0.0,         # [kWhth] energy discharged from SH tank
              'HP_to_dhw': 0.0,                 # [kWhth] energy from HP to DHW tank 
              'HP_to_sh': 0.0,                  # [kWhth] energy from HP to SH tank
              'final_nodes_temp_dhw': 0.0,      # [degC] final temps for DHW tank nodes
              'final_nodes_temp_sh': 0.0,       # [degC] final temps for SH tank nodes
              'soc_dhw': 0.0,
              'soc_sh': 0.0,
              }

        outputs = {'elec_demand': elec_demand,
                   'heat_demand': heat_demand,
                   'RES': RES,
                   'HP': HP,
                   'TS': TS
                   }

        return outputs