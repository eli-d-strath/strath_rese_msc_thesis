"""
Opportunistic PV Charging Rule Based Control (opportunistic_pv_rbc) for Woodside.

The design of these controls are based on Graeme Flett's find_ws_dt_0 script.

They correspond to the existing control paradigm at Woodside, with the addition
of opportunistic charging when PV generation exceeds a preset threshold.

No prioritized use of Wind is implemented. 
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


class OppPVRBC(object):

    def __init__(self, pv_enabled, wind_enabled,
                 dhw_min_T, dhw_lim_T, dhw_max_T,
                 sh_min_T, sh_lim_T, sh_max_T,
                 dhw_source_T, sh_source_T, water_temp,
                 dhw_flow_T, sh_flow_T):

        self.pv_enabled = pv_enabled
        self.wind_enabled = wind_enabled

        self.dhw_min_T = dhw_min_T
        self.dhw_lim_T = dhw_lim_T
        self.dhw_max_T = dhw_max_T

        self.sh_min_T = sh_min_T
        self.sh_lim_T = sh_lim_T
        self.sh_max_T = sh_max_T

        self.dhw_source_T = dhw_source_T
        self.sh_source_T = sh_source_T

        self.water_temp = water_temp
        self.dhw_flow_T = dhw_flow_T
        self.sh_flow_T = sh_flow_T

    def dhw_charging(self, dhw_tank_T, cyc_dhw):
        
        charge_dhw = 0
        dhw_cop = hp_cop(self.water_temp,self.dhw_source_T)

        # Force DHW charge when PV or wind exceeds threshold and top node is below limit temp
        if (self.pv_enabled) and dhw_tank_T[0] < self.dhw_lim_T:
            
            max_charge = dhw_tank.HotWaterTank().max_energy_in_out(
                state='charging', nodes_temp=dhw_tank_T,
                source_temp = self.dhw_source_T, flow_temp=0., return_temp=0., 
                timestep=0, MTS=30)
            
            #cap charge at 7kWh because Emoncms shows DHW charge power does not exceed 14kW...
            charge_dhw = min(7., max_charge)
            dhw_tank_T = temp_dhw(-charge_dhw, dhw_tank_T, 0, self.dhw_source_T, self.dhw_flow_T)
            cyc_dhw = 1
        
        # Force DHW charge when below minimum temp, or previously charging
        elif ((dhw_tank_T[2] < self.dhw_min_T or cyc_dhw == 1) 
              and dhw_tank_T[0] < self.dhw_max_T):
            
            max_charge = dhw_tank.HotWaterTank().max_energy_in_out(
                state='charging', nodes_temp=dhw_tank_T,
                source_temp=self.dhw_source_T, flow_temp=0., return_temp=0., 
                timestep=0, MTS=30)

            #cap charge at 7kWh because Emoncms shows DHW charge power does not exceed 14kW...
            charge_dhw = min(7., max_charge)
            dhw_tank_T = temp_dhw(-charge_dhw, dhw_tank_T, 0, self.dhw_source_T, self.dhw_flow_T)
            cyc_dhw = 1

        if dhw_tank_T[0] >= self.dhw_max_T: 
            cyc_dhw = 0

        return dhw_cop, charge_dhw, dhw_tank_T, cyc_dhw

    def sh_charging(self, sh_tank_T, cyc_sh, charge_dhw):

        charge_sh = 0
        sh_cop = hp_cop(self.water_temp,self.sh_source_T)
        
        # Force SH charge when PV or wind exceeds threshold and top node is below limit temp
        if (self.pv_enabled and charge_dhw == 0 and sh_tank_T[0] < self.sh_lim_T):
            
            max_charge = sh_tank.HotWaterTank().max_energy_in_out(
                state='charging', nodes_temp=sh_tank_T,
                source_temp = self.sh_source_T, flow_temp=0., return_temp=0., 
                timestep=0, MTS=30)
            
            charge_sh = min(8.5, max_charge)
            sh_tank_T = temp_sh(-charge_sh, sh_tank_T, 0, self.sh_source_T, self.sh_flow_T)
            cyc_sh = 1
        
        # Force SH charge when below minimum temp, or previously charging
        elif ((sh_tank_T[0] < self.sh_min_T or cyc_sh == 1) 
              and charge_dhw == 0 and sh_tank_T[0] < self.sh_max_T):

            max_charge = sh_tank.HotWaterTank().max_energy_in_out(
                state='charging', nodes_temp=sh_tank_T,
                source_temp=self.sh_source_T, flow_temp=0., return_temp=0., 
                timestep=0, MTS=30)
            
            # cap charge at 8.5kWh because Emoncms shows SH charge power does not exceed 17kW...
            charge_sh = min(8.5, max_charge)
            sh_tank_T = temp_sh(-charge_sh, sh_tank_T, 0, self.sh_source_T, self.sh_flow_T)
            cyc_sh = 1

        if (sh_tank_T[0] >= self.sh_max_T): 
            cyc_sh = 0
        
        return sh_cop, charge_sh, sh_tank_T, cyc_sh