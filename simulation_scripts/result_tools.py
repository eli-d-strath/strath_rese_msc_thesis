"""
Functions called during result calculation and storing in main.py. 
"""

from IPython import get_ipython
ipython = get_ipython()
# ipython.magic('reset -f')
ipython.run_line_magic('reset','-sf')

import os
import math
import numpy as np
import pandas as pd
from sim_tools import carbon_emissions

def store_timestep_results(TimestepRes, t, ts, 
                           elec_used, charge_dhw, charge_sh,
                           dhw_cop, sh_cop, soc_dhw, soc_sh, 
                           pv_en, wind_en,
                           pv_used, wind_used, grid_used, 
                           pv_cost, wind_cost, grid_cost):
    
    TimestepRes[t,0] = ts                       # timestep
    TimestepRes[t,1] = elec_used                # combined electrical charge [kWhe]
    TimestepRes[t,2] = charge_dhw/dhw_cop           # DHW charge electrical input [kWhe]
    TimestepRes[t,3] = charge_sh/sh_cop            # SH charge electrical input [kWhe]
    TimestepRes[t,4] = charge_dhw + charge_sh   # combined thermal charge [kWhth]
    TimestepRes[t,5] = charge_dhw               # DHW thermal charge [kWhth]
    TimestepRes[t,6] = charge_sh                # SH thermal charge [kWhth]
    TimestepRes[t,7] = soc_dhw             # DHW state of charge assuming 45C flow [kWhth]
    TimestepRes[t,8] = soc_sh              # SH state of charge assuming 45C flow [kWhth]
    TimestepRes[t,9] = pv_en                   # total PV generation [kWhe]
    TimestepRes[t,10] = wind_en                 # net site renewables [kWh]
    TimestepRes[t,11] = pv_used                  # PV used to charge [kWhe]
    TimestepRes[t,12] = wind_used                # wind used to charge [kWhe]
    TimestepRes[t,13] = grid_used                # grid used to charge [kWhe]
    TimestepRes[t,14] = pv_cost + wind_cost + grid_cost # total charging cost [£]
    TimestepRes[t,15] = pv_cost                  # PV cost for charge [£]
    TimestepRes[t,16] = wind_cost                # wind cost for charge [£]
    TimestepRes[t,17] = grid_cost                # grid cost for charge [£]

    return TimestepRes

def store_node_temps(Temps, t, ts,
                     dhw_tank_T, sh_tank_T,
                     dhw_disch_vol, sh_disch_vol,
                     dhw_en_dem, sh_en_dem, 
                     charge_dhw, charge_sh):

    Temps[t,0] = ts
    Temps[t,1] = dhw_tank_T[0]
    Temps[t,2] = dhw_tank_T[1]
    Temps[t,3] = dhw_tank_T[2]
    Temps[t,4] = dhw_tank_T[3]
    Temps[t,5] = dhw_tank_T[4]
    Temps[t,6] = dhw_disch_vol
    Temps[t,7] = dhw_en_dem
    Temps[t,8] = charge_dhw
    Temps[t,9] = None
    Temps[t,10] = sh_tank_T[0]
    Temps[t,11] = sh_tank_T[1]
    Temps[t,12] = sh_tank_T[2]
    Temps[t,13] = sh_tank_T[3]
    Temps[t,14] = sh_tank_T[4]
    Temps[t,15] = sh_disch_vol
    Temps[t,16] = sh_en_dem
    Temps[t,17] = charge_sh

    return Temps

def calculate_kpis(KPIs, tot_cost, tot_elec_ch, 
                    tot_therm_ch, tot_therm_disch,
                    tot_pv, tot_wind, tot_grid,
                    tot_dhw_elec_ch, tot_dhw_therm_ch,
                    tot_dhw_therm_disch, tot_sh_elec_ch,
                    tot_sh_therm_ch, tot_sh_therm_disch):
    
    KPIs[0,0] = tot_cost / tot_elec_ch                      # LCOE electricity [£/kWhe]
    KPIs[0,1] = tot_cost / tot_therm_disch                  # OCOH heat [£/kWhth]
    KPIs[0,2] = ((tot_pv + tot_wind) / tot_elec_ch) * 100   # renewable electricity share [%]
    KPIs[0,3] = (tot_grid / tot_elec_ch) * 100              # grid electricity share [%]
    KPIs[0,4] = tot_cost                                    # total electricity cost [£]
    KPIs[0,5] = carbon_emissions(tot_pv,tot_wind,tot_grid)  # total carbon emissions [kg]
    KPIs[0,6] = KPIs[0,5]/tot_therm_disch*1000              # heat carbon intensity [gCO2eq/kWhth] 
    KPIs[0,7] = None
    KPIs[0,8] = tot_elec_ch                                 # total electricity used [kWhe]
    KPIs[0,9] = tot_therm_ch                                # total thermal charge [kWhth]
    KPIs[0,10] = tot_therm_disch                            # total thermal discharge [kWhth]
    KPIs[0,11] = tot_dhw_elec_ch                            # total DHW electrical charge [kWhe]
    KPIs[0,12] = tot_dhw_therm_ch                           # total DHW thermal charge [kWhth]
    KPIs[0,13] = tot_dhw_therm_disch                        # total DHW thermal discharge [kWhth]
    KPIs[0,14] = tot_sh_elec_ch                             # total SH electrical charge [kWhe]
    KPIs[0,15] = tot_sh_therm_ch                            # total SH thermal charge [kWhth]
    KPIs[0,16] = tot_sh_therm_disch                         # total SH thermal discharge [kWhth]
    KPIs[0,17] = tot_pv                                     # total PV used [kWh]
    KPIs[0,18] = tot_wind                                   # total wind used [kWh]
    KPIs[0,19] = tot_grid                                   # total grid used [kWh]

    return KPIs

def export_results(results_path, results_file, temps_file, kpis_file,
                   TimestepRes, Temps, KPIs):
    
    # Remove Any Files with Same Name
    try:
        os.remove(results_path + results_file)
    except FileNotFoundError:
        pass
    try:
        os.remove(results_path + temps_file)
    except FileNotFoundError:
        pass
    try:
        os.remove(results_path + kpis_file)
    except FileNotFoundError:
        pass

    # Create Headers
    results_header = np.array(["Timestep", "Combined Electrical Input [kWh]", "DHW Electrical Input [kWh]",
                               "SH Electrical Input [kWh]", "Combined Thermal Charge [kWhth]", 
                               "DHW Thermal Charge [kWhth]", "SH Thermal Charge [kWhth]", "DHW SOC [kWhth]",
                               "SH SOC [kWhth]", "PV Generation [kWh]", "Net Wind Generation [kWh]",
                               "PV Used [kWh]", "Wind Used [kWh]", "Grid Used [kWh]", "Electricity Cost [£]",
                               "PV Cost [£]", "Wind Cost [£]", "Grid Cost [£]"])
    
    kpis_header = np.array(["LCOE Electricity [£/kWhe]", "OCOH Provided [£/kWhth]", "Renewable Electricity Share [%]",
                        "Grid Electricity Share [%]", "Total Electricity Cost [£]", "Total Carbon Emissions [kg]",
                        "Heat Carbon Intensity [gCO2eq/kWhth]", None,
                        "Electrical Charge [kWh]", "Thermal Charge [kWh]", "Thermal Discharge [kWh]",
                        "DHW Electrical Charge [kWh]", "DHW Thermal Charge [kWh]", "DHW Thermal Discharge [kWh]",
                        "SH Electrical Charge [kWh]", "SH Thermal Charge [kWh]", "SH Thermal Discharge [kWh]",
                        "PV Used [kWh]", "Wind Used [kWh]","Grid Used [kWh]"])
    
    temps_header = np.array(["Timestep","DHW Node 1", "DHW Node 2","DHW Node 3","DHW Node 4","DHW Node 5",
                         "DHW Discharge Vol [L]","DHW Discharge Energy [kWh]", "DHW Charge Energy [kWh]", None,
                         "SH Node 1", "SH Node 2","SH Node 3","SH Node 4","SH Node 5",
                         "SH Discharge Vol [L]","SH Discharge Energy [kWh]", "SH Charge Energy [kWh]"])
    
    # Export Results
    df_results = pd.DataFrame(TimestepRes)
    df_temps = pd.DataFrame(Temps)
    df_kpis = pd.DataFrame(KPIs)

    df_results.to_csv(results_path + results_file, sep=',',index=False,header=results_header,encoding='utf-8')
    df_temps.to_csv(results_path + temps_file, sep=',',index=False,header=temps_header,encoding='utf-8')
    df_kpis.to_csv(results_path + kpis_file, sep=',',index=False,header=kpis_header,encoding='utf-8')

    return
