from IPython import get_ipython
ipython = get_ipython()
# ipython.magic('reset -f')
ipython.run_line_magic('reset','-sf')

import numpy as np
import hot_water_tank_5_ws_vol
import hot_water_tank_5_ws_sh

#%% Input Data
day = 25 # start day of month
mth = 3 # start month
doy = 84 # start day of year
ts = 10*48 # number of modelled timesteps
cyc = 0 # 
cycS = 0 #
TankT = np.array([52.,50.8,50.6,34.5]) # Starting DHW tank temperature

T_min = 47. # Minimum DHW tank top sensor temperature before forcing a charge
T_lim = 50. # Minimum DHW tank top sensor temperature before allowing a charge
T_max = 52. # DHW tank bottom sensor temperature at which charging stops
ST_min = 45. # Minimum SH tank top sensor temperature before forcing a charge
ST_lim = 47. # Minimum SH tank top sensor temperature before allowing a charge
ST_max = 50. # SH tank bottom sensor temperature at which charging stops
P_lim = 4. # PV forced charging minimum kW 

STankT = np.array([50.,50.,50.,50.,50.]) # Starting SH tank temperature

#%% Base Data
doyt = np.array([0,31,59,90,120,151,181,212,243,273,303,334,365])
SrcT = np.array([9.3,8.7,8.4,9.1,10.2,11.8,13.5,14.1,13.9,13.1,12.0,10.6]) # Heat Pump Source Temperature (Water)
HW_Act = np.loadtxt('c:/data/ws_dhw_tot.csv', delimiter=',') # HW Data
PV_Act = np.loadtxt('c:/data/ws_pv_act.csv', delimiter=',') # PV Data
SH_Act = np.loadtxt('c:/data/ws_sh_act.csv', delimiter=',') # SH Data

soc_45 = np.zeros(ts+1)
soc_45[0] = hot_water_tank_5_ws_vol.HotWaterTank().max_energy_in_out( # DHW State of charge based on 45C minimum
    'discharging', TankT,
    0., 45., 15., 0, 30, 1)

soc_S = np.zeros(ts+1)
soc_S[0] = hot_water_tank_5_ws_sh.HotWaterTank().max_energy_in_out( # SH State of charge based on 45C minimum
    'discharging', STankT,
    0., 45., 35., 0, 30, 1)

#%% Initialise Outputs
telch = 0.
thwch = 0.
tshch = 0.
tpv = 0.
cnt = 0

FTankT = []
FSTankT = []

Res = np.zeros((ts,8))

#%% DHW Tank Temperature Model
def temp(delta, prev_temp, vol):    
    
    if delta > 0.:
        new_temp = hot_water_tank_5_ws_vol.HotWaterTank().new_nodes_temp(
            'discharging', prev_temp, 55., 40., 45.,
            15., 0., delta, 0., 1, 30, vol)[4]
        
    elif delta < 0.:
        new_temp = hot_water_tank_5_ws_vol.HotWaterTank().new_nodes_temp(
            'charging', prev_temp, 55., TankT[0] - TankT[3], 50.,
            TankT[3], -delta, 0., 0., 1, 30, vol)[4]
        
    else:
        new_temp = prev_temp
    
    new_temp = hot_water_tank_5_ws_vol.HotWaterTank().new_nodes_temp(   # Heat loss only
            'charging', new_temp, 60., 40., 50.,
            15., 0., 0., 0, 3, 30, vol)[4]
    
    return new_temp

#%% SH Tank Temperature Model
def tempS(deltaS, prev_tempS):    
    
    if deltaS > 0.:
        new_tempS = hot_water_tank_5_ws_sh.HotWaterTank().new_nodes_temp(
            'discharging', prev_tempS, 55., 10., 45.,
            35., 0., deltaS, 0., 1, 30)[4]
        
    elif deltaS < 0.:
        new_tempS = hot_water_tank_5_ws_sh.HotWaterTank().new_nodes_temp(
            'charging', prev_tempS, 55., STankT[0] - STankT[3], 50.,
            STankT[3], -deltaS, 0., 0., 1, 30)[4]
        
    else:
        new_tempS = prev_tempS
    
    new_tempS = hot_water_tank_5_ws_sh.HotWaterTank().new_nodes_temp(   # Heat loss only
            'charging', new_tempS, 60., 40., 50.,
            15., 0., 0., 0, 3, 30)[4]
    
    return new_tempS


#%% Timestep Model
for t in range(ts):
    chrg = 0.
    chrgS = 0.
    
    if t>0 and np.remainder(t,48) == 0.:
        doy = doy + 1
        
    if doy > doyt[mth]:
        mth = mth + 1
        
    cop = 0.7 * (8.77 -
           0.15 * (55. - SrcT[mth]) +
           0.000734 * (55. - SrcT[mth]) ** 2
           )

    pv_act = PV_Act[t]
    sh_act = SH_Act[t]

# Tank Model
    dem = (HW_Act[t] * 4.181 * 40.) / 3600.
    old_temp = np.copy(TankT)
    TankT = temp(dem, old_temp, HW_Act[t])
    STankT = tempS(sh_act,STankT)
    
# Control Logic

    # DHW (1st Priority)
    if (TankT[0] < T_min or cyc == 1): # Force or continue DHW charging
        maxc = hot_water_tank_5_ws_vol.HotWaterTank().max_energy_in_out(
            'charging', TankT,
            55., 0., 0., 0, 30, 1)
        
        chrg = min(7., maxc)
        
        TankT = temp(-chrg,TankT, 0.)
        cyc = 1
        print(t,1,chrg)
        
    elif pv_act > (P_lim/2.) and TankT[0] < T_lim: # Force DHW charge if PV above threshold
        maxc = hot_water_tank_5_ws_vol.HotWaterTank().max_energy_in_out(
            'charging', TankT,
            55., 0., 0., 0, 30, 1)
        
        chrg = min(7., maxc)
        
        TankT = temp(-chrg, TankT, 0.)
        
        cyc = 1
        print(t,2,chrg)
        
    if TankT[2] > T_max and cyc == 1: # Stop DHW charge if mid temperature above limit
        cyc = 0
        
    # SH (2nd Priority)
    if (cyc == 0 and STankT[1] < ST_min) or cycS == 1:
        maxS = hot_water_tank_5_ws_sh.HotWaterTank().max_energy_in_out(
            'charging', STankT,
            55., 0., 0., 0, 30, 1)
        
        chrgS = min(7., maxS)
        
        STankT = tempS(-chrgS, STankT)
        
        cycS = 1
        
    elif pv_act > (P_lim/2.) and STankT[1] < ST_lim: # Force DHW charge if PV above threshold
        maxS = hot_water_tank_5_ws_sh.HotWaterTank().max_energy_in_out(
            'charging', STankT,
            55., 0., 0., 0, 30, 1)
        
        chrgS = min(7., maxS)
        
        STankT = tempS(-chrgS, STankT)
        
        cycS = 1
    
    if cycS == 1 and STankT[3] > ST_max: # Stop SH charge if mid temperature above limit
        cycS = 0
        
    # Results
        
    soc_45[t+1] = hot_water_tank_5_ws_vol.HotWaterTank().max_energy_in_out(
        'discharging', TankT,
        0., 45., 15., 0, 30, 1)
    
    soc_S[t+1] = hot_water_tank_5_ws_sh.HotWaterTank().max_energy_in_out(
        'discharging', STankT,
        0., 45., 35., 0, 30, 1)
    
    thwch += chrg
    tshch += chrgS    
    telch += chrg/cop + chrgS/cop
    tpv += min(pv_act,chrg/cop + chrgS/cop)
    
    if t == 0:
        FTankT = np.reshape(TankT,(1,-1))
    else:
        FTankT = np.vstack((FTankT,np.reshape(TankT,(1,-1))))
        
    if t == 0:
        FSTankT = np.reshape(STankT,(1,-1))
    else:
        FSTankT = np.vstack((FSTankT,np.reshape(STankT,(1,-1))))
        
    # Results Output
    Res[t,0] = pv_act # PV output (kWh)
    Res[t,1] = chrg/cop # DHW charge electrical input (kWh)
    Res[t,2] = chrgS/cop # SH charge electrical input (kWh)
    Res[t,3] = min(pv_act, chrg/cop + chrgS/cop) # PV to charging
    Res[t,4] = soc_45[t+1] # DHW store charge
    Res[t,5] = TankT[0] # DHW tank top sensor temperature
    Res[t,6] = soc_S[t+1] # SH store charge
    Res[t,7] = STankT[1] # SH tank top sensor temperature
    
    
    

