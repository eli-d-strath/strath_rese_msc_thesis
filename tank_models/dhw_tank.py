"""
Modelling the Domestic Hot Water (DHW) water storage tank at Woodside.

Based on the water tank models "hot_water_tank_5_ws_vol.py" provided by Graeme Flett
and "hot_water_tank.py" from PyLESA, created by Andrew Lyden.

Modifications made for readability, unit adjustments, and control logic.
"""
import os
import pandas as pd
import math
import numpy as np

from scipy.integrate import odeint

class HotWaterTank(object):

    def __init__(self):

        self.capacity = 1           # [m^3] 
        self.number_nodes = 5       # 5 sensors in tank
        
        self.UA = np.array([2.5,2.0,2.5,8,30]) * 0.5
        self.node_masses = np.array([150,200,250,275,125])

        self.node_list = list(range(self.number_nodes))         # [0, 1, 2, 3, 4]

        self.insulation_k_value = 0.025             # [W/m-C] polyurethane 
        self.ambient_temp = 28.                     # [degC] tank is inside
        self.cp_water = 4180.                       # [J/kg-C] water specific heat
        
    def _discharging_function(self, state, nodes_temp, flow_temp):
        """ Determine discharging status of each node

        Arguments:
            state {str} -- state of the water tank (charging or discharging)
            nodes_temp {np.array} -- list of node temperatures temperatures [degC]
            flow_temp {float} -- required SH flow temperature [degC]

        Returns:
            function {dict} -- dict with node number as key and charging status as value       
        """

        function = {}

        if state.lower() == 'discharging':

            for node in self.node_list:

                # if we are looking at the top node,
                # and if the flow temp is below this node's temp
                if node == 0 and flow_temp <= nodes_temp[0]:
                    function[node] = 1

                # if we are looking at the top node,
                # and if the flow temp is above this node's temp
                elif node == 0 and flow_temp >= nodes_temp[0]:
                    function[node] = 0

                # if this node's temp exceeds the flow temp & the previous node did not
                elif nodes_temp[node-1] <= flow_temp < nodes_temp[node]:
                    function[node] = 1

                # for out of bounds nodes, shouldnt occur
                elif node < 0 or node == self.number_nodes + 1:
                    function[node] = 0

                else:
                    function[node] = 0

        elif state.lower() == 'charging' or state.lower() == 'standby':
            for node in self.node_list:
                function[node] = 0

        return function

    def _discharging_node_in(self, state, nodes_temp, flow_temp):
        """Inlet flow from SH loop to bottom node when tank is discharging

        Arguments:
            state {str} -- state of the water tank (charging or discharging)
            nodes_temp {np.array} -- list of node temperatures temperatures [degC]
            flow_temp {float} -- required SH flow temperature [degC]

        Returns:
            function {dict} -- dict with node number as key and charging status as value
        """

        function = {}
        bottom_node = self.number_nodes - 1

        # create list of discharging status from the _discharging_function
        df = self._discharging_function(state, nodes_temp, flow_temp)
        df_list = []
        for i in self.node_list:
            df_list.append(df[i])
            
        #if any node is discharging, assign bottom node to recharge
        if 1 in df_list:
            for node in self.node_list:
                if node == bottom_node:
                    function[node] = 1
                else:
                    function[node] = 0
        
        #if no nodes are discharging, don't refill the bottom node
        else:
            for node in self.node_list:
                function[node] = 0

        return function

    def _charging_function(self, state, nodes_temp, source_temp):
        """ Determines which node receives the charging water from the HP.

        If the entering mass exceeds the node volume, then the next node is also charged

        Arguments:
            state {str} -- state of the water tank (charging or discharging)
            nodes_temp {np.array} -- list of node temperatures temperatures [degC]
            source_temp {float} -- temperature of HP water entering tank [degC]

        Returns:
            function {dict} -- dict with node number as key and charging status as value
        """

        function = {}
        if state == 'charging':
            
            #! Modified - heat enters via heat exchanger in the 4th node
            for node in self.node_list:
                if node == 3 and source_temp >= nodes_temp[node]:
                    function[node] = 1
                # elif node == 3 and source_temp <= nodes_temp[node]:
                #     function[node] = 0
                # elif node == 2 and nodes_temp[node] <= source_temp <= nodes_temp[node+1]:
                #     function[node] = 1
                # elif node == 1 and nodes_temp[node] <= source_temp <= nodes_temp[node+1]:
                #     function[node] = 1
                # elif node == 0 and nodes_temp[node] <= source_temp <= nodes_temp[node+1]:
                    # function[node] = 1
                else:
                    function[node] = 0

        elif state == 'discharging' or 'standby':
            for node in self.node_list:
                function[node] = 0

        # print(f"Charging function: {function}")
        return function

    def _charging_node_out(self, state, nodes_temp, source_temp):
        """ Schedules outlet flow to HP from bottom node when tank is charging.

        Arguments:
            state {str} -- state of the water tank (charging or discharging)

        Returns:
            function {dict} -- dict with node number as key and charging status as value
        """
        
        function = {}
        bottom_node = self.number_nodes - 1
        
        #identify which node is the charging node
        cf = self._charging_function(state, nodes_temp, source_temp)
        cf_list = []
        for i in range(self.number_nodes):
            cf_list.append(cf[i])

        if 1 in cf_list:
            for n in range(self.number_nodes):
                if cf[n] == 1:
                    node_charging = n
        else:
            node_charging = bottom_node + 1

        for node in self.node_list:

            """Solution 1 - mixing through all nodes - same as original"""
            # in this mixing version, the charging node flows outward; previous version used node == bottom_node
            if state == 'charging' and node == node_charging:
                function[node] = 1
            else:
                function[node] = 0

            """Solution 2 - flow upwards from charging node and out"""
            # # in this upwards flow version, mass flows out of the top node top node "loses" heat outwards, not the bottom
            # if state == 'charging' and node == 0:
            #     function[node] = 1  
            # else:
            #     function[node] = 0

        return function

    def _mixing_function(self, state, node, nodes_temp,
                        source_temp, flow_temp):
        """ Assigns mixing function values for the node of interest to enable heat & mass flow calculations.

        Arguments:
            state {str} -- state of the water tank (charging or discharging)
            node {int} -- node of interest
            nodes_temp {np.array} -- list of node temperatures temperatures [degC]
            source_temp {float} -- temperature of HP water entering tank [degC]
            flow_temp {float} -- required SH flow temperature [degC]
            source {int} -- the source of charged water

        Returns:
            mf {dict} -- mixing function with Fcnt, Fdnt, Fcnb, Fdnt keys and 0/1 values
        """

        bottom_node = self.number_nodes - 1

        cf = self._charging_function(state, nodes_temp, source_temp)
        cf_list = []
        for i in range(self.number_nodes):
            cf_list.append(cf[i])

        df = self._discharging_function(state, nodes_temp, flow_temp)
        df_list = []
        for i in range(self.number_nodes):
            df_list.append(df[i])

        mf = {}
        
        if 1 in cf_list:
            for n in range(self.number_nodes):
                if cf[n] == 1:
                    node_charging = n
        else:
            node_charging = bottom_node + 1
            check = 'not charging'
             
        if 1 in df_list:
            for n in range(self.number_nodes):
                if df[n] == 1:
                    node_discharging = n
        else:
            node_discharging = bottom_node + 1
            check = 'not discharging'

        # [Fcnt] --> signals Q into node from above (charging) Coeff B
        # [Fdnt] --> signals Q from node to above (discharging) Coeff A
        # [Fcnb] --> signals Q from node to below (charging) Coeff A
        # [Fdnb] --> signals Q into node from below (discharging) - Coeff C

        #when there is no charging, node_charging = 
        if state == 'charging' and check != 'not charging':
            
            """Solution 1 - mixing through all nodes, 'out' of charging node"""

            if node <= 0 or node > bottom_node: #top node or unreal - nothing above
                mf['Fcnt'] = 0 
                mf['Fdnt'] = 0
            elif node <= node_charging: #is or above charging node - receives from above, gives to above
                mf['Fcnt'] = 1 
                mf['Fdnt'] = 1
            else: #is bottom node - no interaction with above
                mf['Fcnt'] = 0 
                mf['Fdnt'] = 0

            if node < 0 or node >= bottom_node: #node bottom or unreal - nothing below
                mf['Fcnb'] = 0 
                mf['Fdnb'] = 0
            elif node == 0 or node < node_charging: #top node or above charging node - gives to below, receives from below
                mf['Fcnb'] = 1
                mf['Fdnb'] = 1
            else:   #charging node - doesn't interact with below
                mf['Fcnb'] = 0  #was 1 in unmodified Sol1
                mf['Fdnb'] = 0

            """Solution 2 - flow upwards from charging node and 'out' of top node"""

            # if node <= 0 or node > node_charging: #top, below charging node - no interaction with above
            #     mf['Fcnt'] = 0
            #     mf['Fdnt'] = 0
            # elif node <= node_charging: #is or above charging node - give to above
            #     mf['Fcnt'] = 0
            #     mf['Fdnt'] = 1
            # else:
            #     mf['Fcnt'] = 0
            #     mf['Fdnt'] = 0

            # if node < 0 or node >= node_charging: #charging node, below, or unreal node - no interaction with below
            #     mf['Fcnb'] = 0
            #     mf['Fdnb'] = 0
            # elif node == 0 or node < node_charging: #top or above charging node - receives from below
            #     mf['Fcnb'] = 0
            #     mf['Fdnb'] = 1
            # else:
            #     mf['Fcnb'] = 0
            #     mf['Fdnb'] = 0

        elif state == 'discharging' and check != 'not discharging':
            # [Fcnt] --> signals Q into node from above (charging) Coeff B
            # [Fdnt] --> signals Q from node to above (discharging) Coeff A
            
            if node <= 0 or node > bottom_node: #top or unreal
                mf['Fcnt'] = 0
                mf['Fdnt'] = 0
            else:
                mf['Fcnt'] = 0
                mf['Fdnt'] = 1 

            # [Fcnb] --> signals Q from node to below (charging) Coeff A
            # [Fdnb] --> signals Q into node from below (discharging) - Coeff C
            if node >= bottom_node or node < 0: #bottom or unreal
                mf['Fcnb'] = 0
                mf['Fdnb'] = 0
            else:
                mf['Fcnb'] = 0
                mf['Fdnb'] = 1

        # standby
        else:
                mf['Fcnt'] = 0
                mf['Fdnt'] = 0
                mf['Fcnb'] = 0
                mf['Fdnb'] = 0

        return mf

    def _connection_losses(self):
        """Calculates heat losses due to tank connections

        Arguments:

        Returns:
            loss {dict} -- heat losses [W] 
        """

        tank_opening = 5                        # number of openings (e.g. thermostat pocket) 
        tank_opening_diameter = 35.             # [millimeters] for node temp sensors
        uninsulated_connections = 0             # number of unins. pipes or fittings (e.g. PTR valve)
        uninsulated_connections_diameter = 35.  # [millimeters]
        insulated_connections = 2.              # number of insulated pipes/fittings
        insulated_connections_diameter = 50.8   # [millimeters] for inlet/outlet

        # 27, 5, and 3.5 are empirical approximations used to arrive at a kWh/day value
        # loss divided by 0.024 to convert from kWh/day to W  
        # diameters are converted from millimeters to meters
        
        tank_opening_loss = (                               # Area * num_openings * 27 [kWh/day] * 1000 [W/kW] * 1/24 [day/h]
            (((tank_opening_diameter * 0.001) ** 2)/4) * math.pi * 
            tank_opening * 27 / (0.024))  # [W]


        uninsulated_connections_loss = (                    # pipe_diam * num_openings * 5 [kWh/day] * 1000 [W/kW] * 1/24 [day/h] 
            uninsulated_connections_diameter * 0.001 *
            uninsulated_connections * 5 / 0.024)        # [W]

        insulated_connections_loss = (                      # pipe_diam * num_openings * 3.5 [kWh/day] * 1000 [W/kW] * 1/24 [day/h] 
            insulated_connections_diameter * 0.001 *
            insulated_connections * 3.5 / 0.024)        # [W]

        loss = (
            tank_opening_loss +
            uninsulated_connections_loss +
            insulated_connections_loss)                 # [W]

        return loss

    def _charging_mass(self, thermal_output, source_temp, temp_node_charging):
        """ Calculate the mass flow into the thermal store during charging 

        Arguments: 
            thermal_output {float} -- heat injected from HP to tank [kJ] 
            source_temp {float} -- temperature of HP water entering tank [degC]
            source_delta_t {float} -- [degC]
            return_temp {float} -- water return temperature after providing heat [degC]
            flow_temp {float} -- required SH flow temperature [degC]
            demand {float} -- demand for space heating [kJ]
            temp_node_charging {float} -- temperature at charging node of tank [degC]

        Returns:
            ts_mass {float} -- mass charged in every tank timestep, minutely [kg]
        """

        cp1 = self.cp_water / 1000.0     # [kJ/kg-C]
        charge_mass = (thermal_output) / (cp1 * (source_temp-temp_node_charging))    # [kg] = kJ / (kJ/kg-K * K)

        return charge_mass

    def _discharging_mass(self, vol):
        """ Calculate mass discharged in every tank timestep"""
        
        # assume 1 L = 1 kg for water 

        discharge_mass = vol
        return abs(discharge_mass)

    def _mass_flow_calc(self, state, source_temp, thermal_output, temp_node_charging, vol):

        if state == 'charging':
            mass_ts = self._charging_mass(thermal_output, source_temp,temp_node_charging)

        elif state == 'discharging':
            mass_ts = 0.66*self._discharging_mass(vol) #used for RBC
            # mass_ts = self._discharging_mass(vol)

        elif state == 'standby':
            mass_ts = 0
        
        return mass_ts

    def _coefficient_A(self, state, node, nodes_temp, mass_flow,
                      source_temp, flow_temp, MTS):
        """ Calculate coefficient A {T(i) terms} for the node energy balance.

        Arguments:
            state {str} -- state of the tank (charging, discharging, or standby)
            node {int} -- current node
            nodes_temp {np.array} -- list of node temperatures temperatures [degC]
            mass_flow {} -- [kg]
            source_temp {float} -- temperature of HP water entering tank [degC]
            flow_temp {float} -- required DHW flow temperature [degC]
            
            MTS {} -- ?

        Returns:
            A {float} -- energy balance coefficient for T(i) terms [unitless]
        """

        node_mass = self.node_masses[node]
        cp = self.cp_water
        Fd = self._discharging_function(state, nodes_temp, flow_temp)[node]
        mf = self._mixing_function(state, node, nodes_temp,
                                  source_temp, flow_temp)

        Fco = self._charging_node_out(state,nodes_temp,source_temp)[node]

        # divide conductive & connection losses by number_nodes to calculate losses per node

        # whole term with inter-node transfer and ambient losses
        A = ((- Fd * mass_flow * cp)              # flow to SH loop from top node (discharging)
             - (mf['Fdnt'] * mass_flow * cp)      # mixing Q flow from node i to above (discharging)
             - (mf['Fcnb'] * mass_flow * cp)      # mixing Q flow from node i to below (charging)
             - (Fco * mass_flow * cp)              # flow back to HP from the charging node (charging)
             - (((MTS*60.) / (self.number_nodes)) * self.UA[node])    # losses
            ) / (node_mass * cp)

        return A
    
    def _coefficient_B(self, state, node, nodes_temp, mass_flow, 
                       source_temp, flow_temp):
        """ Calculate coefficient B {T(i-1) terms} for the node energy balance.

        Arguments:
            state {str} -- state of the tank (charging, discharging, or standby)
            node {int} -- current node
            nodes_temp {np.array} -- list of node temperatures temperatures [degC]
            mass_flow {} -- [kg/h]
            source_temp {float} -- temperature of HP water entering tank [degC]
            flow_temp {float} -- required DHW flow temperature [degC]
            

        Returns:
            B {float} -- energy balance coefficient for T(i-1) terms [unitless]
        """

        node_mass = self.node_masses[node]
        mf = self._mixing_function(state, node, nodes_temp,
                                  source_temp, flow_temp)

        # Whole-Term Method with inter-node transfer and ambient losses
        # B = mf['Fcnt'] * mass_flow / node_mass  # mixing Q flow into node i from above (charging)

        B = mf['Fcnt'] * mass_flow / node_mass  # mixing Q flow into node i from above (charging)

        #print("Calling Coefficient B")
        return B

    def _coefficient_C(self, state, node, nodes_temp, mass_flow, 
                       source_temp, flow_temp):
        """ Calculate coefficient C {T(i+1) terms} for the node energy balance.

        Arguments: 
            state {str} -- state of the tank (charging, discharging, or standby)
            node {int} -- current node
            nodes_temp {np.array} -- list of node temperatures temperatures [degC]
            mass_flow {} -- [kg/h]
            source_temp {float} -- temperature of HP water entering tank [degC]
            flow_temp {float} -- required DHW flow temperature [degC]
            

        Returns:
            C {float} -- energy balance coefficient for T(i+1) terms [unitless]
        """
        
        node_mass = self.node_masses[node]
        mf = self._mixing_function(state, node, nodes_temp,
                                  source_temp, flow_temp)

        # Whole-Term Method with inter-node transfer and ambient losses
        C = mf['Fdnb'] * mass_flow / node_mass  # mixing Q flow into node i from below (discharging)
        
        #print("Calling Coefficient C")
        return C

    def _coefficient_D(self, state, node, nodes_temp, mass_flow, 
                       source_temp, flow_temp, return_temp, timestep, MTS):
        """ Calculate coefficient D {unrelated to node temp} 
            for the node energy balance.

        Arguments:
            state {str} -- state of the tank (charging, discharging, or standby)
            node {int} -- current node
            nodes_temp {np.array} -- list of node temperatures temperatures [degC]
            mass_flow {} -- [kg/h]
            source_temp {float} -- temperature of HP water entering tank [degC]
            flow_temp {float} -- required DHW flow temperature [degC]
            
            MTS {} -- ?

        Returns:
            D {float} -- energy balance coefficient for terms unrelated to node temperature [unitless]
        """

        node_mass = self.node_masses[node]

        cp = self.cp_water
        Fc = self._charging_function(state, nodes_temp, source_temp)[node]
        Fdi = self._discharging_node_in(state, nodes_temp, flow_temp)[node]
        Ta = self.ambient_temp

        # divide conductive & connection losses by number_nodes to calculate losses per node

        # Whole-Term Method with inter-node transfer and ambient losses
        D = (Fc * mass_flow * cp * source_temp      # flow into charging node from HP (charging)
             + Fdi * mass_flow * cp * return_temp   # return flow from SH loop into bottom node (discharging)
             + (((MTS*60.) / (self.number_nodes)) * self.UA[node] * Ta) # losses
              ) / (node_mass * cp)

        return D

    def _set_of_coefficients(self, state, nodes_temp, source_temp, flow_temp, return_temp,
                            thermal_output, temp_node_charging, timestep, MTS, vol):
        # [kg or L per internal timestep]
        
        mass_flow = self._mass_flow_calc(state, source_temp, thermal_output, temp_node_charging, vol)
        
        c = []
        for node in range(self.number_nodes):
            coefficients = {'A': self._coefficient_A(
                state, node, nodes_temp, mass_flow, source_temp, flow_temp, MTS),
                            'B': self._coefficient_B(
                state, node, nodes_temp, mass_flow, source_temp, flow_temp),
                            'C': self._coefficient_C(
                state, node, nodes_temp, mass_flow, source_temp, flow_temp),
                            'D': self._coefficient_D(
                state, node, nodes_temp, mass_flow, source_temp, flow_temp,
                return_temp, timestep, MTS)}
            c.append(coefficients)
        
        return c

    def new_nodes_temp(self, state, nodes_temp, source_temp,
                       source_delta_t, flow_temp, return_temp,
                       thermal_output, demand, timestep, MTS, vol):
        
        if self.capacity == 0:
            return nodes_temp

        check = 0.0
        for node in range(len(nodes_temp)):
            check += nodes_temp[node]
                
        if check == source_temp * len(nodes_temp) and state == 'charging':
            return nodes_temp * len(nodes_temp)

        def model_temp(z, t, c):
            dzdt = []
            for node in range(self.number_nodes):

                if node == 0:
                    Ti = nodes_temp[node]
                    Ti_b = nodes_temp[node + 1]

                    dTdt = (c[node]['A'] * Ti +
                            c[node]['C'] * Ti_b +
                            c[node]['D'])

                    dzdt.append(dTdt)

                elif node == (self.number_nodes - 1):
                    Ti = nodes_temp[node]
                    Ti_a = nodes_temp[node - 1]

                    dTdt = (c[node]['A'] * Ti +
                            c[node]['B'] * Ti_a +
                            c[node]['D'])

                    dzdt.append(dTdt)

                else:
                    Ti = nodes_temp[node]
                    Ti_b = nodes_temp[node + 1]
                    Ti_a = nodes_temp[node - 1]

                    dTdt = (c[node]['A'] * Ti +
                            c[node]['B'] * Ti_a +
                            c[node]['C'] * Ti_b +
                            c[node]['D'])

                    dzdt.append(dTdt)

            return dzdt

        # number of time points - limiting factor is volume of one node
        # therefore minimum time points needed is number of nodes
        t = self.number_nodes
        # debug
        # t = 1

        # node indexes
        top = 0
        bottom = self.number_nodes - 1

        # convert from kWh to kJ per internal tank timestep
        thermal_output = thermal_output * 3600. / float(t)
        demand = demand * 3600. / float(t) #not used - just use volume later
        vol = vol / float(t)

        # initial condition of coefficients and node_temp_list
        coefficients = []
        node_temp_list = []

        # solve ODE
        for i in range(0, t):
            nodes_temp[bottom] = min(source_temp - 0.01, nodes_temp[bottom])
            
            # span for next time step
            tspan = [i, i+1]
            
            # solve for next step
            
            # new coefficients for new internal timestep
            coefficients.append((self._set_of_coefficients(
                state, nodes_temp, source_temp,
                flow_temp, return_temp, thermal_output, 
                nodes_temp[3], timestep, MTS, vol)))

            # solve ODE
            z = odeint(
                model_temp, nodes_temp, tspan,
                args=(coefficients[i],))

            # reassign nodes_temp from solved ODE
            nodes_temp = z[1] # skip [0] as it is old nodes_temp
            nodes_temp = sorted(nodes_temp, reverse=True)
            node_temp_list.append(nodes_temp) #creating a list of lists of node temps

        # node temperature correction during charging steps
        returned_node_temp_list = node_temp_list[t-1]
        if state == "charging":
            for node in self.node_list:
                if returned_node_temp_list[node] > source_temp and node != bottom:
                    #calculated energy removed - Q = m*cp*deltaT
                    mass = self.node_masses[node] #m dot
                    cp = self.cp_water*1/1000 * 1/3600  #cp from J/kg-K to kWh/kg-K
                    delta_T = returned_node_temp_list[node] - source_temp #deltaT
                    q_removed = mass * cp * delta_T     #kWh reallocated from top node

                    returned_node_temp_list[node] = source_temp

                    #calculate next node's new temp - Tnew = Q/m*cp + Told
                    if node+1 != bottom:
                        returned_node_temp_list[node+1] = returned_node_temp_list[node+1] + q_removed/(cp*self.node_masses[node+1])

        # return node_temp_list[self.number_nodes-1] #return the last set of node temps calculated during loop
        return returned_node_temp_list
    
    def _coefficient_A_max(self, state, node, nodes_temp, source_temp,
                          flow_temp, MTS):
        """ Calculate maximum coefficient A {T(i) terms} for 
            the node energy balance assuming entire node mass flows.

        Arguments:
            state {str} -- state of the tank (charging, discharging, or standby)
            node {int} -- current node
            nodes_temp {np.array} -- list of node temperatures temperatures [degC]
            source_temp {float} -- temperature of HP water entering tank [degC]
            flow_temp {float} -- required SH flow temperature [degC]

        Returns:
            A {float} -- maximum energy balance coefficient for T(i) terms [unitless]
        """
        node_mass = self.node_masses[node]

        cp = self.cp_water
        Fd = self._discharging_function(state, nodes_temp, flow_temp)[node]
        mf = self._mixing_function(state, node, nodes_temp,
                                  source_temp, flow_temp)
        Fco = self._charging_node_out(state, nodes_temp, source_temp)[node]

        mass_flow = self.node_masses[node]
        
        # divide conductive & connection losses by number_nodes to calculate losses per node

        # whole term with inter-node transfer and ambient losses
        A = ((- Fd * mass_flow * cp)              # flow to SH loop from top node (discharging)
             - (mf['Fdnt'] * mass_flow * cp)      # mixing Q flow from node i to above (discharging)
             - (mf['Fcnb'] * mass_flow * cp)      # mixing Q flow from node i to below (charging)
             - (Fco * mass_flow * cp)             # flow to HP from bottom node (charging)
             - (((MTS*60.) / (self.number_nodes-1.)) * self.UA[node])    # losses
            ) / (node_mass * cp)

        return A

    def _coefficient_B_max(self, state, node, nodes_temp, source_temp,
                          flow_temp):
        """ Calculate maximum coefficient B {T(i-1) terms} for 
            the node energy balance assuming entire node mass flows.

        Arguments:
            state {str} -- state of the tank (charging, discharging, or standby)
            node {int} -- current node
            nodes_temp {np.array} -- list of node temperatures temperatures [degC]
            source_temp {float} -- temperature of HP water entering tank [degC]
            flow_temp {float} -- required SH flow temperature [degC]

        Returns:
            B {float} -- maximum energy balance coefficient for T(i-1) terms [unitless]
        """
        #print("Coefficient B Max")
        node_mass = self.node_masses[node]
        mf = self._mixing_function(state, node, nodes_temp,
                                  source_temp, flow_temp)
        mass_flow = self.node_masses[node]

        B = mf['Fcnt'] * mass_flow / node_mass  # mixing Q flow into node i from above (charging)

        
        return B

    def _coefficient_C_max(self, state, node, nodes_temp, source_temp,
                          flow_temp, timestep):
        """ Calculate maximum coefficient C {T(i+1) terms} for 
            the node energy balance assuming entire node mass flows.

        Arguments: 
            state {str} -- state of the tank (charging, discharging, or standby)
            node {int} -- current node
            nodes_temp {np.array} -- list of node temperatures temperatures [degC]
            source_temp {float} -- temperature of HP water entering tank [degC]
            flow_temp {float} -- required SH flow temperature [degC]
            

        Returns:
            C {float} -- maximum energy balance coefficient for T(i+1) terms [unitless]
        """
        #print("Coefficient C Max")
        node_mass = self.node_masses[node]
        mf = self._mixing_function(state, node, nodes_temp,
                                  source_temp, flow_temp)
        mass_flow = node_mass

        C = mf['Fdnb'] * mass_flow / node_mass

        
        return C

    def _coefficient_D_max(self, state, node, nodes_temp, source_temp,
                          flow_temp, return_temp, timestep, MTS):
        """ Calculate maximum coefficient D {unrelated to node temp} 
            for the node energy balance assuming entire node mass flows.

        Arguments:
            state {str} -- state of the tank (charging, discharging, or standby)
            node {int} -- current node
            nodes_temp {np.array} -- list of node temperatures temperatures [degC]
            source_temp {float} -- temperature of HP water entering tank [degC]
            flow_temp {float} -- required SH flow temperature [degC]
            

        Returns:
            D {float} -- maximum energy balance coefficient for terms unrelated to node temperature [unitless]
        """

        node_mass = self.node_masses[node]

        cp = self.cp_water
        Fc = self._charging_function(state, nodes_temp, source_temp)[node]
        Fdi = self._discharging_node_in(state, nodes_temp, flow_temp)[node]
        Ta = self.ambient_temp

        mass_flow = node_mass
        
        D = (Fc * mass_flow * cp * source_temp      # flow into top node from HP (charging)
             + Fdi * mass_flow * cp * return_temp   # return flow from SH loop into bottom node (discharging)
             + (((MTS*60.) / (self.number_nodes - 1)) * self.UA[node] * Ta) # losses
              ) / (node_mass * cp)

        return D

    def _set_of_max_coefficients(self, state, nodes_temp, source_temp,
                                flow_temp, return_temp, timestep, MTS):
        """ Creates a list of dictionaries that contain the maximum coefficients for each node.

        Arguments:
            state {str} -- state of the tank (charging, discharging, or standby)
            nodes_temp {np.array} -- list of node temperatures temperatures [degC]
            source_temp {float} -- temperature of HP water entering tank [degC]
            flow_temp {float} -- required DHW flow temperature [degC]
            return_temp {float} -- water return temperature after providing heat [degC]
            timestep {int} -- length of time in the timestep {1}
            

        Returns:
            c {list} -- list of n (num_nodes) dictionaries containing maximum coefficient key / value     
        """
        c = []
        for node in range(self.number_nodes):
            coefficients = {'A': self._coefficient_A_max(
                state, node, nodes_temp, source_temp, flow_temp, MTS),
                            'B': self._coefficient_B_max(
                state, node, nodes_temp, source_temp, flow_temp),
                            'C': self._coefficient_C_max(
                state, node, nodes_temp, source_temp, flow_temp,
                timestep),
                            'D': self._coefficient_D_max(
                state, node, nodes_temp, source_temp, flow_temp,
                return_temp, timestep, MTS)}
            c.append(coefficients)
            
        return c

    def max_energy_in_out(self, state, nodes_temp, source_temp,
                          flow_temp, return_temp, timestep, MTS):

        nodes_temp_sum = 0.0
        for node in range(len(nodes_temp)):
            nodes_temp_sum += nodes_temp[node]
        if nodes_temp_sum >= source_temp * len(nodes_temp) and state == 'charging':
            return 0.0

        if nodes_temp_sum <= return_temp * len(nodes_temp) and state == 'discharging':
            return 0.0

        def model_temp(z, t, c):
            dzdt = []
            for node in range(self.number_nodes):

                if node == 0:
                    Ti = nodes_temp[node]
                    Ti_b = nodes_temp[node + 1]

                    dTdt = (c[node]['A'] * Ti +
                            c[node]['C'] * Ti_b +
                            c[node]['D'])

                    dzdt.append(dTdt)

                elif node == (self.number_nodes - 1):
                    Ti = nodes_temp[node]
                    Ti_a = nodes_temp[node - 1]

                    dTdt = (c[node]['A'] * Ti +
                            c[node]['B'] * Ti_a +
                            c[node]['D'])

                    dzdt.append(dTdt)

                else:
                    Ti = nodes_temp[node]
                    Ti_b = nodes_temp[node + 1]
                    Ti_a = nodes_temp[node - 1]

                    dTdt = (c[node]['A'] * Ti +
                            c[node]['B'] * Ti_a +
                            c[node]['C'] * Ti_b +
                            c[node]['D'])

                    dzdt.append(dTdt)

            return dzdt

        # number of time points
        t = self.number_nodes
        #debug 
        # t = 1

        # node indexes
        top = 0
        bottom = self.number_nodes - 1

        # initial condition of coefficients
        coefficients = []

        energy_list = []
        mass_flow = self.node_masses[0]

        cp = self.cp_water

        # solve ODE
        for i in range(0, t):
            
            if state == 'charging' and source_temp > nodes_temp[bottom]:
                energy = 1.0 * mass_flow * cp * (
                    source_temp - nodes_temp[bottom]) 

            elif state == 'discharging' and nodes_temp[top] > (flow_temp):
                energy = 1.0 * mass_flow * cp * (
                    nodes_temp[top] - return_temp)
            else:
                energy = 0

                
            energy_list.append(energy)

            # span for next time step
            tspan = [i, i+1]

            # solve for next step

            # new coefficients
            coefficients.append((self._set_of_max_coefficients(
                state, nodes_temp, source_temp,
                flow_temp, return_temp, timestep, MTS)))

            z = odeint(
                model_temp, nodes_temp, tspan,
                args=(coefficients[i],))
            
            nodes_temp = z[1]   # skip [0] as it is old nodes_temp
            nodes_temp = sorted(nodes_temp, reverse=True)

        # convert J to kWh by divide by 3600000
        energy_total = sum(energy_list) / 3600000

        return energy_total