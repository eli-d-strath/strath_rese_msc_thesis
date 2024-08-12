"""
Modelling the Space Heating (SH) water storage tank at Woodside

Based on the water tank models "hot_water_tank_5_ws_sh.py" provided by Graeme Flett
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
        """HotWaterTank class object

        Specifications are for the Akvaterm AKVA 1500E model, available at
        https://www.northmangroup.co.uk/Akva_EK_1500.html#:~:text=The%20customisable%20AKVA%20EK%20accumulator,cell%20polyurethane%20is%20100mm%20thick.

        """
        self.capacity = 1.500                       # [m^3] tank capacity 
        self.number_nodes = 5                       # number of modeled tank nodes [unitless]
        
        factor = 3.3        # set according to real tank specs
        self.width_internal = 2. * (self.capacity / (factor * math.pi)) ** (1. / 3.) #approx 10.5 decimeters
        self.height_internal = 0.5 * factor * self.width_internal        #approx 17.32 decimeters

        self.insulation = 0.1               # [m] insulation added to internal width/height
        
        self.internal_radius = (0.5 * self.width_internal)       # [m]
        self.external_radius = self.internal_radius + self.insulation # [m]

        # materials, environment, and constants
        self.insulation_k_value = 0.025             # [W/m-C] polyurethane 
        self.ambient_temp = 15.                     # [degC] tank is inside 
        self.insulation_factor = 1.                 # [unitless] see dissertation
        self.overall_factor = 1.                    # [unitless] see dissertation
        self.cp_water = 4180.                       # [J/kg-C] water specific heat

        # self.node_masses = np.array([300.,300.,300.,300.,300.]) #[kg or L]  #! causes an error
        self.node_masses = np.array([300.,250.,250.,300.,400.]) #[kg or L] up to V7
        # self.node_masses = np.array([200.,400.,400.,300.,200.]) #[kg or L]
        self.node_list = list(range(self.number_nodes))         # [0, 1, 2, 3, 4]
        

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

                # for out of bounds nodes; shouldnt occur
                elif node < 0 or node == self.number_nodes + 1:
                    function[node] = 0

                else:
                    function[node] = 0

        # assign all nodes a discharging function value of 0
        elif state.lower() == 'charging' or state.lower() == 'standby':
            for node in self.node_list:
                function[node] = 0

        return function

    def _discharging_bottom_node(self, state, nodes_temp,
                                return_temp, flow_temp):
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

        df = self._discharging_function(state, nodes_temp, flow_temp)
        df_list = []
        
        # create list of discharging status from the _discharging_function
        for i in self.node_list:
            df_list.append(df[i])

        #if any node is discharging, assign bottom node to recharge
        if 1 in df_list:

            for node in self.node_list:

                # this asks if we are looking at the bottom node
                # and the top node temp is greater than the flow temp
                if node == bottom_node and nodes_temp[0] >= flow_temp:
                    function[node] = 1

                # this asks if we are looking at the bottom node
                # and the top node temp is less than the flow temp
                elif node == bottom_node and nodes_temp[0] < flow_temp:
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

        # assign nodes a function of charging (1) or inactive (0)
        if state == 'charging':

            for node in self.node_list:

                # if we are looking at the top node,
                # and if the charging water is above this node's temp
                if node == 0 and source_temp >= nodes_temp[0]:
                    function[node] = 1

                # if we are looking at the top node,
                # and source temp is lower than top node temp
                elif node == 0 and source_temp <= nodes_temp[0]:
                    function[node] = 0

                # top node then goes in other node
                # if source temp is above new node's temp
                elif nodes_temp[node] <= source_temp < nodes_temp[node - 1]:
                    function[node] = 0 

                # for out of bounds nodes; shouldnt occur
                elif node < 0 or node == self.number_nodes + 1:
                    function[node] = 0

                else:
                    function[node] = 0

        # assign all nodes a charging function value of 0
        elif state == 'discharging' or 'standby':
            for node in self.node_list:
                function[node] = 0
        
        return function

    def _charging_bottom_node(self, state):
        """ Schedules outlet flow to HP from bottom node when tank is charging.

        Arguments:
            state {str} -- state of the water tank (charging or discharging)

        Returns:
            function {dict} -- dict with node number as key and charging status as value
        """

        function = {}
        for node in self.node_list:

            if state == 'charging' and node == self.number_nodes - 1:
                function[node] = 1
            else:
                function[node] = 0

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

        Returns:
            mf {dict} -- mixing function with Fcnt, Fdnt, Fcnb, Fdnt keys and 0/1 values
        """
        
        total_nodes = self.number_nodes
        bottom_node = total_nodes - 1

        cf = self._charging_function(state, nodes_temp, source_temp)
        cf_list = []
        for i in range(total_nodes):
            cf_list.append(cf[i])

        df = self._discharging_function(state, nodes_temp, flow_temp)
        df_list = []
        for i in range(total_nodes):
            df_list.append(df[i])

        mf = {}

        if 1 in cf_list:
            for n in range(self.number_nodes):
                if cf[n] == 1:
                    node_charging = n
        else:
            node_charging = bottom_node + 1
            
        if 1 in df_list:
            for n in range(self.number_nodes):
                if df[n] == 1:
                    node_discharging = n
        else:
            node_discharging = bottom_node + 1

        if state == 'charging': 
            if node <= node_charging:
                mf['Fcnt'] = 0
                mf['Fdnt'] = 0
            else:
                mf['Fcnt'] = 1
                mf['Fdnt'] = 0

            if node >= bottom_node or node < node_charging:
                mf['Fcnb'] = 0
                mf['Fdnb'] = 0
            else:
                mf['Fcnb'] = 1
                mf['Fdnb'] = 0
                
        # discharging
        elif state == 'discharging':
            if node == 0 or node <= node_discharging:
                mf['Fcnt'] = 0
                mf['Fdnt'] = 0
            else:
                mf['Fcnt'] = 0
                mf['Fdnt'] = 1

            if node == bottom_node or node < node_discharging:
                mf['Fcnb'] = 0
                mf['Fdnb'] = 0
            else:
                mf['Fcnb'] = 0
                mf['Fdnb'] = 1
        # standby or solar thermal input
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

    def _charging_mass(self, thermal_output, source_temp,
                                      source_delta_t, return_temp, flow_temp,
                                      demand, temp_tank_bottom):
        """ Calculate the mass flow into the thermal store during charging 

        Arguments: 
            thermal_output {float} -- heat injected from HP to tank [kJ] 
            source_temp {float} -- temperature of HP water entering tank [degC]
            source_delta_t {float} -- [degC]
            return_temp {float} -- water return temperature after providing heat [degC]
            flow_temp {float} -- required SH flow temperature [degC]
            demand {float} -- demand for space heating [kJ]
            temp_tank_bottom {float} -- temperature at bottom of the tank [degC]

        Returns:
            ts_mass {float} -- mass charged in every tank timestep, minutely [kg]
        """
        
        #! simpler way to calculate
        cp1 = self.cp_water / 1000.0     # [kJ/kg-C]
        ts_mass = (thermal_output - demand) / (cp1 * (source_temp-temp_tank_bottom)) 

        return ts_mass

    def _discharging_mass(self, thermal_output, source_temp,
                                         source_delta_t, return_temp,
                                         flow_temp,
                                         demand, temp_tank_top,vol):
        """ Calculate mass discharged in every tank timestep using required energy

        Arguments: 
            thermal_output {float} -- heat injected from HP to tank [kJ] 
            source_temp {float} -- temperature of HP water entering tank [degC]
            source_delta_t {float} -- [degC]
            return_temp {float} -- water return temperature after providing heat [degC]
            flow_temp {float} -- required SH flow temperature [degC]
            demand {float} -- demand for space heating [kJ] 
            temp_tank_top {float} -- temperature at top of the tank [degC]
            vol {float} --volumetric water demand for SH [L]

        Returns:
            ts_mass {float} -- mass discharged in every tank timestep, minutely [kg=L]
        """
        # mass discharged in every tank timestep, minutely
              
        #! simpler way to calculate...
        cp1 = self.cp_water / 1000.0    # [kJ/kg-K]
        
        ts_mass = (demand - thermal_output) / (cp1 * (temp_tank_top - return_temp))
        
        # ts_mass = vol   # [L = kg] for water

        return abs(ts_mass)

    def _mass_flow_calc(self, state, flow_temp, return_temp,
                       source_temp, source_delta_t, thermal_output, demand,
                       temp_tank_bottom, temp_tank_top,vol):
        if state == 'charging':
            mass_ts = self._charging_mass(
                thermal_output, source_temp,
                source_delta_t, return_temp, flow_temp,
                demand, temp_tank_bottom)

        elif state == 'discharging':
            mass_ts = self._discharging_mass(
                thermal_output, source_temp, source_delta_t,
                return_temp, flow_temp, demand, temp_tank_top,vol)
            
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
            mass_flow {} -- mass flow [kg]
            source_temp {float} -- temperature of HP water entering tank [degC]
            flow_temp {float} -- required SH flow temperature [degC]

        Returns:
            A {float} -- energy balance coefficient for T(i) terms [unitless]
        """
        
        node_mass = self.node_masses[node]
        
        # specific heat at temperature of node i
        cp = self.cp_water

        # thermal conductivity of insulation material
        k = self.insulation_k_value*3600    # [J/h-m-C] from [W/m-C]
        
        # dimensions
        r1 = self.internal_radius
        r2 = self.external_radius
        h = self.height_internal 
        
        # correction and heat transfer factors
        Fi = self.insulation_factor
        Fe = self.overall_factor
        Fd = self._discharging_function(state, nodes_temp, flow_temp)[node]
        mf = self._mixing_function(state, node, nodes_temp,
                                  source_temp, flow_temp)
        Fco = self._charging_bottom_node(state)[node]
        
        # divide conductive & connection losses by number_nodes to calculate losses per node

        # Whole-Term Method with inter-node transfer and ambient losses
        A = ((- Fd * mass_flow * cp)            # flow to SH loop from top node (discharging)
             - (mf['Fdnt'] * mass_flow * cp)    # mixing Q flow from node i to above (discharging)
             - (mf['Fcnb'] * mass_flow * cp)    # mixing Q flow from node i to below (charging)
             - (Fco * mass_flow * cp)           # flow to HP from bottom node (charging)
             - (Fe * Fi * k * ((1) / (r2 - r1)) * math.pi * (1./(self.number_nodes)) *
                (((2. / self.number_nodes) * (r1 ** 2)) + (h / self.number_nodes) * (r2 + r1)))  #conductive side and top losses term
             ) / (node_mass * cp)
            
        return A

    def _coefficient_B(self, state, node, nodes_temp, mass_flow, source_temp,
                      flow_temp):
        """ Calculate coefficient B {T(i-1) terms} for the node energy balance.

        Arguments:
            state {str} -- state of the tank (charging, discharging, or standby)
            node {int} -- current node
            nodes_temp {np.array} -- list of node temperatures temperatures [degC]
            mass_flow {} -- [kg/h]
            source_temp {float} -- temperature of HP water entering tank [degC]
            flow_temp {float} -- required SH flow temperature [degC]

        Returns:
            B {float} -- energy balance coefficient for T(i-1) terms [unitless]
        """

        node_mass = self.node_masses[node]
        mf = self._mixing_function(state, node, nodes_temp,
                                  source_temp, flow_temp)
        
        # Whole-Term Method with inter-node transfer and ambient losses
        B = mf['Fcnt'] * mass_flow / node_mass  # mixing Q flow into node i from above (charging)


        return B

    def _coefficient_C(self, state, node, nodes_temp, mass_flow, source_temp,
                      flow_temp):
        """ Calculate coefficient C {T(i+1) terms} for the node energy balance.

        Arguments: 
            state {str} -- state of the tank (charging, discharging, or standby)
            node {int} -- current node
            nodes_temp {np.array} -- list of node temperatures temperatures [degC]
            mass_flow {} -- [kg/h]
            source_temp {float} -- temperature of HP water entering tank [degC]
            flow_temp {float} -- required SH flow temperature [degC]

        Returns:
            C {float} -- energy balance coefficient for T(i+1) terms [unitless]
        """
        
        node_mass = self.node_masses[node]
        mf = self._mixing_function(state, node, nodes_temp,
                                  source_temp, flow_temp)
        
        # Whole-Term Method with inter-node transfer and ambient losses
        C = mf['Fdnb'] * mass_flow / node_mass  # mixing Q flow into node i from below (discharging)

        return C

    def _coefficient_D(self, state, node, nodes_temp, mass_flow, source_temp,
                      flow_temp, return_temp, timestep, MTS):
        """ Calculate coefficient D {unrelated to node temp} 
            for the node energy balance.

        Arguments:
            state {str} -- state of the tank (charging, discharging, or standby)
            node {int} -- current node
            nodes_temp {np.array} -- list of node temperatures temperatures [degC]
            mass_flow {} -- [kg/h]
            source_temp {float} -- temperature of HP water entering tank [degC]
            flow_temp {float} -- required SH flow temperature [degC]
            MTS {} -- ?

        Returns:
            D {float} -- energy balance coefficient for terms unrelated to node temperature [unitless]
        """

        node_mass = self.node_masses[node]
        
        # specific heat at temperature of node i
        cp = self.cp_water                  # [J/kg-C]

        # thermal conductivity of insulation material
        k = self.insulation_k_value*3600    # [J/h-m-C] from [W/m-C]

        # dimensions
        r1 = self.internal_radius           # [m]
        r2 = self.external_radius           # [m]
        h = self.height_internal 

        # correction factors
        Fi = self.insulation_factor
        Fe = self.overall_factor

        Fc = self._charging_function(state, nodes_temp, source_temp)[node]
        Fdi = self._discharging_bottom_node(
            state, nodes_temp, return_temp, flow_temp)[node]
        Ta = self.ambient_temp

        cl = self._connection_losses()*3600
        
        # divide conductive & connection losses by number_nodes to calculate losses per node

        # Whole-Term Method with inter-node transfer and ambient losses
        D = (Fc * mass_flow * cp * source_temp      # flow into top node from HP (charging)
             + Fdi * mass_flow * cp * return_temp   # return flow from SH loop into bottom node (discharging)
             + (Fe * Fi * k * ((Ta) / (r2 - r1)) * math.pi * (1./(self.number_nodes)) *  
             (((1. / self.number_nodes) * (r1 ** 2)) + (h / self.number_nodes) * (r2 + r1))    # conductive top and sides losses
             - (Fe * (cl / self.number_nodes) * (1./(self.number_nodes))
                )    # connection losses for this node
             )) / (node_mass * cp)
            
        return D

    def _set_of_coefficients(self, state, nodes_temp, source_temp,
                            source_delta_t, flow_temp, return_temp,
                            thermal_output, demand, temp_tank_bottom,
                            temp_tank_top, timestep, MTS,vol):
        """ Creates a list of dictionaries that contain the coefficients for each node.

        Arguments:
            state {str} -- state of the tank (charging, discharging, or standby)
            nodes_temp {np.array} -- list of node temperatures temperatures [degC]
            source_temp {float} -- temperature of HP water entering tank [degC]
            source_delta_t {float} -- [degC]
            flow_temp {float} -- required SH flow temperature [degC]
            return_temp {float} -- water return temperature after providing heat [degC]
            thermal_output {float} -- heat injected from HP to tank [kJ]
            demand {float} -- energy demand for space heating [kJ]
            temp_tank_bottom {float} -- bottom node temperature [degC]
            temp_tank_top {float} -- top node temperature [degC]
            timestep {int} -- length of time in the timestep {1}
            MTS {} -- ?
            vol {float} -- volumetric water demand for SH [L]

        Returns:
            c {list} -- list of n (num_nodes) dictionaries containing coefficient key / value  
        """
        mass_flow = self._mass_flow_calc(
            state, flow_temp, return_temp, source_temp, source_delta_t,
            thermal_output, demand, temp_tank_bottom, temp_tank_top,vol)

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
                       thermal_output, demand, timestep, MTS,vol):
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

        # number of time points
        # limiting factor is volume of one node
        # therefore minimum time points needed is number of nodes
        t = self.number_nodes

        # node indexes
        top = 0
        bottom = self.number_nodes - 1

        # divide thermal output and demand across timesteps
        # convert from kWh to kJ
        thermal_output = thermal_output * 3600. / float(t)
        demand = demand * 3600. / float(t)
        vol = vol / float(t)
        
        nodes_temp[bottom] = min(source_temp - 0.01, nodes_temp[bottom])

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
                state, nodes_temp, source_temp, source_delta_t,
                flow_temp, return_temp, thermal_output, demand,
                nodes_temp[bottom], nodes_temp[top], timestep, MTS,vol)))
            
            # solve ODE
            z = odeint(
                model_temp, nodes_temp, tspan,
                args=(coefficients[i],))

            # reassign nodes_temp from solved ODE
            nodes_temp = z[1]   # skip [0] as it is old nodes_temp
            nodes_temp = sorted(nodes_temp, reverse=True)
            node_temp_list.append(nodes_temp) #creating a list of lists of node temps

        # node temperature correction if going below ambient
        returned_node_temp_list = node_temp_list[t-1]
        for node in self.node_list:
            if returned_node_temp_list[node] < self.ambient_temp:
                returned_node_temp_list[node] = self.ambient_temp

        return returned_node_temp_list #return the last set of node temps calculated during loop

    def _coefficient_A_max(self, state, node, nodes_temp, source_temp,
                          flow_temp):
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

        # specific heat at temperature of node i
        cp = self.cp_water              # [J/kg-C]

        # thermal conductivity of insulation material
        k = self.insulation_k_value*3600    # [J/h-m-C] from [W/m-C]

        # dimensions
        r1 = self.internal_radius           # [m]
        r2 = self.external_radius           # [m]
        h = self.height_internal            # [m]

        # correction factors
        Fi = self.insulation_factor
        Fe = self.overall_factor
        Fd = self._discharging_function(state, nodes_temp, flow_temp)[node]
        mf = self._mixing_function(state, node, nodes_temp,
                                  source_temp, flow_temp)
        Fco = self._charging_bottom_node(state)[node]
        
        mass_flow = self.node_masses[node]

        # divide conductive & connection losses by number_nodes to calculate losses per node

        # whole term with inter-node transfer and ambient losses
        A = ((- Fd * mass_flow * cp)                # flow to SH loop from top node (discharging)
             - (mf['Fdnt'] * mass_flow * cp)        # mixing Q flow from node i to above (discharging)
             - (mf['Fcnb'] * mass_flow * cp)        # mixing Q flow from node i to below (charging)
             - (Fco * mass_flow * cp)               # flow to HP from bottom node (charging)
             - (Fe * Fi * k * ((1) / (r2 - r1)) * math.pi * 
                (((2. / self.number_nodes) * (r1 ** 2)) + (h / self.number_nodes) * (r2 + r1)))  #conductive side and top losses term
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

        node_mass = self.node_masses[node]
        mf = self._mixing_function(state, node, nodes_temp,
                                  source_temp, flow_temp)
        mass_flow = node_mass

        C = mf['Fdnb'] * mass_flow / node_mass  # mixing Q flow into node i from below (discharging)

        return C

    def _coefficient_D_max(self, state, node, nodes_temp, source_temp,
                          flow_temp, return_temp, timestep):
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

        # specific heat at temperature of node i
        cp = self.cp_water              # [J/kg-C]

        # thermal conductivity of insulation material
        k = self.insulation_k_value*3600    # [J/h-m-C] from [W/m-C]

        # dimensions
        r1 = self.internal_radius           # [m]
        r2 = self.external_radius           # [m]
        h = self.height_internal            # [m]

        # correction factors
        Fi = self.insulation_factor
        Fe = self.overall_factor

        Fc = self._charging_function(state, nodes_temp, source_temp)[node]
        Fdi = self._discharging_bottom_node(
            state, nodes_temp, return_temp, flow_temp)[node]
        Ta = self.ambient_temp

        cl = self._connection_losses()

        mass_flow = node_mass

        D = (Fc * mass_flow * cp * source_temp      # flow into top node from HP (charging)
             + Fdi * mass_flow * cp * return_temp   # return flow from SH loop into bottom node (discharging)
             + (Fe * Fi * k * ((Ta) / (r2 - r1)) * math.pi * (1./(self.number_nodes)) *
             (((2. / self.number_nodes) * (r1 ** 2)) + (h / self.number_nodes) * (r2 + r1))    # conductive top and sides losses
             - ((1./(self.number_nodes)) * Fe * cl / self.number_nodes)     # connection losses for this node
             )) / (node_mass * cp)

        return D

    def _set_of_max_coefficients(self, state, nodes_temp, source_temp,
                                flow_temp, return_temp, timestep):
        """ Creates a list of dictionaries that contain the maximum coefficients for each node.

        Arguments:
            state {str} -- state of the tank (charging, discharging, or standby)
            nodes_temp {np.array} -- list of node temperatures temperatures [degC]
            source_temp {float} -- temperature of HP water entering tank [degC]
            flow_temp {float} -- required SH flow temperature [degC]
            return_temp {float} -- water return temperature after providing heat [degC]
            timestep {int} -- length of time in the timestep {1}

        Returns:
            c {list} -- list of n (num_nodes) dictionaries containing maximum coefficient key / value     
        """

        c = []
        for node in range(self.number_nodes):
            coefficients = {'A': self._coefficient_A_max(
                state, node, nodes_temp, source_temp, flow_temp),
                            'B': self._coefficient_B_max(
                state, node, nodes_temp, source_temp, flow_temp),
                            'C': self._coefficient_C_max(
                state, node, nodes_temp, source_temp, flow_temp,
                timestep),
                            'D': self._coefficient_D_max(
                state, node, nodes_temp, source_temp, flow_temp,
                return_temp, timestep)}
            c.append(coefficients)
            
        return c

    def max_energy_in_out(self, state, nodes_temp, source_temp,
                          flow_temp, return_temp, timestep, MTS):

        nodes_temp_sum = 0.0 #not sure this metric is the best for evaluating ability to charge/discharge?
        for node in range(len(nodes_temp)):
            nodes_temp_sum += nodes_temp[node]
        if nodes_temp_sum >= source_temp * len(nodes_temp) and state == 'charging':
            return 0.0

        if nodes_temp_sum <= return_temp * len(nodes_temp) and state == 'discharging':
            return 0.0
        
        # if nodes_temp[0] > 

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
        t = self.number_nodes - 1

        # initial condition of coefficients
        coefficients = []
        coefficients.append((self._set_of_max_coefficients(
            state, nodes_temp, source_temp,
            flow_temp, return_temp, timestep)))

        energy_list = []
        mass_flow = self.node_masses[0]

        cp = self.cp_water

        # solve ODE
        for i in range(0, t):
#            if state == 'charging':
#                nodes_temp = np.array([nodes_temp[3], nodes_temp[0], nodes_temp[1], nodes_temp[2], nodes_temp[4]])
            
            if state == 'charging' and source_temp > nodes_temp[self.number_nodes - 1]:
                energy = 1.0 * mass_flow * cp * (
                    source_temp - nodes_temp[self.number_nodes - 1])

            elif state == 'discharging' and nodes_temp[0] > (flow_temp):

                energy = mass_flow * cp * (
                    nodes_temp[0] - return_temp)
                
                # if nodes_temp[1] > flow_temp:
                #     energy = mass_flow * cp * (
                #             nodes_temp[0] - return_temp)
                    
                # else:
                #     mass_flow = self.node_masses[node] * (1. - ((flow_temp - nodes_temp[1])/(nodes_temp[0] - nodes_temp[1])))
                #     energy = mass_flow * cp * (
                #             nodes_temp[0] - return_temp)
                

            else:
                energy = 0

                
            energy_list.append(energy)

            # span for next time step
            tspan = [i - 1, i]
            # solve for next step
            # new coefficients
            coefficients.append((self._set_of_max_coefficients(
                state, nodes_temp, source_temp,
                flow_temp, return_temp, timestep)))

            z = odeint(
                model_temp, nodes_temp, tspan,
                args=(coefficients[i],))
            
            nodes_temp = z[1]

        # convert J to kWh by divide by 3600000
        energy_total = sum(energy_list) / 3600000

        return energy_total