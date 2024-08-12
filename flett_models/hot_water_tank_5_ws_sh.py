"""modelling hot watertanks

"""
import os
import pandas as pd
import math
import numpy as np

from scipy.integrate import odeint

class HotWaterTank(object):

    def __init__(self):

        self.capacity = 1500.
        self.number_nodes = 5
        
        factor = 2.5
        self.width = 2. * (self.capacity / (factor * math.pi)) ** (1. / 3.)
        self.height = 0.5 * factor * self.width
        # assuming a ratio of width to insulation thickness
        ins_divider = 8.
        self.insulation = self.width / ins_divider
        
        self.ins_fac = 1.
        self.ov_fac = 1.
        
    def init_temps(self, initial_temp):
        nodes_temp = []
        for node in range(self.number_nodes):
            nodes_temp.append(initial_temp)
        return nodes_temp

    def calc_node_mass(self):
        """calculates the mass of one node

        Returns:
            float -- mass of one node kg
        """

#        node_mass = float(self.capacity) / self.number_nodes
        node_mass = np.array([400.,400.,300.,200.,200.])
        return node_mass

    def insulation_k_value(self):
        """selects k for insulation

        Returns:
            float -- k-value of insulation W/mK
        """

        k = 0.025

        return k

    def specific_heat_water(self, temp):
        """cp of water

        Arguments:
            temp {float} -- temperature of water

        Returns:
            float -- cp of water at given temp - j/(kg deg C)
        """
        # input temp must be between 0 and 100 deg

        # if isinstance(temp, (int, float)) and (temp > 0. and temp < 100.):

        #     df = self.cp_spec
        #     T = round(float(temp), -1)
        #     # convert j/g deg to j/kg deg
        #     cp = df['Cp'][T / 10] * 1000

        # else:
        cp = 4180

        return cp

    def internal_radius(self):
        """calculates internal radius

        Returns:
            float -- internal radius, m
        """
        r1 = (0.5 * self.width) - self.insulation
        
        return r1

    def amb_temp(self, timestep):
        """ambient temperature surrounding tank

        # if location of storage is inside
        # then a 15 deg ambient condition is assumed
        # elif location is outside then outdoor temperature is used

        Arguments:
            timestep {int} --

        Returns:
            float -- ambient temp surrounding tank degC
        """
        ambient_temp = 15.0

        return ambient_temp

    def discharging_function(self, state, nodes_temp, flow_temp):

        # total nodes is the number of nodes being modelled
        # nodes_temp is a dict of the nodes and their temperatures
        # return_temp is the temperature from the scheme going
        # back into the storage

        total_nodes = self.number_nodes
        node_list = range(total_nodes)
        function = {}
        # bottom_node = total_nodes - 1

        if state == 'discharging':

            for node in node_list:

                # this asks if we are looking at the top node
                # and if the charging water is above this nodes temp
                if node == 0 and flow_temp <= nodes_temp[0]:
                    function[node] = 1

                # if the source temp is lower than
                elif node == 0 and flow_temp >= nodes_temp[0]:
                    function[node] = 0

                # top node then goes in other node
                elif flow_temp < nodes_temp[node] and flow_temp >= nodes_temp[node - 1]:
                    function[node] = 1

                # for out of bounds nodes, shouldnt occur
                elif node < 0 or node == total_nodes + 1:
                    function[node] = 0

                else:
                    function[node] = 0

        elif state == 'charging' or state == 'standby':
            for node in node_list:
                function[node] = 0

        # print (function, 'dis_function')
        return function

    def discharging_bottom_node(self, state, nodes_temp,
                                return_temp, flow_temp):

        """
        Inlet flow to bottom node if discharging
        """

        total_nodes = self.number_nodes
        node_list = range(total_nodes)
        function = {}
        bottom_node = total_nodes - 1

        df = self.discharging_function(state, nodes_temp, flow_temp)
        df_list = []
        for i in range(total_nodes):
            df_list.append(df[i])
            
        # print(df_list)

        if 1 in df_list:

            for node in node_list:

                # this asks if we are looking at the bottom node
                if node == bottom_node and nodes_temp[0] >= flow_temp:
                    function[node] = 1

                elif node == bottom_node and nodes_temp[0] < flow_temp:
                    function[node] = 1

                else:
                    function[node] = 0
                    
                # print(node, function[node])

        else:
            for node in node_list:
                function[node] = 0

        # print (function, 'dis_function_bottom_node')
        return function

    def charging_function(self, state, nodes_temp, source_temp):

        # this determines which node recieves the charging water

        # total nodes is the number of nodes being modelled
        # nodes_temp is a dict of the nodes and their temperatures
        # source_temp is the temperature from the source going into the storage

        # if the in mass exceeds the node volume then next node also charged

        total_nodes = self.number_nodes
        node_list = range(total_nodes)
        function = {}
        # print source_temp
        # print nodes_temp[0]
        if state == 'charging':

            for node in node_list:

                # this asks if we are looking at the top node
                # and if the charging water is above this nodes temp
                if node == 0 and source_temp >= nodes_temp[0]:
                    function[node] = 1

                # if the source temp is lower than
                elif node == 0 and source_temp <= nodes_temp[0]:
                    function[node] = 0

                # top node then goes in other node
                elif source_temp >= nodes_temp[node] and source_temp <= nodes_temp[node - 1]:
                    function[node] = 0

                # for out of bounds nodes, shouldnt occur
                elif node < 0 or node == total_nodes + 1:
                    function[node] = 0

                else:
                    function[node] = 0

        elif state == 'discharging' or 'standby':
            for node in node_list:
                function[node] = 0
        # print (function, 'cha_function')
        return function

    def charging_top_node(self, state):

        function = {}
        for node in range(self.number_nodes):

            if state == 'charging' and node == self.number_nodes - 1:
                function[node] = 1
            else:
                function[node] = 0
        # print function, 'charging_top_node'

        return function

    def mixing_function(self, state, node, nodes_temp,
                        source_temp, flow_temp, source):

        total_nodes = self.number_nodes
        bottom_node = total_nodes - 1

        cf = self.charging_function(state, nodes_temp, source_temp)
        cf_list = []
        for i in range(total_nodes):
            cf_list.append(cf[i])

        df = self.discharging_function(state, nodes_temp, flow_temp)
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
            
        # print (node_charging, node_discharging)

        if state == 'charging' and source == 1: # Heat Pump
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

    def connection_losses(self):

        tank_opening = 5
        tank_opening_d = 35.
        uninsulated_connections = 0
        uninsulated_connections_d = 35.
        insulated_connections = 2.
        insulated_connections_d = 50.8

        # divided by 0.024 to convert from kWh/day to W
        tank_opening_loss = (
            ((tank_opening_d * 0.001) ** 2) *
            math.pi * tank_opening * 27 / (4 * 0.024))

        uninsulated_connections_loss = (
            uninsulated_connections_d * 0.001 *
            uninsulated_connections * 5 / 0.024)

        insulated_connections_loss = (
            insulated_connections_d * 0.001 *
            insulated_connections * 3.5 / 0.024)

        loss = (
            tank_opening_loss +
            uninsulated_connections_loss +
            insulated_connections_loss)

        return loss

    def DH_flow_post_mix(self, demand, flow_temp, return_temp):

        cp1 = self.specific_heat_water(flow_temp) / 1000.0
        mass_DH_flow = demand / (
            cp1 * (flow_temp - return_temp))
        return mass_DH_flow

    def source_out_pre_mix(self, thermal_output, source_temp, source_delta_t):

        cp1 = self.specific_heat_water(source_temp) / 1000.0
        mass_out_pre_mix = thermal_output / (
            cp1 * source_delta_t)
        return mass_out_pre_mix

    def thermal_storage_mass_charging(self, thermal_output, source_temp,
                                      source_delta_t, return_temp, flow_temp,
                                      demand, temp_tank_bottom):

        ts_mass = ((self.source_out_pre_mix(
            thermal_output, source_temp, source_delta_t) *
            source_delta_t -
            self.DH_flow_post_mix(demand, flow_temp, return_temp) *
            (flow_temp - return_temp)) /
            (source_temp - temp_tank_bottom))
        return ts_mass

    def thermal_storage_mass_discharging(self, thermal_output, source_temp,
                                         source_delta_t, return_temp,
                                         flow_temp,
                                         demand, temp_tank_top):

        # mass discharged in every tank timestep, minutely

        ts_mass = ((self.DH_flow_post_mix(demand, flow_temp, return_temp) *
                   (flow_temp - return_temp) -
                    self.source_out_pre_mix(
                        thermal_output, source_temp, source_delta_t) *
                   (source_delta_t)) /
                   (temp_tank_top - return_temp))
        
        # print (temp_tank_top, return_temp, demand, thermal_output)
        return abs(ts_mass)

    def mass_flow_calc(self, state, flow_temp, return_temp,
                       source_temp, source_delta_t, thermal_output, demand,
                       temp_tank_bottom, temp_tank_top):
        if state == 'charging':
            mass_ts = self.thermal_storage_mass_charging(
                thermal_output, source_temp,
                source_delta_t, return_temp, flow_temp,
                demand, temp_tank_bottom)
#            print (mass_ts, temp_tank_bottom, temp_tank_top)

        elif state == 'discharging':
            mass_ts = self.thermal_storage_mass_discharging(
                thermal_output, source_temp, source_delta_t,
                return_temp, flow_temp, demand, temp_tank_top)
            
        elif state == 'standby':
            mass_ts = 0
            
#        mass_ts = max(0, mass_ts)
        return mass_ts

    def coefficient_A(self, state, node, nodes_temp, mass_flow,
                      source_temp, flow_temp, source, MTS):
        node_mass = self.calc_node_mass()
        # print node_mass, 'node_mass'

        # specific heat at temperature of node i
        cp = self.specific_heat_water(nodes_temp[node])

        # print cp, 'cp'

        # thermal conductivity of insulation material
        k = self.insulation_k_value()
        # print k, 'k'

        # dimensions
        r1 = self.internal_radius()
        # print r1, 'r1'
        r2 = 0.5 * self.width
        # print r2, 'r2'
        h = self.height
        # print h, 'h'

        # correction factors
        Fi = self.ins_fac
        # print Fi, 'Fi'
        Fe = self.ov_fac
        # print Fe, 'Fe'

        Fd = self.discharging_function(state, nodes_temp, flow_temp)[node]
        # print Fd, 'Fd'
        # print Fd, 'Fd'

        mf = self.mixing_function(state, node, nodes_temp,
                                  source_temp, flow_temp, source)
        # print mf

        Fco = self.charging_top_node(state)[node]
        # print Fco, 'Fco'

#        A = (- Fd * mass_flow * cp -
#             mf['Fdnt'] * mass_flow * cp -
#             mf['Fcnb'] * mass_flow * cp -
#             Fco * mass_flow * cp -
#             Fe * Fi * k * ((1) / (r2 - r1)) *
#             math.pi * ((r1 ** 2) + h * (r2 + r1))
#             ) / (node_mass * cp)
        
        A = ((- Fd * mass_flow * cp) -
             (mf['Fdnt'] * mass_flow * cp) -
             (mf['Fcnb'] * mass_flow * cp) -
             (Fco * mass_flow * cp) -
             ((3600. / (self.number_nodes - 1.)) * Fe * Fi * k * ((1) / (r2 - r1)) *
             math.pi * (((2. / self.number_nodes) * (r1 ** 2)) + (h / self.number_nodes) * (r2 + r1)))
             ) / (node_mass[node] * cp)
        
#        print(state, node, mf['Fcnb'])
        
        # if source != 3:
        #     A = (- Fd * mass_flow * cp -
        #          mf['Fdnt'] * mass_flow * cp -
        #          mf['Fcnb'] * mass_flow * cp -
        #          Fco * mass_flow * cp
        #          ) / (node_mass * cp)
        # else:
        #     # A = ((-MTS / 60.) * (Fe * Fi * k * ((1) / (r2 - r1)) *
        #     #      math.pi * ((r1 ** 2) + h * (r2 + r1))
        #     #      )) / (node_mass * cp)
            
        #     A = (((-3600. / (self.number_nodes - 1.)) * Fe * Fi * k * ((1) / (r2 - r1)) *
        #      math.pi * (((2. / self.number_nodes) * (r1 ** 2)) + (h / self.number_nodes) * (r2 + r1)))
        #      ) / (node_mass * cp)
            
        return A

    def coefficient_B(self, state, node, nodes_temp, mass_flow, source_temp,
                      flow_temp, source):

        node_mass = self.calc_node_mass()
        mf = self.mixing_function(state, node, nodes_temp,
                                  source_temp, flow_temp, source)

        B = mf['Fcnt'] * mass_flow / node_mass[node]

        return B

    def coefficient_C(self, state, node, nodes_temp, mass_flow, source_temp,
                      flow_temp, source):
        node_mass = self.calc_node_mass()
        mf = self.mixing_function(state, node, nodes_temp,
                                  source_temp, flow_temp, source)

        C = mf['Fdnb'] * mass_flow / node_mass[node]
        return C

    def coefficient_D(self, state, node, nodes_temp, mass_flow, source_temp,
                      flow_temp, return_temp, timestep, source, MTS):

        node_mass = self.calc_node_mass()

        # specific heat at temperature of node i
        cp = self.specific_heat_water(nodes_temp[node])

        # thermal conductivity of insulation material
        k = self.insulation_k_value()

        # dimensions
        r1 = self.internal_radius()
        r2 = 0.5 * self.width
        h = self.height

        # correction factors
        Fi = self.ins_fac
        Fe = self.ov_fac

        Fc = self.charging_function(state, nodes_temp, source_temp)[node]
        Fdi = self.discharging_bottom_node(
            state, nodes_temp, return_temp, flow_temp)[node]
        Ta = self.amb_temp(timestep)

        cl = self.connection_losses()
        
#        D = (Fc * mass_flow * cp * source_temp +
#             Fdi * mass_flow * cp * return_temp +
#             Fe * Fi * k * ((Ta) / (r2 - r1)) * math.pi *
#             ((r1 ** 2) + h * (r2 + r1)) + Fe * cl
#             ) / (node_mass * cp)
        
        D = ((Fc * mass_flow * cp * source_temp) +
             (Fdi * mass_flow * cp * return_temp) +
             (((3600. / (self.number_nodes - 1)) * Fe * Fi * k * ((Ta) / (r2 - r1)) * math.pi *
             (((2. / self.number_nodes) * (r1 ** 2)) + (h / self.number_nodes) * (r2 + r1))) -
             ((3600. / (self.number_nodes - 1.)) * Fe * (cl / self.number_nodes))
             )) / (node_mass[node] * cp)
        
        # if source != 3:
        #     D = (Fc * mass_flow * cp * source_temp +
        #          Fdi * mass_flow * cp * return_temp
        #          ) / (node_mass * cp)
        # else:
        #     # D = ((MTS / 60.) * (Fe * Fi * k * ((Ta) / (r2 - r1)) * math.pi *
        #     #      ((r1 ** 2) + h * (r2 + r1)) + Fe * cl
        #     #      )) / (node_mass * cp)
            
        #     D = (((3600. / (self.number_nodes - 1)) * Fe * Fi * k * ((Ta) / (r2 - r1)) * math.pi *
        #      (((2. / self.number_nodes) * (r1 ** 2)) + (h / self.number_nodes) * (r2 + r1))) - 
        #      ((3600. / (self.number_nodes - 1.)) * Fe * (cl / self.number_nodes))
        #      ) / (node_mass * cp)
            

        return D

    def set_of_coefficients(self, state, nodes_temp, source_temp,
                            source_delta_t, flow_temp, return_temp,
                            thermal_output, demand, temp_tank_bottom,
                            temp_tank_top, timestep, source, MTS):

        mass_flow = self.mass_flow_calc(
            state, flow_temp, return_temp, source_temp, source_delta_t,
            thermal_output, demand, temp_tank_bottom, temp_tank_top)

        c = []
        for node in range(self.number_nodes):
            coefficients = {'A': self.coefficient_A(
                state, node, nodes_temp, mass_flow, source_temp, flow_temp, source, MTS),
                            'B': self.coefficient_B(
                state, node, nodes_temp, mass_flow, source_temp, flow_temp, source),
                            'C': self.coefficient_C(
                state, node, nodes_temp, mass_flow, source_temp, flow_temp, source),
                            'D': self.coefficient_D(
                state, node, nodes_temp, mass_flow, source_temp, flow_temp,
                return_temp, timestep, source, MTS)}
            c.append(coefficients)
        return c

    def new_nodes_temp(self, state, nodes_temp, source_temp,
                       source_delta_t, flow_temp, return_temp,
                       thermal_output, demand, timestep, source, MTS):
        
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
        t = self.number_nodes - 1

        # node indexes
        top = 0
        bottom = self.number_nodes - 1

        # initial condition of coefficients
        coefficients = []
        # divide thermal output and demand accross timesteps
        # convert from kWh to kJ
        thermal_output = thermal_output * 3600. / float(t)
        demand = demand * 3600. / float(t)
        
        nodes_temp[bottom] = min(source_temp - 0.01, nodes_temp[bottom])

        coefficients.append((self.set_of_coefficients(
            state, nodes_temp, source_temp, source_delta_t, flow_temp,
            return_temp, thermal_output, demand,
            nodes_temp[bottom], nodes_temp[top], timestep, source, MTS)))

        node_temp_list = []
        node_temp_list.append(nodes_temp)

        # solve ODE
        for i in range(1, t + 1):
            nodes_temp[bottom] = min(source_temp - 0.01, nodes_temp[bottom])
            # span for next time step
            tspan = [i - 1, i]
            # solve for next step
            # new coefficients
            coefficients.append((self.set_of_coefficients(
                state, nodes_temp, source_temp, source_delta_t,
                flow_temp, return_temp, thermal_output, demand,
                nodes_temp[bottom], nodes_temp[top], timestep, source, MTS)))
            z = odeint(
                model_temp, nodes_temp, tspan,
                args=(coefficients[i],))

            nodes_temp = z[1]
            nodes_temp = sorted(nodes_temp, reverse=True)
            node_temp_list.append(nodes_temp)

        return node_temp_list

    def coefficient_A_max(self, state, node, nodes_temp, source_temp,
                          flow_temp, source):

        node_mass = self.calc_node_mass()
        # print node_mass, 'node_mass'

        # specific heat at temperature of node i
        cp = self.specific_heat_water(nodes_temp[node])
        # print cp, 'cp'

        # thermal conductivity of insulation material
        k = self.insulation_k_value()
        # print k, 'k'

        # dimensions
        r1 = self.internal_radius()
        # print r1, 'r1'
        r2 = 0.5 * self.width
        # print r2, 'r2'
        h = self.height
        # print h, 'h'

        # correction factors
        Fi = self.ins_fac
        # print Fi, 'Fi'
        Fe = self.ov_fac
        # print Fe, 'Fe'

        Fd = self.discharging_function(state, nodes_temp, flow_temp)[node]
        # print Fd, 'Fd'
        # print Fd, 'Fd'
        mf = self.mixing_function(state, node, nodes_temp,
                                  source_temp, flow_temp, source)
        # print mf

        Fco = self.charging_top_node(state)[node]
        # print Fco, 'Fco'

        mass_flow = self.calc_node_mass()

        # A = (- Fd * mass_flow * cp -
        #      mf['Fdnt'] * mass_flow * cp -
        #      mf['Fcnb'] * mass_flow * cp -
        #      Fco * mass_flow * cp -
        #      Fe * Fi * k * ((1) / (r2 - r1)) *
        #      math.pi * ((r1 ** 2) + h * (r2 + r1))
        #      ) / (node_mass * cp)
        
        A = ((- Fd * mass_flow[node] * cp) -
             (mf['Fdnt'] * mass_flow[node] * cp) -
             (mf['Fcnb'] * mass_flow[node] * cp) -
             (Fco * mass_flow[node] * cp) -
             ((3600. / (self.number_nodes - 1.)) * Fe * Fi * k * ((1) / (r2 - r1)) *
             math.pi * (((2. / self.number_nodes) * (r1 ** 2)) + (h / self.number_nodes) * (r2 + r1)))
             ) / (node_mass[node] * cp)
             
#        print(1,state, node, Fd, mf['Fdnt'], mf['Fcnb'])

        return A

    def coefficient_B_max(self, state, node, nodes_temp, source_temp,
                          flow_temp, source):

        node_mass = self.calc_node_mass()
        mf = self.mixing_function(state, node, nodes_temp,
                                  source_temp, flow_temp, source)
        mass_flow = self.calc_node_mass()

        B = mf['Fcnt'] * mass_flow[node] / node_mass[node]
        
#        print(2,state, node, mf['Fcnt'])

        return B

    def coefficient_C_max(self, state, node, nodes_temp, source_temp,
                          flow_temp, timestep, source):

        node_mass = self.calc_node_mass()
        mf = self.mixing_function(state, node, nodes_temp,
                                  source_temp, flow_temp, source)
        mass_flow = node_mass[node]

        C = mf['Fdnb'] * mass_flow / node_mass[node]
        
#        print(3,state, node, mf['Fdnb'])

        return C

    def coefficient_D_max(self, state, node, nodes_temp, source_temp,
                          flow_temp, return_temp, timestep):

        node_mass = self.calc_node_mass()

        # specific heat at temperature of node i
        cp = self.specific_heat_water(nodes_temp[node])

        # thermal conductivity of insulation material
        k = self.insulation_k_value()

        # dimensions
        r1 = self.internal_radius()
        r2 = 0.5 * self.width
        h = self.height

        # correction factors
        Fi = self.ins_fac
        Fe = self.ov_fac

        Fc = self.charging_function(state, nodes_temp, source_temp)[node]
        Fdi = self.discharging_bottom_node(
            state, nodes_temp, return_temp, flow_temp)[node]
        Ta = self.amb_temp(timestep)

        cl = self.connection_losses()

        mass_flow = node_mass[node]

        # print source_temp, 'here?'
        # print return_temp

        # D = (Fc * mass_flow * cp * source_temp +
        #      Fdi * mass_flow * cp * return_temp +
        #      Fe * Fi * k * ((Ta) / (r2 - r1)) * math.pi *
        #      ((r1 ** 2) + h * (r2 + r1)) + Fe * cl
        #      ) / (node_mass * cp)
        
        D = ((Fc * mass_flow * cp * source_temp) +
             (Fdi * mass_flow * cp * return_temp) +
             (((3600. / (self.number_nodes - 1)) * Fe * Fi * k * ((Ta) / (r2 - r1)) * math.pi *
             (((2. / self.number_nodes) * (r1 ** 2)) + (h / self.number_nodes) * (r2 + r1))) -
             ((3600. / (self.number_nodes - 1.)) * Fe * (cl / self.number_nodes))
             )) / (node_mass[node] * cp)
        
#        print(4,state, node, Fc, Fdi)

        return D

    def set_of_max_coefficients(self, state, nodes_temp, source_temp,
                                flow_temp, return_temp, timestep, source):

        c = []
        for node in range(self.number_nodes):
            coefficients = {'A': self.coefficient_A_max(
                state, node, nodes_temp, source_temp, flow_temp, source),
                            'B': self.coefficient_B_max(
                state, node, nodes_temp, source_temp, flow_temp, source),
                            'C': self.coefficient_C_max(
                state, node, nodes_temp, source_temp, flow_temp,
                timestep, source),
                            'D': self.coefficient_D_max(
                state, node, nodes_temp, source_temp, flow_temp,
                return_temp, timestep)}
            c.append(coefficients)
            
        return c

    def max_energy_in_out(self, state, nodes_temp, source_temp,
                          flow_temp, return_temp, timestep, MTS, source):

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

        # initial condition of coefficients
        coefficients = []
        coefficients.append((self.set_of_max_coefficients(
            state, nodes_temp, source_temp,
            flow_temp, return_temp, timestep, source)))

        energy_list = []
        mass_flow = self.calc_node_mass()[0]

        cp = self.specific_heat_water(source_temp)

        # solve ODE
        for i in range(1, t + 1):
            
#            print(2,nodes_temp)
#            
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
                #     mass_flow = self.calc_node_mass() * (1. - ((flow_temp - nodes_temp[1])/(nodes_temp[0] - nodes_temp[1])))
                #     energy = mass_flow * cp * (
                #             nodes_temp[0] - return_temp)
                
#                print (mass_flow, cp, nodes_temp[0], return_temp, energy)

            else:
                energy = 0

                
            energy_list.append(energy)

            # span for next time step
            tspan = [i - 1, i]
            # solve for next step
            # new coefficients
            coefficients.append((self.set_of_max_coefficients(
                state, nodes_temp, source_temp,
                flow_temp, return_temp, timestep, source)))

            z = odeint(
                model_temp, nodes_temp, tspan,
                args=(coefficients[i],))
            
#            print(3,nodes_temp)
            
            nodes_temp = z[1]
            
#            print(4,nodes_temp)
            
#            print (49,t,i,energy,state)
#            print (z)

        # convert J to kWh by divide by 3600000
        energy_total = sum(energy_list) / 3600000

        return energy_total
