# -*- coding: utf-8 -*-
"""
Created on Wed July  24  2024

@author: LS276867
"""

import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import xpart as xp
import sys
sys.path.append('/mnt/c/muco/code')
from fodo_hybrid_time_var import GradientHFODO
from sklearn.linear_model import LinearRegression
import os
from tools.save_para_rcs import save_para
from tools.load_file import load_file
import json
from pathlib import Path

plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

step_quad = 1e-8

class DS_Hybrid:
    def __init__(self, gradient, n_turn_rcs, add_quad, save_fig_path=False):
        self.gradient = gradient
        self.qp_para = gradient.qp_para
        self.rcs = gradient.rcs
        self.option = gradient.option
        self.n_turn_rcs = n_turn_rcs
        self.add_quad = add_quad
        self.save_fig_path = save_fig_path

        if 'gradient_inj' in self.qp_para['qp_nc'].keys():
            self.t_g0 = self.qp_para['qp_nc']['gradient_inj']/(self.qp_para['qp_nc']['gradient_inj']-self.qp_para['qp_nc']['gradient'])
        else:
            self.t_g0 = 0.5

        self.list_qp_DS = np.arange(self.rcs.nb_cell_arc - 3, self.rcs.nb_cell_arc + 3)

        self.line_FODO_tg0, self.line_tr_tg0 = fodo_hybrid_qp_symm1(RCS=self.rcs, quad_dico=self.qp_para, n_turn_rcs=self.n_turn_rcs, 
                            time_frac=self.t_g0, option=self.gradient.option, add_quad=self.add_quad)

        tw_FODO = self.line_FODO_tg0.twiss(method='4d')
        if self.rcs.nb_cell_arc%2 == 0:
            tw_init_FODO = tw_FODO.get_twiss_init('qp_sc_foc2')
        else:
            tw_init_FODO = tw_FODO.get_twiss_init('qp_sc_defoc2')

        tw_init_FODO.element_name = self.line_tr_tg0.element_names[0]

        self.knobs_qp_sc = self.attach_knobs(self.line_tr_tg0)
        self.match_tr(self.line_tr_tg0, tw_init_FODO)

    def gradient_nc(self, time : float):
        """Return the gradient in NC quad at a given time

        Args:
            time (float): time normalized between 0 and 1.

        Returns:
            float: NC gradient
        """
        if 'gradient_inj' in self.qp_para['qp_nc'].keys():
            g_nc_time = self.qp_para['qp_nc']['gradient_inj'] + (
            self.qp_para['qp_nc']['gradient'] - self.qp_para['qp_nc']['gradient_inj'])*time

        else:
            g_nc_time = self.qp_para['qp_nc']['gradient']*(2*time-1)

        return g_nc_time

    def adapt_DS_time(self, time : float):
        """Scale the quadrupoles from the reference at tg0 (when G_nc=0).

        Args:
            time (float): Time normalized between 0 and 1

        Raises:
            ValueError: Raise error if quad not named after nc or sc. 

        Returns:
            line_tr, tw_init_FODO: transfer line and twiss init. 
        """
        g_nc_time = self.gradient_nc(time) 

        line_fodo_ad, line_tr_ad = fodo_hybrid_qp_symm1(RCS=self.rcs, quad_dico=self.qp_para, n_turn_rcs=self.n_turn_rcs, 
                                        time_frac=time, option=self.option, add_quad=self.add_quad)
        
        tab_fodo = line_fodo_ad.get_table()
        for quad in tab_fodo.rows[tab_fodo.element_type == 'Quadrupole'].name:
            # Only need to scale in energy from the constant SC gradient
            if 'sc' in quad:
                line_fodo_ad[quad].k1 = self.line_FODO_tg0[quad].k1*self.rcs.func_Brho(self.t_g0)/self.rcs.func_Brho(time)
            # Need to ramp the gradient and scale in energy
            elif 'nc' in quad:
                if 'defoc' in quad:
                    foc = -1
                else:
                    foc = 1
                line_fodo_ad[quad].k1 = foc*g_nc_time/self.rcs.func_Brho(time)
            else:
                raise ValueError('Quad is not named after nc or sc.')

        tw_FODO = line_fodo_ad.twiss4d()

        if self.rcs.nb_cell_arc%2 == 0:
            tw_init_FODO = tw_FODO.get_twiss_init('qp_sc_foc2')
        else:
            tw_init_FODO = tw_FODO.get_twiss_init('qp_sc_defoc2')
        
        tab_tr = line_tr_ad.get_table()
        for quad in tab_tr.rows[tab_tr.element_type == 'Quadrupole'].name:
            if 'sc' in quad:
                line_tr_ad[quad].k1 = self.line_tr_tg0[quad].k1 * self.rcs.func_Brho(self.t_g0) / self.rcs.func_Brho(time)
                if self.line_tr_tg0[quad].k1 > 0:
                    foc = 1
                else:
                    foc = -1
            elif 'nc' in quad:
                for key in self.knobs_qp_sc.keys():
                    num_qd = str(self.knobs_qp_sc[key])
                    if num_qd in quad:
                        line_tr_ad[f'qp_nc{num_qd}'].k1 = (1-self.rcs.func_Brho(self.t_g0)/self.rcs.func_Brho(time))*(
                        self.line_tr_tg0[f'qp_sc{num_qd}'].k1*self.gradient.l_sc/self.gradient.l_nc)
                        break
                else:
                    line_tr_ad[quad].k1 = foc*g_nc_time/self.rcs.func_Brho(time)
            else:
                raise ValueError('Quad is not named after nc or sc.')

        tw_init_FODO.element_name = line_tr_ad.element_names[0]

        return line_tr_ad, tw_init_FODO

    def attach_knobs(self, line):
        """Attch knobs to SC quadrupoles. 

        Args:
            line (_type_): line
        Raises:
            ValueError: Raise error if number of cells per arc < 4. 

        Returns:
            dict : knobs name and quad number. 
        """
        n_qd_match = self.rcs.nb_cell_arc + 2
        dico_knobs = {}
        if self.rcs.nb_cell_arc < 4 :
            raise ValueError(f'DS matching not possible for {self.rcs.nb_cell_arc} cells per arc')
        if self.rcs.nb_cell_arc >= 4:
            for i in np.arange(0,6):
                line.vars[f'ks{i+1}'] = line[f'qp_sc{n_qd_match-i}'].k1
                line.element_refs[f'qp_sc{n_qd_match-i}'].k1 = line.vars[f'ks{i+1}']
                dico_knobs[f'ks{i+1}'] = int(n_qd_match-i)
        if self.rcs.nb_cell_arc >= 5:
            line.vars['ksa'] = line[f'qp_sc{n_qd_match-6}'].k1
            for i in np.arange(6, n_qd_match, 2):
                line.element_refs[f'qp_sc{n_qd_match-i}'].k1 = line.vars[f'ksa']
        if self.rcs.nb_cell_arc >= 6:
            line.vars['ksb'] = line[f'qp_sc{n_qd_match-7}'].k1
            for i in np.arange(7, n_qd_match, 2):
                line.element_refs[f'qp_sc{n_qd_match-i}'].k1 = line.vars['ksb']
        return dico_knobs
    
    def match_tr(self, line, tw_init_FODO):
        """Match DS using SC quads. 

        Args:
            line (_type_): transfer line to match. 
            tw_init_FODO (_type_): twiss init. 

        Returns:
            _type_: matcher.
        """
        matcher = line.match(
                start = line.element_names[0],
                end = line.element_names[-1],
                init = tw_init_FODO,
                vary = [xt.VaryList([
                                    'ks1',
                                    'ks2',
                                    'ks3',
                                    'ks4',
                                    'ks5',
                                    'ks6'
                                    ], step=1e-8, tag='quad')],
                targets=[xt.TargetSet(dx=0, dpx=0, alfx=0, alfy=0, at='_end_point', tol=1e-8)], 
                method='4d',
                solve=False
                )

        matcher.step(50)
        # print('RESULTS FROM TR MATCHING')
        # matcher.target_status()
        # matcher.vary_status()

        return matcher
    
    def scaling_from_tg0(self, t_scan = np.linspace(0,1,6)):
        """Build the DS at tg0 (when G_NC=0) and scales in energy for the other time.
        Keep the k_nc*L_nc + k_sc*L_sc = constant. 

        Args:
            t_scan (np.array, optional): Time normalized between 0 and 1. Defaults to np.linspace(0,1,5).
        """
        self.tw_sacling_tg0 = []
        for t in t_scan:
            line_tr_opt, tw_init_FODO = self.adapt_DS_time(time=t)
            self.tw_sacling_tg0.append(line_tr_opt.twiss4d(init=tw_init_FODO))

        self.index_RFins_beg_arc = line_tr_opt.element_names.index('drift_RF_ins_1')
        fig_dx_rf = plot_dx_RF(line_tr_opt, t_scan, self.tw_sacling_tg0, '', 'drift_RF_ins_1')

        fig = plot_n_twiss(t_scan, self.tw_sacling_tg0)
        if self.save_fig_path:
            fig.savefig(self.save_fig_path + '/scaling_tg0/twiss.png', bbox_inches='tight')


    def build_tr_time(self, time : float, get_bet = False, DS_half_bend = False):
        """Build transfer line at a given time. 

        Args:
            time (float): time normalized between 0 and 1. 

        Returns:
            line_tr, tw_init_FODO: transfer line line_tr, and init twiss object. 
        """          

        line_fodo_opt, line_tr_opt = fodo_hybrid_qp_symm1(RCS=self.rcs, quad_dico=self.qp_para, n_turn_rcs=self.n_turn_rcs, 
                                        time_frac=time, option=self.option, add_quad=self.add_quad, half_bend=DS_half_bend)
        tw_FODO = line_fodo_opt.twiss4d()
        if self.rcs.nb_cell_arc%2 == 0:
            tw_init_FODO = tw_FODO.get_twiss_init('qp_sc_foc2')
            # tw_init_FODO = tw_FODO.get_twiss_init('qp_sc_foc')
        else:
            tw_init_FODO = tw_FODO.get_twiss_init('qp_sc_defoc2')
            # tw_init_FODO = tw_FODO.get_twiss_init('qp_sc_defoc1')
        tw_init_FODO.element_name = line_tr_opt.element_names[0]

        betx_foc = tw_FODO['betx', 'qp_sc_foc1']
        bety_foc = tw_FODO['bety', 'qp_sc_foc1']

        if get_bet:
            return line_tr_opt, tw_init_FODO, betx_foc, bety_foc
        else:
            return line_tr_opt, tw_init_FODO

    
    def apply_halfbend(self, line, n_FODO : int = 2):
        """ Divide the bending field of the dipoles in the n_FODO last FODO cells. 

        Args:
            line (_type_): Xsuite line object
            n_FODO (int, optional): Number of FODO cells. Defaults to 2.
        """
        tab_tr = line.get_table()
        n_dip_halfB = self.rcs.nd*2*n_FODO
        name_dip_halfB = tab_tr.rows[tab_tr.element_type == 'Bend'].name[-n_dip_halfB:]
        for dip in name_dip_halfB:
            line[dip].k0 /= 2
            line[dip].angle /= 2

    def DS_half_bend(self, t_scan = np.linspace(0, 1, 6)):
        """Construct the DS by dividing bending field by 2 in the last 2 FODO cells. 
        Plot the dispersion function of the transfer line. 

        Args:
            t_scan (np.array, optional): Time to build the DS. Defaults to np.linspace(0, 1, 5).
        """
        self.tw_hbend = []
        for t in t_scan:
            line, tw_init_FODO = self.build_tr_time(t)
            print('Theta before ', line.survey().theta[-1])
            line, tw_init_FODO = self.build_tr_time(t, DS_half_bend=True)
            self.apply_halfbend(line)
            self.tw_hbend.append(line.twiss4d(init=tw_init_FODO))
            print('Theta after ', line.survey().theta[-1])

        fig = plot_n_twiss(t_scan, self.tw_hbend)
        fig_dx_rf = plot_dx_RF(line, t_scan, self.tw_hbend, '', 'drift_RF_ins_1')
        if self.save_fig_path:
            fig.savefig(self.save_fig_path + '/half_bend/twiss.png', bbox_inches='tight')
            fig_dx_rf.savefig(self.save_fig_path+'/half_bend/dx_RF.png', bbox_inches='tight')


    def DS_half_bend_quad(self, t_scan = np.linspace(0,1,6)):
        """Construct the DS by dividing bending field by 2 in the last 2 FODO cells and adapting the QP
        at all energies, then applying linear reg. 

        Args:
            t_scan (np.array, optional): Time to scan. Defaults to np.linspace(0,1,5).
        """
        self.tw_hbend_quad = []
        self.hbend_k = {}
        for i_qp in self.list_qp_DS:
            self.hbend_k['qp_nc'+ str(i_qp)] = []
        for t in t_scan:
            line, tw_init_FODO = self.build_tr_time(t)
            print('Theta before ', line.survey().theta[-1])
            line, tw_init_FODO = self.build_tr_time(t, DS_half_bend=True)
            self.apply_halfbend(line)
            res = self.match_ds_NC(line, tw_init_FODO)
            self.tw_hbend_quad.append(line.twiss4d(init=tw_init_FODO))
            print('Theta after ', line.survey().theta[-1])
            for key in res.keys():
                self.hbend_k[key].append(res[key])

        fig_twiss0 = plot_n_twiss(t_scan, self.tw_hbend_quad)

        fig_k = self.plot_k(t_scan, self.hbend_k)
        k_fit, g_fit, fig_gradient = self.gradient_fit(t_scan, self.hbend_k)

        tw_fit = []
        for i_t, t in enumerate(t_scan):
            line, tw_init_FODO = self.build_tr_time(t)
            self.apply_halfbend(line)
            for key in self.hbend_k:
                line[key].k1 = k_fit[key][i_t]
            tw_fit.append(line.twiss4d(init=tw_init_FODO))

        fig_twiss1 = plot_n_twiss(t_scan, tw_fit)
        fig_dx_rf = plot_dx_RF(line, t_scan, tw_fit, '', 'drift_RF_ins_1')

        if self.save_fig_path:
            fig_k.savefig(self.save_fig_path + '/half_bend_qp/k_matched.png', bbox_inches='tight')
            fig_gradient.savefig(self.save_fig_path + '/half_bend_qp/gradient_matched_linreg.png', bbox_inches='tight')
            fig_twiss0.savefig(self.save_fig_path + '/half_bend_qp/twiss_matched.png', bbox_inches='tight')
            fig_twiss1.savefig(self.save_fig_path + '/half_bend_qp/twiss_linreg.png', bbox_inches='tight')
            fig_dx_rf.savefig(self.save_fig_path + '/half_bend_qp/dx_RF_linreg.png', bbox_inches='tight')

    def plot_k(self, t_scan, k_ds):
        """Plot quadrupole k1 [1/m²].

        Args:
            t_scan (np.array): Time normalized between 0 and 1.
            k_ds (dict): Quad k1 value as a function of time.
        """
        fig, ax = plt.subplots()
        for key in k_ds.keys():
            k = k_ds[key]
            ax.plot(t_scan, k, label=key)
        ax.set_xlabel('Time normalized')
        ax.set_ylabel(r'k1 [$m^{-2}$]')
        ax.legend(fontsize=10)
        fig.show()

        return fig

    def plot_gradient(self, t_scan, k_ds):
        """Plot gradient [T/m].

        Args:
            t_scan (np.array): Time normalized between 0 and 1.
            k_ds (dict): Quad k1 value as a function of time.
        """
        list_Brho = np.array([self.rcs.func_Brho(t) for t in t_scan])
        fig, ax = plt.subplots()
        for key in k_ds.keys():
            k = k_ds[key]
            ax.plot(t_scan, np.array(k) * list_Brho, label=key)
        ax.set_xlabel('Time normalized')
        ax.set_ylabel(r'Gradient [$T/m$]')
        ax.legend(fontsize=10)
        fig.show()

        return fig
    
    def match_ds_NC(self, line, tw_init_FODO):
        """ Match the DS at the end of a transfer line using quadrupoles k1. 
        Returns the strength k1 of each quad in a dict. 

        Args:
            line (_type_): transfer line to match.
            tw_init_FODO (_type_): twiss init object.

        Returns:
            dict: quad k1 after matching
        """
        self.attach_knobs_NC(line)
        self.match_tr_NC(line, tw_init_FODO)

        results_k = {}
        for i_qp in self.list_qp_DS:
            qp_name = 'qp_nc' + str(i_qp)
            results_k[qp_name] = line[qp_name].k1

        return results_k
    
    def match_ds_NC_beta(self, line, tw_init_FODO, betx, bety):
        """ Match the DS at the end of a transfer line using quadrupoles k1. 
        Returns the strength k1 of each quad in a dict. 

        Args:
            line (_type_): transfer line to match.
            tw_init_FODO (_type_): twiss init object.

        Returns:
            dict: quad k1 after matching
        """
        self.attach_knobs_NC(line)
        self.match_tr_NC_beta(line, tw_init_FODO, betx, bety)

        results_k = {}
        for i_qp in self.list_qp_DS:
            qp_name = 'qp_nc' + str(i_qp)
            results_k[qp_name] = line[qp_name].k1

        return results_k
    
    
    def attach_knobs_NC(self, line):
        """ Attach knobs to the NC quadrupoles"""
        for n_k, i_qp in enumerate(self.list_qp_DS):
            line.vars[f'kn{n_k+1}'] = line[f'qp_nc{str(i_qp)}'].k1
            line.element_refs[f'qp_nc{str(i_qp)}'].k1 = line.vars[f'kn{n_k+1}']

    def match_tr_NC(self, line, tw_init_FODO):
        """Function that matches the dispersion and alpha for the DS.

        Args:
            line (_type_): transfer line to match.
            tw_init_FODO (_type_): twiss init object.

        Returns:
            _type_: Matcher object. 
        """
        matcher = line.match(
                start = line.element_names[0],
                end = line.element_names[-1], 
                init = tw_init_FODO,
                vary = [xt.VaryList([
                                    'kn1', 
                                    'kn2',
                                    'kn3',
                                    'kn4',
                                    'kn5',
                                    'kn6'
                                    ], step=1e-8, tag='quad')],
                targets=[xt.TargetSet(dx=0, dpx=0, alfx=0, alfy=0, at='_end_point', tol=1e-5)], 
                method='4d',
                solve=False
                )

        matcher.step(50)
        # print('RESULTS FROM TR MATCHING, USING NC QUADRUPOLES')
        # matcher.target_status()
        # matcher.vary_status()

        return matcher
    
    def match_tr_NC_beta(self, line, tw_init_FODO, betax, betay):
        """Function that matches the dispersion and alpha for the DS.

        Args:
            line (_type_): transfer line to match.
            tw_init_FODO (_type_): twiss init object.

        Returns:
            _type_: Matcher object. 
        """
        matcher = line.match(
                start = line.element_names[0],
                end = line.element_names[-1], 
                init = tw_init_FODO,
                vary = [xt.VaryList([
                                    'kn1', 
                                    'kn2',
                                    'kn3',
                                    'kn4',
                                    'kn5',
                                    'kn6'
                                    ], step=1e-8, tag='quad')],
                targets=[xt.TargetSet(dx=0, dpx=0, alfx=0, alfy=0, at='_end_point', tol=1e-5),
                         xt.TargetSet(betx=betax, bety=betay, at=line.element_names[-1], tol=1e-5)], 
                method='4d',
                solve=False
                )

        matcher.step(50)
        # print('RESULTS FROM TR MATCHING, USING NC QUADRUPOLES')
        matcher.target_status()
        # matcher.vary_status()

        return matcher
    
    def adapt_qp(self, t_scan = np.linspace(0,1,6)):
        """Build DS only using NC QP.
        Scan for different time and get strength/gradient required in quad to build the DS.  

        Args:
            t_scan (np.array, optional): Time to scan. Defaults to np.linspace(0,1,10).
        """
        self.only_quad_k = {}
        self.tw_only_quad = []
        # t_scan = np.array([1])
        for i_qp in self.list_qp_DS:
            self.only_quad_k['qp_nc' + str(i_qp)]=[]
        for t in t_scan:
            line, tw_init_FODO, betx_beg, bety_beg = self.build_tr_time(t, get_bet=True)
            res = self.match_ds_NC(line, tw_init_FODO)
            # res = self.match_ds_NC_beta(line, tw_init_FODO, betx_beg, bety_beg) #Try to match the beta at the end
            self.tw_only_quad.append(line.twiss4d(init=tw_init_FODO))
            for key in res.keys():
                self.only_quad_k[key].append(res[key])
            
        fig_twiss0 = plot_n_twiss(t_scan, self.tw_only_quad)
        fig_k = self.plot_k(t_scan, self.only_quad_k)

        k_fit, g_fit, fig_gradient = self.gradient_fit(t_scan, self.only_quad_k)
        self.tw_only_quad_linreg = []
        self.line_arc_only_qp = []
        self.tw_tr_onlyquad = []
        self.tscan_only_qp = t_scan
        for i_t, t in enumerate(t_scan):
            line, tw_init_FODO = self.build_tr_time(t)
            for key in self.only_quad_k:
                line[key].k1 = k_fit[key][i_t]
            self.tw_tr_onlyquad.append(line.twiss4d(init=tw_init_FODO))

            line1 = -line
            nd = len(self.rcs.pattern)-1
            for id, name in enumerate(line1.element_names):
                if ('BSC' in name) or ('BNC' in name):
                    line1.element_names[id] = name[:3] + str(nd-int(name[3])) + name[4:]

            line_arc = xt.Line(elements = line1.elements + line.elements,
                    element_names = [el + '_l' for el in line1.element_names] + [
                                    el + '_r' for el in line.element_names])

            line_arc.particle_ref = line.particle_ref
            # tw_fit.append(line.twiss4d(init=tw_init_FODO))
            # tw = line_arc.twiss(matrix_stability_tol=0.1)
            # self.tw_only_quad_linreg.append(tw)
            # self.line_arc_only_qp.append(line_arc)

        fig_twiss_tr = plot_n_twiss(t_scan, self.tw_tr_onlyquad)
        fig_twiss1 = plot_n_twiss(t_scan, self.tw_only_quad_linreg)
        self.index_RFins_beg_arc = line.element_names.index('drift_RF_ins_1')
        fig_dx_rf = plot_dx_RF(line, t_scan, self.tw_tr_onlyquad, '', 'drift_RF_ins_1')
        # fig_dx_rf = plot_dx_RF(line_arc, t_scan, self.tw_only_quad_linreg, '', 'drift_RF_ins_1_r')


        if self.save_fig_path:
            fig_k.savefig(self.save_fig_path + '/qp_only/k_matched.png', bbox_inches='tight')
            fig_gradient.savefig(self.save_fig_path + '/qp_only/gradient_matched_linreg.png', bbox_inches='tight')
            fig_twiss0.savefig(self.save_fig_path + '/qp_only/twiss_matched.png', bbox_inches='tight')
            fig_twiss1.savefig(self.save_fig_path + '/qp_only/twiss_linreg.png', bbox_inches='tight')
            fig_dx_rf.savefig(self.save_fig_path + '/qp_only/dx_RF_linreg.png', bbox_inches='tight')


    def gradient_fit(self, t_scan, k_data):
        """Make linear regression on the gradient. 
        Returns the fitted strength k1 and gradient. 

        Args:
            t_scan (np.array): Time normalized between 0 and 1. 
            k_data (dict): Matched strength k1 of each quadrupoles. 

        Returns:
            k_fit, g_fit: dict of fitted strength k1 and gradient. 
        """
        g_fit = {}
        k_fit = {}
        list_Brho = np.array([self.rcs.func_Brho(t) for t in t_scan])
        for key in k_data:
            model = LinearRegression().fit(t_scan.reshape((-1, 1)), np.array(k_data[key])*list_Brho)
            g_fit[key] =  model.intercept_ + model.coef_*t_scan
            k_fit[key] = g_fit[key] / list_Brho
    
        fig, ax = plt.subplots()  # Create a figure and an axes.

        for i, key in enumerate(k_data):
            exact = ax.plot(t_scan, k_data[key] * list_Brho, label=rf'Qp{i}')
            color = exact[0].get_color()
            ax.scatter(t_scan, g_fit[key], marker='+', color=color)

        ax.set_xlabel(r'$t \ / \ t_{acc}$')
        ax.set_ylabel(r'Gradient [T/m]')
        ax.legend(fontsize=10, loc='lower center', ncol=2)
        fig.show()

        return k_fit, g_fit, fig


def plot_n_twiss(t_scan, tw_list):
    """ Plot beta and dispersion functions for a list of twiss objects.

    Args:
        t_scan (np.array): Time normalized between 0 and 1.
        tw_list (list): list of twiss objects.
    """
    fig = plt.figure()
    fig.set_size_inches(12, 8)  # Adjust the width and height of the figure

    betx = plt.subplot(3, 1, 1)
    bety = plt.subplot(3, 1, 2, sharex=betx)
    disp = plt.subplot(3, 1, 3, sharex=betx)

    lines = []
    labels = []

    # Use the 'tab10' colormap
    cmap = plt.get_cmap('tab10')

    for i, t in enumerate(t_scan):
        color = cmap(i % 10)  # Cycle through colors if more than 10 lines
        line1, = betx.plot(tw_list[i].s, tw_list[i].betx, color=color)
        line2, = bety.plot(tw_list[i].s, tw_list[i].bety, color=color)
        line3, = disp.plot(tw_list[i].s, tw_list[i].dx, color=color)
        lines.append(line1)  # Only need to add one line per t_scan for the legend
        labels.append(f't={np.round(t, 2)}')

    betx.set_ylabel(r'$\beta_{x}$ [m]')
    betx.set_ylim(0,400)
    bety.set_ylabel(r'$\beta_{y}$ [m]')
    bety.set_ylim(0,200)
    disp.set_ylabel(r'$D_{x}$ [m]')
    disp.set_ylim(-0.5,1.2)
    disp.set_xlabel(r"s [m]")

    fig.legend(lines, labels, loc='center right', bbox_to_anchor=(0.91, 0.5))
    plt.subplots_adjust(right=0.8)

    plt.show()

    return fig


# def plot_dx_RF(line, t_scan, tw, title, name_el_start):
#     """Plot the dispersion Dx from a given object in the line. 

#     Args:
#         line (_type_): Line. 
#         t_scan (np.array): Time.
#         tw (list): List of twiss object.
#         title (str) : Title of the plot.
#         name_el_start (str) : Name of the object to start at 
#     """
#     plt.figure()
#     plt.title(title)
#     index = line.element_names.index(name_el_start)
#     for i, t in enumerate(t_scan):
#         plt.plot(tw[i].s[index:], tw[i].dx[index:], label=f't={np.round(t,2)}')
#     plt.ylabel(r'$D_{x}$ [m]')
#     plt.xlabel(r"s [m]")
#     plt.legend(prop={'size': 8})
#     plt.subplots_adjust(left=.15, right=.92, hspace=.27)
#     plt.show()

def plot_dx_RF(line, t_scan, tw, title, name_el_start):
    """Plot the dispersion Dx from a given object in the line. 

    Args:
        line (_type_): Line. 
        t_scan (np.array): Time.
        tw (list): List of twiss object.
        title (str) : Title of the plot.
        name_el_start (str) : Name of the object to start at 
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)  # Create an Axes object

    ax.set_title(title)
    index = line.element_names.index(name_el_start)
    for i, t in enumerate(t_scan):
        ax.plot(tw[i].s[index:], tw[i].dx[index:], label=f't={np.round(t, 2)}')

    ax.set_ylabel(r'$D_{x}$ [m]')
    ax.set_xlabel(r"s [m]")
    ax.set_ylim(-0.5,0.5)
    ax.legend(loc='lower left', prop={'size': 11}) 
    plt.subplots_adjust(left=0.15, right=0.92, hspace=0.27)

    plt.show()

    return fig

def fodo_hybrid_qp_symm1(RCS, quad_dico, n_turn_rcs, time_frac, option, half_bend = False, add_quad=0, t_ref = 0.5):

    #Get data from class_geo
    nb_arc = RCS.nb_arc
    nb_cell_arc = RCS.nb_cell_arc
      
    nb_cell_arc = int(RCS.nb_cell_arc)
    pattern = RCS.pattern
    Ldd = RCS.dipole_spacing 
    L_ins = RCS.insertion_length #+RCS.L_extra_arc
    list_hn = RCS.dip_bending(time_frac) #List of bendings
    list_hn_ref = RCS.dip_bending(t_ref) #List of reference bendings
    if half_bend:
        list_hn = [h*RCS.nb_cell_arc/(RCS.nb_cell_arc-2) for h in list_hn]
        list_hn_ref = [h*RCS.nb_cell_arc/(RCS.nb_cell_arc-2) for h in list_hn_ref]
    epsilon = RCS.epsilon(time_frac)
    eps_ref = RCS.epsilon(t_ref)
    L_dip_path = RCS.L_dip_path(time_frac)
    L_dip_path_ref = RCS.L_dip_path(t_ref)
    L_dd_path = RCS.L_dd_path(time_frac)
    L_dd_path_ref = RCS.L_dd_path(t_ref)
    L_qd_path = RCS.L_qp_dip_path(time_frac)
    Ls = RCS.length_short_insertion #Length of short straight section
    
    l_sc, l_nc = quad_dico['qp_sc']['length'], quad_dico['qp_nc']['length']

    g_sc = quad_dico['qp_sc']['gradient']
    k_sc = g_sc/RCS.func_Brho(time_frac)

    g_nc_ext = quad_dico['qp_nc']['gradient']

    #If gradient_inj not precised, gradient at injection is negative value of g_nc_ext
    if 'gradient_inj' in quad_dico['qp_nc'].keys():
        g_nc_time = quad_dico['qp_nc']['gradient_inj'] + (
            g_nc_ext - quad_dico['qp_nc']['gradient_inj'])*time_frac
    else:
        g_nc_time = g_nc_ext*(2*time_frac-1)

    # print('gradient at t', g_nc_time)
    k_nc = g_nc_time/RCS.func_Brho(time_frac)

    #RF data
    n_turns = n_turn_rcs*nb_arc
    energy_increment_per_arc = (RCS.extraction_energy-RCS.injection_energy)/n_turns
    sync_phase = 45
    phase = 180 - sync_phase
    volt_cell = energy_increment_per_arc/np.sin(phase*np.pi/180)

    # Define elements: drifts, quads, sextupoles
    L_ins = L_ins - 2*Ls - 2*RCS.dipole_spacing  # 2 interconnections added to symmetrize the arc
    #!! We put quad for the rf insertion on the place for the rf, to me modified later by according margin from filling factor
    drift_ins = xt.Drift(length = L_ins/2)

    def quad_f():
        hyb_qp_foc = [ 
                xt.Quadrupole(k1 = k_sc, length = l_sc),
                xt.Drift(length = Ldd),
                xt.Quadrupole(k1 = k_nc, length = l_nc)
                ]
        return hyb_qp_foc
    
    def quad_f_mid():
        hyb_qp_foc = [ 
                xt.Quadrupole(k1 = k_sc, length = l_sc/2),
                xt.Drift(length = Ldd),
                # xt.Quadrupole(k1 = k_nc, length = l_nc)
                xt.Quadrupole(k1 = k_nc, length = l_nc/2)
                ]
        return hyb_qp_foc
    
    def half_quad_f_beg():
        hyb_qp_foc = [ 
                xt.Quadrupole(k1 = k_nc, length = l_nc/2),
                xt.Drift(length = Ldd),
                xt.Quadrupole(k1 = k_sc, length = l_sc/2),
                ]
        return hyb_qp_foc
    
    def quad_d():
        hyb_qp_foc = [ 
                xt.Quadrupole(k1 = -k_sc, length = l_sc),
                xt.Drift(length = Ldd),
                xt.Quadrupole(k1 = -k_nc, length = l_nc),
                ]
        return hyb_qp_foc
    
    def quad_d_mid():
        hyb_qp_foc = [ 
                xt.Quadrupole(k1 = -k_sc, length = l_sc/2),
                xt.Drift(length = Ldd),
                # xt.Quadrupole(k1 = -k_nc, length = l_nc),
                xt.Quadrupole(k1 = -k_nc, length = l_nc/2),
                ]
        return hyb_qp_foc
    
    def half_quad_d_beg():
        hyb_qp_foc = [ 
                xt.Quadrupole(k1 = -k_nc, length = l_nc/2),
                xt.Drift(length = Ldd),
                xt.Quadrupole(k1 = -k_sc, length = l_sc/2)
                ]
        return hyb_qp_foc
    
    def hcell():
        hcell_el=[]
        hcell_name=[]
        for id, item in enumerate(pattern):
            hcell_el.append(xt.Drift(length = L_dd_path[id]))
            hcell_name.append(f'drift_dd{id+1}')
            if option == 'var_ref':
                hcell_el.append(xt.Bend(
                    k0 = list_hn[id], 
                    # h = list_hn[id], 
                    angle = list_hn[id] * L_dip_path[id],
                    length = L_dip_path[id], 
                    edge_entry_angle = epsilon[id],
                    edge_exit_angle = -epsilon[id+1]
                    ))
            elif option == 'var_k':
                hcell_el.append(xt.Bend(
                    k0 = list_hn[id], 
                    # h = list_hn_ref[id], 
                    angle = list_hn_ref[id] * L_dip_path_ref[id],
                    length = L_dip_path_ref[id],
                    edge_entry_angle = eps_ref[id], 
                    edge_exit_angle = -eps_ref[id+1]
                    ))
            else:
                raise ValueError("Invalid option: {}".format(option))
            hcell_name.append(item + f'{id}')

        hcell_el.append(xt.Drift(length = L_qd_path))
        hcell_name.append(f'drift_dd{id+2}')

        return hcell_el, hcell_name
    
    def drift_qp():
        hyb_qp_foc = [ 
                xt.Drift(length = l_nc/2*0),
                ]
        return hyb_qp_foc

    #Cavity settings
    cavity = xt.Cavity(voltage = volt_cell/2, frequency = 1.3e9, lag = phase)
    RF_acc = xt.ReferenceEnergyIncrease(Delta_p0c = energy_increment_per_arc/2*0)
    
    FODO_el = [xt.Quadrupole(k1 = k_nc, length = l_nc/2), xt.Drift(length = Ldd), xt.Quadrupole(k1 = k_sc, length = l_sc/2), 
        xt.Quadrupole(k1 = k_sc, length = l_sc/2), xt.Drift(length = Ldd), xt.Quadrupole(k1 = k_nc, length = l_nc/2)] + hcell()[0] + [
        xt.Quadrupole(k1 = -k_nc, length = l_nc/2), xt.Drift(length = Ldd), xt.Quadrupole(k1 = -k_sc, length = l_sc/2), xt.Quadrupole(k1 = -k_sc, length = l_sc/2), 
        xt.Drift(length = Ldd), xt.Quadrupole(k1 = -k_nc, length = l_nc/2)] + hcell()[0]
    FODO_names = ['qp_nc_foc1', 'drift_qp1', 'qp_sc_foc1', 'qp_sc_foc2', 'drift_qp1', 'qp_nc_foc2'] + [name + '_1' for name in hcell()[1]] + [
        'qp_nc_defoc1', 'drift_qp1', 'qp_sc_defoc1', 'qp_sc_defoc2', 'drift_qp1', 'qp_nc_defoc2'] + [name + "_2" for name in hcell()[1]]
    
    list_el = []
    list_name = []

    nb_cell_arc = int(RCS.nb_cell_arc)

    if RCS.nb_cell_arc%2 == 0:
        list_el += quad_f_mid() + drift_qp() + hcell()[0] + quad_d() + hcell()[0]
        list_name += [f'qp_sc1', 'drift_qp1', f'qp_nc1'] + ['drift_qp'] + [name + "_1" for name in hcell()[1]] + [
            f'qp_sc2', 'drift_qp2', f'qp_nc2'] + [name + "_2" for name in hcell()[1]]

        for i in range( int(nb_cell_arc/2 - 1) ):
            list_el = list_el + quad_f() + hcell()[0] + quad_d() + hcell()[0]
            list_name = list_name + [f'qp_sc{2*(i+1)+1}', f'drift_qp{2*(i+1)+1}', f'qp_nc{2*(i+1)+1}'] + [
                name + f"_{3+i*2}" for name in hcell()[1]] + [
                    f'qp_sc{2*(i+1)+2}', f'drift_qp{2*(i+1)+1}',f'qp_nc{2*(i+1)+2}'] + [
                name + f"_{4+i*2}" for name in hcell()[1]]
            
            i_last = 2*(i+1)+3

    else:
        list_el += quad_d_mid() + drift_qp() + hcell()[0]
        list_name += [f'qp_sc1', 'drift_qp1', f'qp_nc1'] + ['drift_qp'] + [name + "_1" for name in hcell()[1]]
        for i in range( int((nb_cell_arc-1)/2) ):
            list_el = list_el + quad_f() + hcell()[0] + quad_d() + hcell()[0]
            list_name = list_name + [f'qp_sc{2*(i+1)}', f'drift_qp{2*(i+1)}', f'qp_nc{2*(i+1)}'] + [
                name + f"_{2+i*2}" for name in hcell()[1]] + [
                    f'qp_sc{2*(i+1)+1}', f'drift_qp{2*(i+1)+1}',f'qp_nc{2*(i+1)+1}'] + [
                        name + f"_{3+i*2}" for name in hcell()[1]]

            i_last = 2*(i+1)+2

    if add_quad == 1:
        L_ins = L_ins - 2*Ls
        drift_ins = xt.Drift(length = L_ins/4)

        list_el += quad_f() + [drift_ins] + quad_d() + [drift_ins, cavity, RF_acc] + half_quad_f_beg()
        list_name += [f'qp_sc{i_last}', f'drift_qp{i_last}', f'qp_nc{i_last}', 'drift_RF_ins_1', 
                    f'qp_sc{i_last+1}', f'drift_qp{i_last+1}', f'qp_nc{i_last+1}', 'drift_RF_ins_2', 
                    'cavity', 'RF_acc'] + [f'qp_nc{i_last+2}', f'drift_qp{i_last+2}', f'qp_sc{i_last+2}']
    
    elif add_quad == 3:
        L_ins = L_ins - 6*Ls
        drift_ins = xt.Drift(length = L_ins/8)

        list_el += quad_f() + [drift_ins] + quad_d() + [drift_ins] + quad_f() + [drift_ins] + quad_d() + [
                    drift_ins, cavity, RF_acc] + half_quad_f_beg()
        list_name += [f'qp_sc{i_last}', f'drift_qp{i_last}', f'qp_nc{i_last}', 'drift_RF_ins_1', 
                      f'qp_sc{i_last+1}', f'drift_qp{i_last+1}', f'qp_nc{i_last+1}', 'drift_RF_ins_2',
                      f'qp_sc{i_last+2}', f'drift_qp{i_last+2}', f'qp_nc{i_last+2}', 'drift_RF_ins_3',
                      f'qp_sc{i_last+3}', f'drift_qp{i_last+3}', f'qp_nc{i_last+3}', 'drift_RF_ins_4',
                      'cavity', 'RF_acc'] + [f'qp_nc{i_last+4}', f'drift_qp{i_last+4}', f'qp_sc{i_last+4}']

    elif add_quad == 5:
        L_ins = L_ins - 10*Ls
        drift_ins = xt.Drift(length = L_ins/12)

        list_el += quad_f() + [drift_ins] + quad_d() + [drift_ins] + quad_f() + [drift_ins] + quad_d() + [drift_ins
                        ] + quad_f() + [drift_ins] + quad_d() + [drift_ins, cavity, RF_acc] + half_quad_f_beg()
        list_name += [f'qp_sc{i_last}', f'drift_qp{i_last}', f'qp_nc{i_last}', 'drift_RF_ins_1', 
                      f'qp_sc{i_last+1}', f'drift_qp{i_last+1}', f'qp_nc{i_last+1}', 'drift_RF_ins_2',
                      f'qp_sc{i_last+2}', f'drift_qp{i_last+2}', f'qp_nc{i_last+2}', 'drift_RF_ins_3',
                      f'qp_sc{i_last+3}', f'drift_qp{i_last+3}', f'qp_nc{i_last+3}', 'drift_RF_ins_4',
                      f'qp_sc{i_last+4}', f'drift_qp{i_last+4}', f'qp_nc{i_last+4}', 'drift_RF_ins_5',
                      f'qp_sc{i_last+5}', f'drift_qp{i_last+5}', f'qp_nc{i_last+5}', 'drift_RF_ins_6',
                      'cavity', 'RF_acc'] + [f'qp_nc{i_last+6}', f'drift_qp{i_last+6}', f'qp_sc{i_last+6}']
    
    elif add_quad == 7:
        L_ins = L_ins - 14*Ls
        drift_ins = xt.Drift(length = L_ins/14)

        list_el += quad_f() + [drift_ins] + quad_d() + [drift_ins] + quad_f() + [drift_ins] + quad_d() + [drift_ins
                        ] + quad_f() + [drift_ins] + quad_d() + [drift_ins] + quad_f() + [drift_ins] + quad_d() + [
                            drift_ins, cavity, RF_acc] + half_quad_f_beg()
        list_name += [f'qp_sc{i_last}', f'drift_qp{i_last}', f'qp_nc{i_last}', 'drift_RF_ins_1', 
                      f'qp_sc{i_last+1}', f'drift_qp{i_last+1}', f'qp_nc{i_last+1}', 'drift_RF_ins_2',
                      f'qp_sc{i_last+2}', f'drift_qp{i_last+2}', f'qp_nc{i_last+2}', 'drift_RF_ins_3',
                      f'qp_sc{i_last+3}', f'drift_qp{i_last+3}', f'qp_nc{i_last+3}', 'drift_RF_ins_4',
                      f'qp_sc{i_last+4}', f'drift_qp{i_last+4}', f'qp_nc{i_last+4}', 'drift_RF_ins_5',
                      f'qp_sc{i_last+5}', f'drift_qp{i_last+5}', f'qp_nc{i_last+5}', 'drift_RF_ins_6',
                      f'qp_sc{i_last+6}', f'drift_qp{i_last+6}', f'qp_nc{i_last+6}', 'drift_RF_ins_7',
                      f'qp_sc{i_last+7}', f'drift_qp{i_last+7}', f'qp_nc{i_last+7}', 'drift_RF_ins_8',
                      'cavity', 'RF_acc'] + [f'qp_nc{i_last+8}', f'drift_qp{i_last+8}', f'qp_sc{i_last+8}']
        
    elif add_quad == 9:
        L_ins = L_ins - 18*Ls
        drift_ins = xt.Drift(length = L_ins/18)

        list_el += quad_f() + [drift_ins] + quad_d() + [drift_ins
                ] + quad_f() + [drift_ins] + quad_d() + [drift_ins
                ] + quad_f() + [drift_ins] + quad_d() + [drift_ins
                ] + quad_f() + [drift_ins] + quad_d() + [drift_ins
                ] + quad_f() + [drift_ins] + quad_d() + [drift_ins
                ] + quad_f() + [drift_ins] + quad_d() + [drift_ins, cavity, RF_acc] + half_quad_f_beg()
        list_name += [f'qp_sc{i_last}', f'drift_qp{i_last}', f'qp_nc{i_last}', 'drift_RF_ins_1', 
                      f'qp_sc{i_last+1}', f'drift_qp{i_last+1}', f'qp_nc{i_last+1}', 'drift_RF_ins_2',
                      f'qp_sc{i_last+2}', f'drift_qp{i_last+2}', f'qp_nc{i_last+2}', 'drift_RF_ins_3',
                      f'qp_sc{i_last+3}', f'drift_qp{i_last+3}', f'qp_nc{i_last+3}', 'drift_RF_ins_4',
                      f'qp_sc{i_last+4}', f'drift_qp{i_last+4}', f'qp_nc{i_last+4}', 'drift_RF_ins_5',
                      f'qp_sc{i_last+5}', f'drift_qp{i_last+5}', f'qp_nc{i_last+5}', 'drift_RF_ins_6',
                      f'qp_sc{i_last+6}', f'drift_qp{i_last+6}', f'qp_nc{i_last+6}', 'drift_RF_ins_7',
                      f'qp_sc{i_last+7}', f'drift_qp{i_last+7}', f'qp_nc{i_last+7}', 'drift_RF_ins_8',
                      f'qp_sc{i_last+8}', f'drift_qp{i_last+8}', f'qp_nc{i_last+8}', 'drift_RF_ins_9',
                      f'qp_sc{i_last+9}', f'drift_qp{i_last+9}', f'qp_nc{i_last+9}', 'drift_RF_ins_10',
                      f'qp_sc{i_last+10}', f'drift_qp{i_last+10}', f'qp_nc{i_last+10}', 'drift_RF_ins_11',
                      f'qp_sc{i_last+11}', f'drift_qp{i_last+11}', f'qp_nc{i_last+11}', 'drift_RF_ins_12',
                      'cavity', 'RF_acc'] + [f'qp_nc{i_last+12}', f'drift_qp{i_last+12}', f'qp_sc{i_last+12}']

    elif add_quad == 0:
        list_el += quad_f() + [drift_ins, cavity, RF_acc] + half_quad_d_beg()
        list_name += [f'qp_sc{i_last}', f'drift_qp{i_last}', f'qp_nc{i_last}'] + ['drift_RF_ins', 'cavity', 'RF_acc'] + [
            f'qp_nc{i_last+1}', f'drift_qp{i_last+1}', f'qp_sc{i_last+1}']
    
    else:
        raise ValueError(r'add_quad {add_quad} not coded.')

    line_FODO = xt.Line(elements=FODO_el, element_names=FODO_names)
    line_FODO.particle_ref = xp.Particles(p0c=RCS.injection_energy+(RCS.extraction_energy-RCS.injection_energy)*time_frac,
                                        q0=1, mass0=xp.MUON_MASS_EV)
    
    line_tr = xt.Line(element_names=list_name, elements=list_el)
    line_tr.particle_ref =  xp.Particles(p0c=RCS.injection_energy+(RCS.extraction_energy-RCS.injection_energy)*time_frac,
                                        q0=1, mass0=xp.MUON_MASS_EV)
    
    return (line_FODO, line_tr)

def create_folders(folder, rcs_name, nb_arc):
    """Create subfolders if they do not exist.

    Args:
        save_path (str): Path to the folder where subfolders will be created.
    """
    main_folder_name = f"{rcs_name}_arc_{nb_arc}"
    main_folder_path = os.path.join(folder, main_folder_name)

    if not os.path.exists(main_folder_path):
        os.makedirs(main_folder_path)
        print(f"Created main folder: {main_folder_path}")
    else:
        print(f"Main folder already exists: {main_folder_path}")
    # Define the subfolder names
    subfolders = ['half_bend', 'qp_only', 'scaling_tg0', 'half_bend_qp']

    # Create each subfolder if it does not exist
    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder_path, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            print(f"Created subfolder: {subfolder_path}")
        else:
            print(f"Subfolder already exists: {subfolder_path}")
    return main_folder_path

if __name__ == '__main__':

    path = Path(__file__).resolve().parent

    MU = np.pi/2
    n_size = 3
    rcs_name = 'RCS3_CERN'
    option = 'var_ref'

    input = load_file(path / 'FODO_cells_energy_spread.json')

    sigma_delta = input[rcs_name]['sig_delta_H_'+f'{n_size}']
    file_input = input[rcs_name]['file']
    n_turn_rcs = input[rcs_name]['n_turn_rcs']
    add_quad = input[rcs_name]['add_quad']
    alpha = input[rcs_name]['alpha']
    # alpha=7
    # add_quad=7
    
    nb_arc = 8
    gsc_max = 400

    input_data = {'rcs_name' : rcs_name,
                'file_input' : file_input,
                'option' : option,
                'mu' : MU,
                'n_size' : 3,
                'sigma_delta' : sigma_delta,
                'n_turns_rcs' : n_turn_rcs,
                'add_quad' : add_quad,
                'alpha' : alpha,
                'nb_arc' : nb_arc,
                'gsc_max' : gsc_max
                }

    gradient = GradientHFODO(
        file_input=file_input, 
        mu=MU, 
        sigma_delta=sigma_delta,
        symmetric = False, 
        gsc_max=gsc_max, 
        option = 'var_ref', 
        alpha=alpha,
        plot=True, 
        opt_geo_for_na=nb_arc
        )
    
    qp_para = gradient.qp_para
    rcs = gradient.rcs

    save_fig = False
    dump_line = False

    folder = 'results_DS_hybrid'
    if save_fig:
            main_folder_path = create_folders(folder, rcs_name, nb_arc)
    else:
        main_folder_path=None
        
    DS = DS_Hybrid(gradient, n_turn_rcs, add_quad, save_fig_path=main_folder_path)
    print('Match DS at tg0 (G_NC=0) and scale with energy.')
    DS.scaling_from_tg0()
    
    # print('Apply half-bend to last FODO cells.')
    # DS.DS_half_bend()

    # print('Apply half-bend to last FODO cells and match quad with linear reg.')
    # DS.DS_half_bend_quad()

    print('Match with QP at all energies, apply linear reg.')
    DS.adapt_qp()

    if save_fig:
        save_para(rcs, main_folder_path+'/rcs_parameter')
        with open(main_folder_path + '/input_parameter.json', 'w') as f:
            json.dump(input_data, f, indent=4)
        with open(main_folder_path + '/qp_para.json', 'w') as f:
            json.dump(qp_para, f, indent=4)

    if dump_line:
        for time, line in zip(DS.tscan_only_qp, DS.line_arc_only_qp):
            line.to_json(main_folder_path+f'/line_only_qp_time_{time:.1f}.json')

    plt.figure()
    plt.plot(DS.tscan_only_qp, 1e4 * np.array([tw.momentum_compaction_factor for tw in DS.tw_only_quad_linreg]))
    plt.xlabel('Time normalized')
    plt.ylabel('MCF [1e4]')
    plt.show()

    plt.figure()
    plt.plot(DS.tscan_only_qp, [tw.qx for tw in DS.tw_only_quad_linreg], label=r'$Q_x$')
    plt.plot(DS.tscan_only_qp, [tw.qy for tw in DS.tw_only_quad_linreg], label=r'$Q_y$')
    plt.xlabel('Time normalized')
    plt.ylabel('Tune')
    plt.legend()
    plt.show()
    