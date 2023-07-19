#%%
from math import log10
import random
import myokit
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from scipy.signal import find_peaks # pip install scipy
import re
from collections import Counter
from itertools import groupby
from operator import itemgetter

def get_ind_data(ind, path = '../', model = 'tor_ord_endo2.mmt'):
    mod, proto, x = myokit.load(path+model)
    if ind is not None:
        for k, v in ind[0].items():
            mod['multipliers'][k].set_rhs(v)

    return mod, proto

def plot_conductances(profile, save_to = 0, title = 0, fig_label = 0, ax = 'yes', a = 1, coloring = 'black', tick_labels = ['GCaL', 'GKs', 'GKr', 'GNaL', 'GNa', 'Gto', 'GK1', 'GNCX', 'GNaK', 'GKb']):
    if ax == 'yes':
        fig, ax = plt.subplots(1, figsize =(10, 7), constrained_layout = True)
    
    if fig_label !=0:
        ax.scatter(list(range(0, len(profile))), profile, color = coloring, s=200, alpha = a,label = fig_label)
        ax.legend()
    else:
        ax.scatter(list(range(0, len(profile))), profile, color = coloring, alpha = a, s=200)

    ax.set_ylabel("Conductance Value", fontsize = 35, fontname="Arial")
    ax.set_xticks(list(range(0, len(profile))))
    ax.set_xticklabels(tick_labels, fontsize = 35, rotation = 45, fontname="Arial")
    ax.tick_params(axis='y', labelsize = 30)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if title !=0:
        ax.set_title(title, fontsize = 30, fontname = 'Arial')


    if save_to != 0:
        plt.savefig(save_to+'conductances.png', transparent = True, bbox_inches='tight')

def remove_noise(t, v_10, v_90):
    data = pd.DataFrame(data = {'t': t[1000:len(t)], 'v_10': v_10[1000:len(t)], 'v_90':v_90[1000:len(t)]})
    data_start = pd.DataFrame(data = {'t': t[150:1000], 'v_10': v_10[150:1000], 'v_90':v_90[150:1000]})
    
    # FILTER V_10
    v_10_new = data.v_10.rolling(400, min_periods = 1, center = True).mean()
    v_10_start = data_start.v_10.rolling(100, min_periods = 1, center = True).mean()
    v_10_new = v_10_new.dropna()
    v_10_new = list(v_10_start) + list(v_10_new)
    t_new = list(data_start['t']) + list(data['t'])

    # FILTER V_90
    v_90_new = data.v_90.rolling(400, min_periods = 1, center = True).mean()
    v_90_start = data_start.v_90.rolling(200, min_periods = 1, center = True).mean()
    v_90_new = v_90_new.dropna()
    v_90_new = list(v_90_start) + list(v_90_new)

    return(t_new, v_10_new, v_90_new)

def get_torord_phys_data(path = '../', filter = 'no'):
    data = pd.read_csv(path+'data/torord_physiologicData.csv')
    time = [x - 9.1666666669999994 for x in list(data['t'])] #shift action potential to match solutions
    t = time[275:len(data['v_10'])]
    v_10 = list(data['v_10'])[275:len(data['v_10'])]
    v_90 = list(data['v_90'])[275:len(data['v_10'])]

    if filter != 'no':
        data = pd.DataFrame(data = {'t': t[1000:len(t)], 'v_10': v_10[1000:len(t)], 'v_90':v_90[1000:len(t)]})
        data_start = pd.DataFrame(data = {'t': t[150:1000], 'v_10': v_10[150:1000], 'v_90':v_90[150:1000]})
        
        # FILTER V_10
        v_10_new = data.v_10.rolling(400, min_periods = 1, center = True).mean()
        v_10_start = data_start.v_10.rolling(100, min_periods = 1, center = True).mean()
        v_10_new = v_10_new.dropna()
        v_10 = list(v_10_start) + list(v_10_new)
        t = list(data_start['t']) + list(data['t'])

        # FILTER V_90
        v_90_new = data.v_90.rolling(400, min_periods = 1, center = True).mean()
        v_90_start = data_start.v_90.rolling(200, min_periods = 1, center = True).mean()
        v_90_new = v_90_new.dropna()
        v_90 = list(v_90_start) + list(v_90_new)


    return(t, v_10, v_90)

def immunize_ind_data(ind, immunization_profile, application = 'multiply'):
    #immunization_profile = [0.3, 1, 2, 1.7, 0.6, 1.6, 1.9, 0.1, 1.5, 1.9] #profile from e1500
    #immunization_profile = [0.5, 1.1, 1.8, 1.75, 0.9, 1.5, 1.8, 0.25, 1.5, 1.75] #mean profile from e2000
    #immunization_profile = [0.49408854383200007, 1.1378567345743085, 1.918958995182459, 1.7838346831830942, 0.9151182733883709, 1.5631241185156528, 1.8399928540280137, 0.2659572406574743, 1.5424713631643459, 1.7280266469590058] #mean exact from e2000
    #immunization_profile = [0.2448681240674403, 1.4422150038253354, 1.9846303314143747, 1.9805655201192005, 0.6759346628744565, 1.97353080973506, 1.953985298216601, 0.1054002499457096, 1.1639621613526687, 1.969101586994144] #mode from e2000
    vals = list(ind.values())
    labs = list(ind.keys())
    if application == 'multiply':
        new_conds = [immunization_profile[i]*vals[i] for i in range(len(immunization_profile))]
    if application == 'add':
        new_conds = [immunization_profile[i]+vals[i] for i in range(len(immunization_profile))]
    immune_ind = dict(zip(labs, new_conds))

    return immune_ind

def load_data(i, path):

    data = pd.read_csv(path)
    #ind = eval(data['ind'][i])
    ind = data.iloc[i].filter(like = 'multiplier').to_dict()
    
    #ind = get_ind(vals=data.filter(like = 'multiplier').to_numpy().tolist()[i])
    return(ind)

def get_ind(vals = [1,1,1,1,1,1,1,1,1,1], celltype = 'adult'):
    if celltype == 'ipsc':
        tunable_parameters=['i_cal_pca_multiplier','i_ks_multiplier','i_kr_multiplier','i_na_multiplier','i_to_multiplier','i_k1_multiplier','i_f_multiplier']
        ind = dict(zip(tunable_parameters, vals))
    else:
        tunable_parameters=['i_cal_pca_multiplier','i_ks_multiplier','i_kr_multiplier','i_nal_multiplier','i_na_multiplier','i_to_multiplier','i_k1_multiplier','i_NCX_multiplier','i_nak_multiplier','i_kb_multiplier']
        ind = dict(zip(tunable_parameters, vals))
    return(ind)

def initialize_individuals():
    """
    Creates the initial population of individuals. The initial 
    population 
    Returns:
        An Individual with conductance parameters 
    """
    tunable_parameters=['i_cal_pca_multiplier','i_ks_multiplier','i_kr_multiplier','i_nal_multiplier','i_na_multiplier','i_to_multiplier','i_k1_multiplier','i_NCX_multiplier','i_nak_multiplier','i_kb_multiplier']
    
    # Builds a list of parameters using random upper and lower bounds.
    lower_exp = log10(0.1)
    upper_exp = log10(2)

    initial_params = []
    for i in range(0, len(tunable_parameters)):
            initial_params.append(10**random.uniform(lower_exp, upper_exp))

    keys = [val for val in tunable_parameters]
    return dict(zip(keys, initial_params))

def get_last_ap_old(dat, AP, type = 'full'):

    if type == 'full':
        # Get t, v, and cai for second to last AP#######################
        i_stim = dat['stimulus.i_stim']

        # This is here so that stim for EAD doesnt interfere with getting the whole AP
        for i in list(range(0,len(i_stim))):
            if abs(i_stim[i])<50:
                i_stim[i]=0

        
        peaks = find_peaks(-np.array(i_stim), distance=100)[0]
        start_ap = peaks[AP] #TODO change start_ap to be after stim, not during
        end_ap = peaks[AP+1]

        t = np.array(dat['engine.time'][start_ap:end_ap])
        t = t - t[0]
        max_idx = np.argmin(np.abs(t-995))
        t = t[0:max_idx]
        end_ap = start_ap + max_idx

        v = np.array(dat['membrane.v'][start_ap:end_ap])
        cai = np.array(dat['intracellular_ions.cai'][start_ap:end_ap])
        i_ion = np.array(dat['membrane.i_ion'][start_ap:end_ap])

        #convert to list for accurate storage in dataframe
        #t = list(t)
        #v = list(v)

        data = {}
        data['t'] = t
        data['v'] = v
        data['cai'] = cai
        data['i_ion'] = i_ion
    
    else:
        # Get t, v, and cai for second to last AP#######################
        i_stim, ti, vol = dat

        # This is here so that stim for EAD doesnt interfere with getting the whole AP
        for i in list(range(0,len(i_stim))):
            if abs(i_stim[i])<50:
                i_stim[i]=0

        peaks = find_peaks(-np.array(i_stim), distance=100)[0]
        start_ap = peaks[AP] #TODO change start_ap to be after stim, not during
        end_ap = peaks[AP+1]

        t = np.array(ti[start_ap:end_ap])
        t = t - t[0]
        max_idx = np.argmin(np.abs(t-995))
        t = t[0:max_idx]
        end_ap = start_ap + max_idx

        v = np.array(vol[start_ap:end_ap])

        #convert to list for accurate storage in dataframe
        #t = list(t)
        #v = list(v)

        data = {}
        data['t'] = t
        data['v'] = v

    return (data)

def get_last_ap(dat, AP, cl = 1000, type = 'full'):

    if type == 'full':
        start_ap = list(dat['engine.time']).index(closest(list(dat['engine.time']), AP*cl))
        end_ap = list(dat['engine.time']).index(closest(list(dat['engine.time']), (AP+1)*cl))

        t = np.array(dat['engine.time'][start_ap:end_ap])
        t = t-t[0]
        v = np.array(dat['membrane.v'][start_ap:end_ap])
        cai = np.array(dat['intracellular_ions.cai'][start_ap:end_ap])
        i_ion = np.array(dat['membrane.i_ion'][start_ap:end_ap])

        data = {}
        data['t'] = t
        data['v'] = v
        data['cai'] = cai
        data['i_ion'] = i_ion
    
    else:
        # Get t, v, and cai for second to last AP#######################
        ti, vol = dat

        start_ap = list(ti).index(closest(ti, AP*cl))
        end_ap = list(ti).index(closest(ti, (AP+1)*cl))

        t = np.array(ti[start_ap:end_ap])
        v = np.array(vol[start_ap:end_ap])

        data = {}
        data['t'] = t
        data['v'] = v

    return (data)

def run_model_old(ind, beats, cl = 1000, prepace = 100, I0 = 0, sim =0, return_sim = 0, path = '../', model = 'tor_ord_endo2.mmt'): 

    if I0 != 0:
        sim.reset()
        sim.set_state(I0)

    else:
        mod, proto = get_ind_data(ind, path, model = model)
        proto.schedule(5.3, 0.1, 1, cl, 0) 
        sim = myokit.Simulation(mod,proto)

    sim.pre(cl * prepace) #pre-pace for 100 beats
    dat = sim.run(beats*cl) 
    IC = sim.state()

    if return_sim != 0:
        return(dat, sim, IC)
    else:
        return(dat, IC) 

def run_model(ind, beats, stim = 5.3, stim_1 = 0, start = 0.1, start_1 = 0, length = 1, length_1 = 0, cl = 1000, prepace = 600, I0 = 0, path = '../', model = 'tor_ord_endo2.mmt', min_dt = None, max_dt = None): 
    mod, proto = get_ind_data(ind, path, model = model)
    proto.schedule(stim, start, length, cl, 0) 
    if stim_1 != 0:
        proto.schedule(stim_1, start_1, length_1, cl, 1)
    sim = myokit.Simulation(mod,proto)

    if I0 != 0:
        sim.set_state(I0)


    sim.set_min_step_size(dtmin=min_dt)
    sim.set_max_step_size(dtmax=max_dt)
    sim.pre(cl * prepace) #pre-pace for 100 beats
    dat = sim.run(beats*cl) 
    IC = sim.state()

    return(dat, IC) 

def percent_change(percent, initial):
    final = ((percent*np.abs(initial))/100) + initial
    return(final)

def detect_EAD_old(t, v):
    #find slope
    slopes = []
    for i in list(range(0, len(v)-1)):
        m = (v[i+1]-v[i])/(t[i+1]-t[i])
        slopes.append(round(m, 2))

    #find rises
    pos_slopes = np.where(slopes > np.float64(0.0))[0].tolist()
    pos_slopes_idx = np.where(np.diff(pos_slopes)!=1)[0].tolist()
    pos_slopes_idx.append(len(pos_slopes)) #list must end with last index

    #pull out groups of rises (indexes)
    pos_groups = []
    pos_groups.append(pos_slopes[0:pos_slopes_idx[0]+1])
    for x in list(range(0,len(pos_slopes_idx)-1)):
        g = pos_slopes[pos_slopes_idx[x]+1:pos_slopes_idx[x+1]+1]
        pos_groups.append(g)

    #pull out groups of rises (voltages and times)
    vol_pos = []
    tim_pos = []
    for y in list(range(0,len(pos_groups))):
        vol = []
        tim = []
        for z in pos_groups[y]:
            vol.append(v[z])
            tim.append(t[z])
        vol_pos.append(vol)
        tim_pos.append(tim) 

    #Find EAD given the conditions (voltage>-70 & time>100)
    EADs = []
    EAD_vals = []
    for k in list(range(0, len(vol_pos))):
        if np.mean(vol_pos[k]) > -70 and np.mean(tim_pos[k]) > 100:
            EAD_vals.append(tim_pos[k])
            EAD_vals.append(vol_pos[k])
            EADs.append(max(vol_pos[k])-min(vol_pos[k]))

    #Report EAD 
    if len(EADs)==0:
        info = "no EAD"
        result = 0
    else:
        info = "EAD:", round(max(EADs))
        result = 1
    
    return result

def detect_EAD(t,v):

    #find slope
    slopes = []
    for i in list(range(0, len(v)-1)):
        if t[i] > 100 and v[i] < 20:
            m = (v[i+1]-v[i])/(t[i+1]-t[i])
            slopes.append(round(m, 2))
        else:
            slopes.append(-2.0)

    rises_idx = np.where(np.array(slopes)>0)
    rises_groups = []
    for k, g in groupby(enumerate(rises_idx[0]), lambda i_x: i_x[0] - i_x[1]):
        rises_groups.append(list(map(itemgetter(1), g)))

    EAD_idx = [group_idx for group_idx in rises_groups if len(group_idx)>10 and abs(group_idx[0]-group_idx[-1])>3]

    #Report EAD 
    if len(EAD_idx)==0:
        info = "no EAD"
        result = 0
    else:
        info = "EAD is present"
        result = 1
    
    return(result)

def get_ead_error(ind, code, path = '../'): 

    ## EAD CHALLENGE: Istim = -.1
    if code == 'stim':
        mod, proto = get_ind_data([ind], path)
        proto.schedule(5.3, 0.1, 1, 1000, 0) 
        proto.schedule(0.1, 3004, 1000-100, 1000, 1) #EAD amp is about 4mV from this
        sim = myokit.Simulation(mod,proto)
        dat = sim.run(5000)
        plt.plot(dat['engine.time'], dat['membrane.v'])

        # Get t, v, and cai for second to last AP#######################
        data = get_last_ap(dat, -2)

    ## EAD CHALLENGE: ICaL = 15x (acute increase - no prepacing here)
    if code == 'ical':
        mod, proto = get_ind_data([ind], path)
        mod['multipliers']['i_cal_pca_multiplier'].set_rhs(ind[0]['i_cal_pca_multiplier']*13)
        proto.schedule(5.3, 0.1, 1, 1000, 0) 
        sim = myokit.Simulation(mod,proto)
        dat = sim.run(5000)
        plt.plot(dat['engine.time'], dat['membrane.v'])

        # Get t, v, and cai for second to last AP#######################
        data = get_last_ap(dat, -2)

    ## EAD CHALLENGE: IKr = 90% block (acute increase - no prepacing here)
    if code == 'ikr':
        mod, proto = get_ind_data([ind], path)
        mod['multipliers']['i_kr_multiplier'].set_rhs(ind[0]['i_kr_multiplier']*0.05)
        mod['multipliers']['i_kb_multiplier'].set_rhs(ind[0]['i_kb_multiplier']*0.05)
        proto.schedule(5.3, 0.1, 1, 1000, 0) 
        sim = myokit.Simulation(mod,proto)
        dat = sim.run(5000)
        plt.plot(dat['engine.time'], dat['membrane.v'])

        # Get t, v, and cai for second to last AP#######################
        data = get_last_ap(dat, -2)


    ########### EAD DETECTION ############# 
    t = data['t']
    v = data['v']
    EAD = detect_EAD(t,v)

    #################### ERROR CALCULATION #######################
    error = 0

    cost = 'function_2'
    if cost == 'function_1':
        error += (0 - (1000*EAD))**2
    else:
        error += 1000*EAD #Since the baseline EAD is 4mV this is multipled by 1000 to get on the same scale as RRC error

    return t,v,EAD

def detect_RF(t,v):

    #find slopes
    slopes = []
    for i in list(range(0, len(v)-1)):
        m = (v[i+1]-v[i])/(t[i+1]-t[i])
        slopes.append(round(m, 1))

    #find times and voltages at which slope is 0
    zero_slopes = np.where(slopes == np.float64(0.0))[0].tolist()
    zero_slopes_idx = np.where(np.diff(zero_slopes)!=1)[0].tolist()
    zero_slopes_idx.append(len(zero_slopes)) #list must end with last index

    #pull out groups of zero slope (indexes)
    zero_groups = []
    zero_groups.append(zero_slopes[0:zero_slopes_idx[0]+1])
    for x in list(range(0,len(zero_slopes_idx)-1)):
        g = zero_slopes[zero_slopes_idx[x]+1:zero_slopes_idx[x+1]+1]
        zero_groups.append(g)

    #pull out groups of zero slopes (voltages and times)
    vol_pos = []
    tim_pos = []
    for y in list(range(0,len(zero_groups))):
        vol = []
        tim = []
        for z in zero_groups[y]:
            vol.append(v[z])
            tim.append(t[z])
        vol_pos.append(vol)
        tim_pos.append(tim) 


    #Find RF given the conditions (voltage<-70 & time>100)
    no_RF = []
    for k in list(range(0, len(vol_pos))):
        if np.mean(vol_pos[k]) < -70 and np.mean(tim_pos[k]) > 100:
            no_RF.append(tim_pos[k])
            no_RF.append(vol_pos[k])

    #Report EAD 
    if len(no_RF)==0:
        info = "Repolarization failure!"
        result = 1
    else:
        info = "normal repolarization - resting membrane potential from t=", no_RF[0][0], "to t=", no_RF[0][len(no_RF[0])-1]
        result = 0
    return result

def detect_abnormal_ap(t, v, cl=1000):

    slopes = []
    for i in list(range(0, len(v)-1)):
        if t[i] > 100 and v[i] < 20:
            m = (v[i+1]-v[i])/(t[i+1]-t[i])
            slopes.append(round(m, 2))
        else:
            slopes.append(-2.0)

    # EAD CODE
    rises_idx = np.where(np.array(slopes)>0)
    rises_groups = []
    for k, g in groupby(enumerate(rises_idx[0]), lambda i_x: i_x[0] - i_x[1]):
        rises_groups.append(list(map(itemgetter(1), g)))

    # RF CODE
    rpm_idx = np.where(np.array(slopes) == 0)
    rpm_groups = []
    for k, g in groupby(enumerate(rpm_idx[0]), lambda i_x: i_x[0] - i_x[1]):
        rpm_groups.append(list(map(itemgetter(1), g)))

    flat_groups = [group for group in rpm_groups if v[group[-1]]<-70]

    # CHECK PHASE 4 RF
    if len(flat_groups)>0:
        RMP_start = flat_groups[0][0]
        v_rm = v[RMP_start:len(v)]
        t_rm = t[RMP_start:len(v)]
        slope = (v_rm[-1]-v_rm[0])/(t_rm[-1]-t_rm[0])
        if slope < 0.01:
            for group in rises_groups:
                if v[group[0]]<-70:
                    rises_groups.remove(group)
    
    # LASTLY, CHECKif cell AP is valid
    if ((min(v) > -60) or (max(v) < 0)):
        morph = 1
    else:
        morph = 0


    if len(flat_groups)>0 and len(rises_groups)==0 and morph==0:
        info = "normal AP" 
        result = 0
    else:
        info = "abnormal AP"
        result = 1

    data = {'info': info, 'result':result, 'EADs':rises_groups, 'RMP':flat_groups}

    return(data)

def detect_alternans(dat):
    APDs = []
    for i in list(range(0,4)):
        data = get_last_ap(dat, i, cl = 270, type = 'part')
        t = data['t']
        v = data['v']
        APD90 = calc_APD(t,v,90)
        APDs.append(APD90)

    #Report EAD 
    if np.abs(APDs[0]-APDs[1])>1:
        info = "alternans: " + str(APDs) 
        result = 1
    else:
        info = "no alternans"
        result = 0
    data = {'result': result, 'APDs': APDs}
    return data

def detect_APD(t, v, apd90_base):
    APD90_i = calc_APD(t, v, 90)
    APD90_error = (APD90_i - apd90_base)/(APD90_i)*100
    if APD90_error < 40:
        result_APD = 0
    else:
        result_APD = 1
    return(result_APD)

def calc_APD(t, v, apd_pct):
    t = np.array(t)
    v = np.array(v)
    t = [i-t[0] for i in t]
    mdp = min(v)
    max_p = max(v)
    max_p_idx = np.argmax(v)
    apa = max_p - mdp
    repol_pot = max_p - apa * apd_pct/100
    idx_apd = np.argmin(np.abs(v[max_p_idx:] - repol_pot))
    apd_val = t[idx_apd+max_p_idx]

    result = detect_RF(t, v)
    if result == 1:
        apd_val = max(t)
    
    # Another way to calculate apd
    # vt90=0.9*s2.state()[m2.get('membrane.v').indice()]
    # apd90 = dat.apd(v='membrane.v', threshold = vt90)

    return(apd_val) 

def rrc_search_old(ind, IC, path = '../', model = 'tor_ord_endo2.mmt'):
    all_t = []
    all_v = []
    stims = [0, 0.3]
    APs = list(range(10004, 100004, 5000))

    mod, proto = get_ind_data(ind, path, model) 
    if model == 'kernik.mmt':
        proto.schedule(1, 0.2, 5, 1000, 0)
    else:
        proto.schedule(5.3, 0.2, 1, 1000, 0)
    proto.schedule(0.3, 5004, 995, 1000, 1)
    sim = myokit.Simulation(mod, proto)
    sim.set_state(IC)
    dat = sim.run(7000)

    d0 = get_last_ap(dat, 4)
    t0 = d0['t']
    v0 = d0['v']
    all_t.append(t0)
    all_v.append(v0)
    result_EAD0 = detect_EAD(t0,v0)
    result_RF0 = detect_RF(t0,v0)

    d3 = get_last_ap(dat, 5)
    t3 = d3['t']
    v3 = d3['v']
    all_t.append(t3)
    all_v.append(v3)
    result_EAD3 = detect_EAD(t3,v3)
    result_RF3 = detect_RF(t3,v3)

    if result_EAD0 == 1 or result_RF0 == 1:
        RRC = 0

    elif result_EAD3 == 0 and result_RF3 == 0:
        # no abnormality at 0.3 stim, return RRC
        RRC = 0.3

    else:
        #low = 0.075
        low = 0
        high = 0.3
        EADs = []
        RFs = []
        for i in list(range(0,len(APs))):
            mid = round((low + (high-low)/2), 4) #THIS WAS USED IN GA 8 AND BEFORE
            #mid = round((low + (high-low)/2), 3)
            stims.append(mid)

            sim.reset()
            sim.set_state(IC)
            proto.schedule(mid, APs[i], 995, 1000, 1)
            sim.set_protocol(proto)
            dat = sim.run(APs[i]+2000)

            data = get_last_ap(dat, int((APs[i]-4)/1000))
            t = data['t']
            v = data['v']
            all_t.append(t)
            all_v.append(v)
            result_EAD = detect_EAD(t,v)
            EADs.append(result_EAD)
            result_RF = detect_RF(t,v)
            RFs.append(result_RF)

            if (high-low)<0.0025: #THIS WAS USED IN GA 8 AND BEFORE
            #if (high-low)<0.01:
                break 
            
            elif result_EAD == 0 and result_RF == 0:
                # no RA so go from mid to high
                low = mid

            else:
                #repolarization failure so go from mid to low 
                high = mid
        
        for i in list(range(1, len(EADs))):
            if EADs[-i] == 0 and RFs[-i] == 0:
                RRC = stims[-i] 
                break
            else:
                RRC = 0 #in this case there would be no stim without an RA
        
    result = {'RRC': RRC, "t_rrc": all_t, 'v_rrc': all_v, 'stims': stims}

    return(result)

def rrc_search(ind, IC, stim = 5.3, start = 0.2, length = 1, cl = 1000, path = '../', model = 'tor_ord_endo2.mmt', min_dt = None, max_dt = None):
    all_data = []
    APs = list(range((10*cl)+(int(length)+4), (100*cl)+(int(length)+4), 5*cl)) #length needs to be an integer so it rounds if needed

    mod, proto = get_ind_data(ind, path, model) 
    proto.schedule(stim, start, length, cl, 0)
    proto.schedule(3, (5*cl)+(int(length)+4), cl-(length+4), cl, 1)
    sim = myokit.Simulation(mod, proto)
    sim.set_state(IC)
    sim.set_min_step_size(dtmin=min_dt)
    sim.set_max_step_size(dtmax=max_dt)
    dat = sim.run(7*cl)

    d0 = get_last_ap(dat, 4, cl=cl)
    result_abnormal0 = detect_abnormal_ap(d0['t'], d0['v'], cl=cl) 
    all_data.append({**{'t_rrc': d0['t'], 'v_rrc': d0['v'], 'stim': 0}, **result_abnormal0})

    d3 = get_last_ap(dat, 5, cl=cl)
    result_abnormal3 = detect_abnormal_ap(d3['t'], d3['v'], cl=cl)
    all_data.append({**{'t_rrc': d3['t'], 'v_rrc': d3['v'], 'stim': 3}, **result_abnormal3})

    #if result_EAD0 == 1 or result_RF0 == 1:
    if result_abnormal0['result'] == 1:
        RRC = 0

    #elif result_EAD3 == 0 and result_RF3 == 0:
    elif result_abnormal3['result'] == 0:
        # no abnormality at 0.3 stim, return RRC
        RRC = 3

    else:
        low = 0
        high = 3
        for i in list(range(0,len(APs))):
            mid = round((low + (high-low)/2), 4) 

            sim.reset()
            sim.set_state(IC)
            sim.set_min_step_size(dtmin=min_dt)
            sim.set_max_step_size(dtmax=max_dt)
            proto.schedule(mid, APs[i], cl-(length+4), cl, 1)
            sim.set_protocol(proto)
            dat = sim.run(APs[i]+(2*cl))

            data = get_last_ap(dat, int((APs[i]-(int(length)+4))/cl), cl=cl)
            result_abnormal = detect_abnormal_ap(data['t'], data['v'])
            all_data.append({**{'t_rrc': data['t'], 'v_rrc': data['v'], 'stim': mid}, **result_abnormal})
            
            if result_abnormal['result'] == 0:
                # no RA so go from mid to high
                low = mid

            else:
                #repolarization failure so go from mid to low 
                high = mid

            #if (high-low)<0.01:
            if (high-low)<0.0025: #THIS WAS USED IN GA 8 AND BEFORE
                break 
        
        for i in list(range(1, len(all_data))):
            if all_data[-i]['result'] == 0:
                RRC = all_data[-i]['stim']
                break
            else:
                RRC = 0 #in this case there would be no stim without an RA

    result = {'RRC':RRC, 'data':all_data}

    return(result)

def assess_challenges(ind, path = '../', model = 'tor_ord_endo2.mmt'):

    ## EAD CHALLENGE: Istim = -.1
    mod, proto = get_ind_data(ind, path, model)
    proto.schedule(5.3, 0.1, 1, 1000, 0) 
    sim = myokit.Simulation(mod,proto)
    sim.pre(600*1000)
    proto.schedule(0.1, 4004, 1000-100, 1000, 1) #EAD amp is about 4mV from this
    sim.set_protocol(proto)
    dat = sim.run(6000)
    IC = sim.state()

    ## EAD CHALLENGE: ICaL = 8x (acute increase - 100 beats prepacing)
    sim.reset()
    sim.set_state(IC)
    sim.set_constant('multipliers.i_cal_pca_multiplier', ind[0]['i_cal_pca_multiplier']*8)
    dat1 = sim.run(100000)

    ## EAD CHALLENGE: IKr = 80% block (acute increase - 100 beats prepacing)
    sim.reset()
    sim.set_state(IC)
    sim.set_constant('multipliers.i_cal_pca_multiplier', ind[0]['i_cal_pca_multiplier'])
    sim.set_constant('multipliers.i_kr_multiplier', ind[0]['i_kr_multiplier']*0.001)
    sim.set_constant('multipliers.i_kb_multiplier', ind[0]['i_kb_multiplier']*0.5)
    dat2 = sim.run(100000)

    return dat, dat1, dat2

def get_rrc_error(RRC, cost):

    #################### RRC DETECTION & ERROR CALCULATION ##########################
    error = 0
    RRC_est = RRC

    if cost == 'function_1':
        error += round((0.3 - (np.abs(RRC)))*20000)

    else:
        # This just returns the error from the first RRC protocol
        stims = np.asarray([0, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3])
        pos_error = [5000, 4500, 4000, 3500, 3000, 2500, 2000, 1500, 1000, 500, 0]
        i = (np.abs(stims - RRC)).argmin()
        check_low = stims[i]-stims[i-1]
        check_high = stims[i]-stims[i+1]

        if check_high<check_low:
            RRC_est = stims[i-1]
            error = pos_error[i-1]
        else:
            RRC_est = i
            error += pos_error[i]

    return error, RRC_est

def get_features(t,v,cai=None):
    """
    Compares the simulation data for an individual to the baseline Tor-ORd values. The returned error value is a sum of the differences between the individual and baseline values.
    Returns
    ------
        error
    """

    ap_features = {}

    # Returns really large error value if cell AP is not valid 
    if ((min(v) > -60) or (max(v) < 0)):
        return 50000000 

    # Voltage/APD features#######################
    mdp = min(v)
    max_p = max(v)
    max_p_idx = np.argmax(v)
    apa = max_p - mdp
    dvdt_max = np.max(np.diff(v[0:100])/np.diff(t[0:100]))

    ap_features['Vm_peak'] = max_p
    #ap_features['Vm_t'] = t[max_p_idx]
    ap_features['dvdt_max'] = dvdt_max

    for apd_pct in [40, 50, 90]:
        apd_val = calc_APD(t,v,apd_pct) 
        ap_features[f'apd{apd_pct}'] = apd_val
 
    ap_features['triangulation'] = ap_features['apd90'] - ap_features['apd40']
    ap_features['RMP'] = np.mean(v[len(v)-50:len(v)])

    if cai is not None: 
        # Calcium/CaT features######################## 
        max_cai = np.max(cai)
        max_cai_idx = np.argmax(cai)
        max_cai_time = t[max_cai_idx]
        cat_amp = np.max(cai) - np.min(cai)
        ap_features['cat_amp'] = cat_amp * 1e5 #added in multiplier since number is so small
        ap_features['cat_peak'] = max_cai_time

        for cat_pct in [90]:
            cat_recov = max_cai - cat_amp * cat_pct / 100
            idx_catd = np.argmin(np.abs(cai[max_cai_idx:] - cat_recov))
            catd_val = t[idx_catd+max_cai_idx]

            ap_features[f'cat{cat_pct}'] = catd_val 

    return ap_features

def closest(lst, K): 
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def check_physio_torord(t, v, path = '../', filter = 'no'):

    # Cut off the upstroke of the AP for profile
    t_ind = list(t[150:len(t)]) 
    v_ind = list(v[150:len(t)])

    # Baseline tor-ord model & cut off upstroke
    base_df = pd.read_csv(path + 'data/baseline_torord_data.csv')
    t_base = list(base_df['t'])[150:len(t)]
    v_base = list(base_df['v'])[150:len(t)]


    # Cut off the upstroke of the AP for the tor-ord data
    if filter == 'no':
        time, vol_10, vol_90 = get_torord_phys_data(path, filter)
        t = time[150:len(time)]
        v_10 = vol_10[150:len(time)]
        v_90 = vol_90[150:len(time)]

    else:
        t, v_10, v_90 = get_torord_phys_data(path, filter)

    result = 0 # valid AP
    #fail_time = 3000 
    error = 0
    check_times = []
    data = {}

    for i in list(range(0, len(t_ind))):
        t_dat = closest(t, t_ind[i]) # find the value closest to the ind's time within the exp data time list
        t_dat_base = closest(t_base, t_ind[i])
        t_dat_i = np.where(np.array(t)==t_dat)[0][0] #find the index of the closest value in the list 
        t_dat_base_i = np.where(np.array(t_base)==t_dat_base)[0][0] #find the index of the closest value in the list 
        v_model = v_ind[i]
        v_lowerbound = v_10[t_dat_i]
        v_upperbound = v_90[t_dat_i]
        v_torord = v_base[t_dat_base_i] 

        check_times.append(np.abs(t_ind[i] - t_dat))

        if v_model < v_lowerbound or v_model > v_upperbound:
            result = 1 # not a valid AP
            #error += 10 - used in GA 5
            error += (v_model - v_torord)**2
    
    data['result'] = result
    data['error'] = error
    data['check_times'] = check_times

    return(data)

def check_physio(ap_features, cost = 'function_2', feature_targets = {'Vm_peak': [10, 33, 55], 'dvdt_max': [100, 347, 1000], 'apd40': [85, 198, 320], 'apd50': [110, 220, 430], 'apd90': [180, 271, 440], 'triangulation': [50, 73, 150], 'RMP': [-95, -88, -80]}):

    error = 0
    if cost == 'function_1':
        for k, v in feature_targets.items():
            if ((ap_features[k] > v[0]) and (ap_features[k] < v[2])):
                error+=0
            else:
                error+=(v[1]-ap_features[k])**2
    else:
        for k, v in feature_targets.items():
            if ((ap_features[k] < v[0]) and (ap_features[k] > v[2])):
                error+=1000

    return(error)

def plot_data(data_frame, c, data_type, title):
    """
    plots data of a population of models that is only within 
    physiological bounds. (0 represents normal, 1 represents abnormal)

    data_frame: population of models data you are looking to analyze
    c: color of plot
    data_type: [['column name of x data', 'column name of y data'], ['x data', 'y data'], etc.]
    title: ['data 1 plot title', 'data 2 plot title', etc.]
    """
    AP_label = []

    fig, axs = plt.subplots(len(data_type), figsize=(15, 15))
    for i in list(range(0, len(data_frame[data_type[0][0]]))):

        # Check for valid AP
        ap_features = get_features(eval(data_frame[data_type[0][0]][i]), eval(data_frame[data_type[0][1]][i]))
        error = check_physio(ap_features)
        if error == 0:
            AP_label.append(0)
        else:
            AP_label.append(1)

        # Plotting Data
        for p in list(range(0,len(data_type))):

            axs[p].plot((eval(data_frame[data_type[p][0]][i])), eval(data_frame[data_type[p][1]][i]), color = c, alpha = 0.5)
            axs[p].set_ylabel("Voltage (mV)")
            axs[p].set_title(title[p])
            if p == len(data_type):
                axs[p].set_xlabel('Time (ms)')

    return(AP_label)

def calc_APD_change(data, base_label, purturbation, element): 
    initial_APD90 = calc_APD(np.array(eval(data['t_'+base_label][element])), np.array(eval(data['v_'+base_label][element])), 90)
    final_APD90 = calc_APD(np.array(eval(data['t_'+purturbation][element])), np.array(eval(data['v_'+purturbation][element])), 90)
    percent_change = ((final_APD90-initial_APD90)/(initial_APD90))*100
    return(percent_change)

def calculate_variance(d):
    vari = []
    for j in list(range(0, len(eval(d['ind'][0])))):
        con = []

        for i in list(range(0,len(d['ind']))):
            con.append(list(eval(d['ind'][i]).values())[j])

        vari.append(np.var(con))

    return(vari) 

def count_abnormal_APs_old(data_frame, data_type):
    result = []

    for i in list(range(0, len(data_frame['t_' + data_type]))):
        time_point = eval(data_frame['t_' + data_type].tolist()[0])[0]
        t = [e-time_point for e in eval(data_frame['t_' + data_type].tolist()[i])]
        v = [eval(data_frame['v_' + data_type].tolist()[i])][0]
        print(t)
        print(v)

        EAD = detect_EAD(t,v)
        RF = detect_RF(t,v)
        #features = check_physio(t, v)

        if EAD==1 or RF==1: #or features['apd90']>440: #or features['Vm_peak']<10:
            result.append(1)
        else:
            result.append(0)
    return(result)

def count_abnormal_APs(data_frame, purturb, data_type):
    result = {}
    for p in list(range(0, len(purturb))):
        results = []
        for i in list(range(0, len(data_frame['t_'+str(purturb[p])+'_'+data_type]))):
            t = eval(data_frame['t_'+str(purturb[p])+'_'+data_type][i])
            v = eval(data_frame['v_'+str(purturb[p])+'_'+data_type][i])
            EAD = detect_EAD(t, v)
            RF = detect_RF(t, v)

            if EAD == 1 or RF ==1:
                results.append(1)
            else:
                results.append(0)
    
        result[str(purturb[p])] = (results.count(1)/len(results))*100
    return(result)

def get_normal_sim_dat(mod, proto):
    """
        Runs simulation for a given individual. If the individuals is None,
        then it will run the baseline model
        Returns
        ------
            t, v, cai, i_ion
    """
    proto.schedule(5.3, 0.1, 1, 1000, 0) 
    sim = myokit.Simulation(mod,proto)
    sim.pre(1000 * 100) #pre-pace for 100 beats
    dat = sim.run(5000)
    IC = sim.state()

    # Get t, v, and cai for second to last AP#######################
    #t, v, cai, i_ion = get_last_ap(dat, -2)
    data = get_last_ap(dat, -2)

    return (data, IC) #return (t, v, cai, i_ion, IC)

def delete_zeros(data, name, path):
    drop_rows = []
    for x in ['t_20', 'v_20', 't_40', 'v_40', 't_60', 'v_60', 't_80', 'v_80', 't_100', 'v_100']:
        for i in list(range(0, len(data['t_20']))):
            if len(data[x][i])<2:
                drop_rows.append(i)

    data.drop(labels = drop_rows, axis = 0, inplace = True)
    data.reset_index(inplace = True)
    data.drop('index', axis = 1, inplace = True)

    return(data)

def format_strings(data, name, path):
    for x in ['t_20', 'v_20', 't_40', 'v_40', 't_60', 'v_60', 't_80', 'v_80', 't_100', 'v_100']:
        for i in list(range(0, len(data['t_20']))):
            if len(data[x][i])>2:
                data[x][i] = re.sub("\s+", ",", data[x][i].strip())
                data[x][i] = data[x][i].rstrip(',')
                if data[x][i][1] == ',':
                    data[x][i] = data[x][i].replace(',', '', 1)        

    return(data)

def plot_error(data, type, ax, title, ylim = [-10,5000], lab = 'best', c_best = 'orange'):
    """
    This function can be used to plot the average and best error found from the GA so convergence could be assessed. 
    Args:
        error: dataframe with error values for each generation
        type: 'average' (average error will be plot for each generation) or 'best' (best error will be plot for each generation)
        max_gen: maximum number of generations that the GA was run for. 
        save_to: path you would like to save the plot to. 
    """
    total_gens = max(data['gen'])
    avgs = []
    bests = []

    for i in list(range(0, total_gens)):
        error = data[data['gen']==i]
        avg = sum(error['fitness'])/len(error['fitness'])
        avgs.append(avg)
        best = min(error['fitness'])  
        bests.append(best)

    if type == 'average':
        ax.scatter(list(range(0,total_gens)), avgs, label = 'average')
        ax.scatter(list(range(0,total_gens)), bests, label = 'best')
        ax.set_ylim(ylim)
        ax.set_xlabel('Number of Generations')
        ax.set_ylabel('Error')
        ax.legend(frameon = False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(title)

    if type == 'best':
        ax.scatter(list(range(0,total_gens)), bests, label = lab, color = c_best)
        ax.tick_params(axis='both', labelsize=20)
        ax.set_xlabel('Number of Generations', fontsize=25)
        ax.set_ylabel('Error', fontsize=25)
        ax.legend(frameon = False, fontsize=18)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_title(title, fontsize=25)

def plot_rrc_protocol(path = './', model = 'tor_ord_endo2.mmt'):
    mod, proto, x = myokit.load(path+model)
    proto.schedule(5.3, 0.2, 1, 1000, 0)
    proto.schedule(0, 4, 995, 1000, 1)
    proto.schedule(0.075, 5004, 995, 1000, 1)
    proto.schedule(0.1, 10004, 995, 1000, 1)
    proto.schedule(0.125, 15004, 995, 1000, 1)
    proto.schedule(0.15, 20004, 995, 1000, 1)
    proto.schedule(0.175, 25004, 995, 1000, 1)
    proto.schedule(0.2, 30004, 995, 1000, 1)
    proto.schedule(0.225, 35004, 995, 1000, 1)
    proto.schedule(0.25, 40004, 995, 1000, 1)
    proto.schedule(0.275, 45004, 995, 1000, 1)
    proto.schedule(0.3, 50004, 995, 1000, 1)

    sim = myokit.Simulation(mod, proto)
    dat = sim.run(52000)

    fig, axs = plt.subplots(2, constrained_layout=True, figsize=(15,7))
    fig.suptitle('RRC Protocol', fontsize=25)
    axs[0].set_title('RRC Challenge: 0, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3')
    axs[0].plot(dat['engine.time'], dat['membrane.v'])
    axs[0].set_ylabel("Membrane Potential (mV)")
    axs[1].plot(dat['engine.time'], dat['stimulus.i_stim'])
    axs[1].set_xlabel("time (ms)")
    axs[1].set_ylabel("stimulus (A/F)")

def visualize_profile(profile, ax, title, ax_current = 0, p = '../', lines = 'solid', model = 'tor_ord_endo2.mmt', stop_duplicates = 0, ap_color = 'black', c_color = ['yellow','blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink']):
    if model == 'kernik.mmt':
        ind = get_ind([profile[0], profile[1], profile[2], profile[3]*187/129, profile[4], profile[5]*11.24/5.67, profile[6]], celltype='ipsc')
        dat, IC = run_model([ind], 1, stim = 1, length = 5, path = p, model=model)
        labels = ['ICaL', 'IKr', 'IK1', 'IKs', 'If', 'Ito']

    else:
        ind = get_ind(vals = profile)
        dat, IC = run_model([ind], 1, path=p, model=model)
        labels = ['ICaL', 'IKr', 'IK1', 'INCX', 'IKs', 'IKb', 'INaK', 'Ito']
    
    
    # Plot AP
    ax.plot(dat['engine.time'], dat['membrane.v'], color = ap_color, linestyle = lines)

    if stop_duplicates == 0:
        t, v_10, v_90 = get_torord_phys_data(path=p)
        ax.plot(t, v_10, color = 'green', alpha = 0.25)
        ax.plot(t, v_90, color = 'green', alpha = 0.25)
        ax.fill_between(t, v_10, v_90, color='green', alpha = 0.25)


    if ax_current !=0:
        # Plot Baseline Currents
        if model == 'tor_ord_endo2.mmt':
            currents = ['ICaL.ICaL', 'IKr.IKr', 'IK1.IK1', 'INaCa.INaCa_i','IKs.IKs', 'IKb.IKb', 'INaK.INaK', 'Ito.Ito']
        if model == 'grandi_flat.mmt':
            currents = ['I_Ca.I_Ca', 'I_Kr.I_kr','I_Ki.I_ki', 'I_NCX.I_ncx', 'I_Ks.I_ks', 'I_Kp.I_kp', 'I_NaK.I_nak', 'I_to.I_to']
        if model == 'kernik.mmt':
            currents = ['ical.i_CaL', 'ikr.i_Kr', 'ik1.i_K1', 'iks.i_Ks', 'ifunny.i_f', 'ito.i_to']
        
        
        for i in list(range(0, len(currents))):
            ax_current.plot(dat['engine.time'], dat[currents[i]], label = labels[i], linestyle = lines, color = c_color[i])
        #ax_current.set_ylim([-8,3])
        ax_current.set_xlim([-10, 600])
        if stop_duplicates == 0:
            ax_current.legend(loc = 'lower right')

    # Set Plot 
    ax.set_title(title)
    ax.set_ylim([-90,40])
    ax.set_xlim([-10,600])

    #for ax in fig.get_axes():
    #    ax.label_outer()
    
    return(dat)

def define_pop_experiment_1(ind, cond, analysis, purturb, p):

    data = []

    # Stimulus Analysis
    if analysis == 'stim':
        mod, proto = get_ind_data([ind], path=p)
        proto.schedule(5.3, 0.1, 1, 1000, 0) 
        sim = myokit.Simulation(mod,proto)
        sim.pre(100*1000) #pre-pace for 100 beats
        IC = sim.state()
        for i in list(range(0,len(purturb))):
            ## each purturbation
            #sim.reset() - starts time at 0 
            sim.set_state(IC)
            proto.schedule(purturb[i], 4004+(i*1000), 995, 1000, 1)
            sim.set_protocol(proto)
            dat1 = sim.run(1000)
            data.append(list(dat1['engine.time']))
            data.append(list(dat1['membrane.v']))
    
    # Alternans Analysis
    if analysis == 'alt':
        dat1, IC = run_model(ind, 4, cl = 270, location = 'extracellular', ion = 'cao', value = 2, path = p)
        data.append(list(dat1['engine.time']))
        data.append(list(dat1['membrane.v'])) 
    
    # Cond Analysis
    if analysis == 'cond':
        mod, proto = get_ind_data([ind], path=p)
        proto.schedule(5.3, 0.1, 1, 1000, 0) 
        sim = myokit.Simulation(mod,proto)
        sim.pre(50*1000) #pre-pace for 50 beats
        IC = sim.state()
    
        for i in list(range(0,len(purturb))):
            sim.set_state(IC)
            sim.set_constant('multipliers.'+cond, ind[cond]*purturb[i])
            sim.pre(100)
            dat_acute = sim.run(5*1000) 

            sim.pre(500)
            sim.set_time()
            dat_long = sim.run(5*1000)

            data.append(list(dat_acute['engine.time']))
            data.append(list(dat_acute['membrane.v']))

            data.append(list(dat_long['engine.time']))
            data.append(list(dat_long['membrane.v']))

    return data

def define_pop_experiment_2(ind, cond, analysis, purturb, p, model = 'tor_ord_endo2.mmt'):

    data = {}

    # Stimulus Analysis
    if analysis == 'stim':
        """
        mod, proto = get_ind_data([ind], path=p)
        proto.schedule(5.3, 0.1, 1, 1000, 0) 
        sim = myokit.Simulation(mod,proto)
        sim.pre(600*1000) #pre-pace for 100 beats
        IC = sim.state()
        for i in list(range(0,len(purturb))):
            ## each purturbation
            sim.reset() # starts time at 0 
            sim.set_state(IC)
            proto.schedule(purturb[i], 4004+(i*1000), 995, 1000, 1)
            sim.set_protocol(proto)
            dat1 = sim.run(1000)
            data.append(list(dat1['engine.time']))
            data.append(list(dat1['membrane.v']))
        """

        APs = list(range(10004, 100004, 5000))

        mod, proto = get_ind_data([ind], path = p, model = model) 
        proto.schedule(5.3, 0.2, 1, 1000, 0)
        sim = myokit.Simulation(mod, proto)
        sim.pre(600)
        IC = sim.state()

        for i in list(range(0,len(purturb))):
            sim.reset()
            sim.set_state(IC)
            proto.schedule(purturb[i], APs[i], 995, 1000, 1)
            sim.set_protocol(proto)
            dat = sim.run(APs[i]+2000)

            #d = get_last_ap(dat, int((APs[i]-4)/1000))
            data['t_'+str(purturb[i])] = list(dat['engine.time'])
            data['v_'+str(purturb[i])] = list(dat['membrane.v'])
    
    # Alternans Analysis
    if analysis == 'alt':
        dat1, IC = run_model([ind], 4, cl = 270, location = 'extracellular', ion = 'cao', value = 2, path = p, model = model)
        data['t'] = (list(dat1['engine.time']))
        data['v'] = (list(dat1['membrane.v'])) 
    
    # Cond Analysis
    if analysis == 'cond':
        dat_initial, sim, IC_initial = run_model([ind], 5, prepace = 600, return_sim=1,  path = p, model = model)
    
        data['t_initial'] = list(dat_initial['engine.time'])
        data['v_initial'] = list(dat_initial['membrane.v'])
        #data.append(list(dat_initial['stimulus.i_stim']))
        
        for i in list(range(0,len(purturb))):
            dat_acute, sim, IC_acute = run_model([ind], 5, location = 'multipliers', ion=cond, value = ind[cond]*purturb[i], prepace = 100, return_sim = 1, sim = sim, I0 = IC_initial, path = p, model = model)

            dat_long, IC_long = run_model([ind], 5, location = 'multipliers', ion=cond, value = ind[cond]*purturb[i], prepace = 500, sim = sim, I0 = IC_acute, path = p, model = model)

            # SAVE DATA
            data['t_'+str(purturb[i])+'_acute'] = list(dat_acute['engine.time'])
            data['v_'+str(purturb[i])+'_acute'] = list(dat_acute['membrane.v'])
            #data.append(list(dat_acute['stimulus.i_stim']))

            data['t_'+str(purturb[i])+'_long'] = list(dat_long['engine.time'])
            data['v_'+str(purturb[i])+'_long'] = list(dat_long['membrane.v'])
            #data.append(list(dat_long['stimulus.i_stim']))

    return data

def define_pop_experiment(ind, cond, analysis, purturb, p, model = 'tor_ord_endo2.mmt'):
    cond_initial = ind[cond]
    #print(cond, cond_initial)

    data = {}

    # Stimulus Analysis
    if analysis == 'stim':

        APs = list(range(10004, 100004, 5000))

        mod, proto = get_ind_data([ind], path = p, model = model) 
        proto.schedule(5.3, 0.2, 1, 1000, 0)
        sim = myokit.Simulation(mod, proto)
        sim.pre(600)
        IC = sim.state()

        for i in list(range(0,len(purturb))):
            sim.reset()
            sim.set_state(IC)
            proto.schedule(purturb[i], APs[i], 995, 1000, 1)
            sim.set_protocol(proto)
            dat = sim.run(APs[i]+2000)

            #d = get_last_ap(dat, int((APs[i]-4)/1000))
            data['t_'+str(purturb[i])] = list(dat['engine.time'])
            data['v_'+str(purturb[i])] = list(dat['membrane.v'])
    
    # Alternans Analysis - THIS NEEDS TO BE FIXED - WILL NOT WORK!
    if analysis == 'alt':
        dat1, IC = run_model([ind], 4, cl = 270, location = 'extracellular', ion = 'cao', value = 2, path = p, model = model)
        data['t'] = (list(dat1['engine.time']))
        data['v'] = (list(dat1['membrane.v'])) 
    
    # Cond Analysis
    if analysis == 'cond':
        dat_initial, IC_initial = run_model([ind], 5, prepace = 600,  path = p, model = model)
    
        data['t_initial'] = list(dat_initial['engine.time'])
        data['v_initial'] = list(dat_initial['membrane.v'])
        
        for i in list(range(0,len(purturb))):
            #print('cond initial', cond_initial)
            #print(ind[cond])
            ind[cond] = cond_initial*purturb[i] 
            #print(ind[cond])
            dat_long, IC_long = run_model([ind], 5, prepace = 600, I0 = IC_initial, path = p, model = model)

            # SAVE DATA
            data['t_'+str(purturb[i])+'_long'] = list(dat_long['engine.time'])
            data['v_'+str(purturb[i])+'_long'] = list(dat_long['membrane.v'])

            # COUNT ABNORMAL AP
            ap_analysis = []
            for ap in list(range(1,5)):
                df_lastap = get_last_ap([dat_long['engine.time'], dat_long['membrane.v']], ap, type = 'part')
                t_ap = df_lastap['t']-df_lastap['t'][0]
                v_ap = df_lastap['v']

                EAD = detect_EAD(t_ap, v_ap)
                RF = detect_RF(t_ap, v_ap)

                if EAD == 1 or RF ==1:
                    ap_analysis.append(1)
                else:
                    ap_analysis.append(0)

            if ap_analysis.count(1) != 0:
                data['abAP_'+str(purturb[i])] = 1
            else:
                data['abAP_'+str(purturb[i])] = 0

    return data

def collect_pop_data(args):
    i, immunization_profile, cond, analysis, purturb, p, p_models = args
    #ind = initialize_individuals()
    ind = load_data(i, p_models+'pop_models.csv')
    #print(ind['i_kr_multiplier'])

    ind_i = immunize_ind_data(ind, immunization_profile)
    #print(ind_i['i_kr_multiplier'])
    #ind_i = load_data(i, 'population/filtered_immune_data.csv')


    try:
        data_base = define_pop_experiment(ind.copy(), cond, analysis, purturb, p)
        data_immune = define_pop_experiment(ind_i.copy(), cond, analysis, purturb, p)
        
        data_immune = { k+'_i': v for k, v in data_immune.items() }
        ind_i = { k+'_i': v for k, v in ind_i.items() }
        
        data = {**ind, **ind_i, **data_base, **data_immune}
    
    except:

        data = {**ind, **ind_i}

    return(data)

def collect_pop_data_1(args):
    i, cond, analysis, purturb, p, p_models, p_optmodels = args
    ind = load_data(i, p_models)
    ind_i = load_data(i, p_optmodels)

    try:
        data_base = define_pop_experiment(ind.copy(), cond, analysis, purturb, p)
        data_immune = define_pop_experiment(ind_i.copy(), cond, analysis, purturb, p)
        
        data_immune = { k+'_i': v for k, v in data_immune.items() }
        ind_i = { k+'_i': v for k, v in ind_i.items() }
        
        data = {**ind, **ind_i, **data_base, **data_immune}
    
    except:

        data = {**ind, **ind_i}

    return(data)

def collect_sensitivity_data(args):
    i, cond, purturb, p, model = args
    ind = get_ind()
    data = {}

    dat_initial, IC_initial = run_model([ind], 5, prepace = 600,  path = p, model = model)

    print('finished initial conditions')

    data['t_initial'] = list(dat_initial['engine.time'])
    data['v_initial'] = list(dat_initial['membrane.v'])
    
    for pur in list(range(0,len(purturb))):
        #location = 'multipliers', ion=cond[i], value = ind[cond[i]]*purturb[pur]
        ind[cond[i]] = purturb[pur]
        print(cond[i], purturb[pur])
        dat_long, IC_long = run_model([ind], 5, prepace = 600, I0 = IC_initial, path = p, model = model)
        
        print('finished running model')

        try:
            results = rrc_search([ind], IC_long, path = p, model=model) 
            RRC = results['RRC']
        except:
            RRC = 0

        print('finished finding RRC')

        # SAVE DATA
        data['t_'+str(purturb[pur])] = list(dat_long['engine.time'])
        data['v_'+str(purturb[pur])] = list(dat_long['membrane.v'])
        data['rrc_'+str(purturb[pur])] = RRC

        #for ap in list(range(0,len(results['t']))):
        #    plt.plot(results['t'][ap], results['v'][ap], label = results['stims'][ap])
        #plt.legend()
        #plt.show()

    data = {**{'cond': cond[i]}, **data}

    return(data)

def collect_apd_sensitivity_data(args):
    ind, i, cond, purturb, p, model = args
    #ind = get_ind()
    data = {}

    dat_initial, IC_initial = run_model([ind], 5, prepace = 600,  path = p, model = model)

    print('finished initial conditions')

    data['t_initial'] = list(dat_initial['engine.time'])
    data['v_initial'] = list(dat_initial['membrane.v'])

    for pur in list(range(0,len(purturb))):
        #location = 'multipliers', ion=cond[i], value = ind[cond[i]]*purturb[pur]
        ind[cond[i]] = purturb[pur]
        print(cond[i], purturb[pur])
        dat_long, IC_long = run_model([ind], 5, prepace = 600, I0 = IC_initial, path = p, model = model)
        
        print('finished running model')

        apds = []
        for ap in list(range(0,5)):
            ap_data_initial = get_last_ap([dat_initial['engine.time'], dat_initial['membrane.v']], ap, cl = 1000, type = 'part')
            ap_data_final = get_last_ap([dat_long['engine.time'], dat_long['membrane.v']], ap, cl = 1000, type = 'part')
            initial_APD90 = calc_APD(ap_data_initial['t'], ap_data_initial['v'], 90)
            final_APD90 = calc_APD(ap_data_final['t'], ap_data_final['v'], 90)
            change = final_APD90-initial_APD90 
            apds.append(change)
        
        print('finished finding change APD')

        #try:
        results = rrc_search([ind], IC_long, path = p, model=model) 
        RRC = results['RRC']
        #except:
        #RRC = 0

        print('finished finding RRC')

        # SAVE DATA
        data['t_'+str(purturb[pur])] = list(dat_long['engine.time'])
        data['v_'+str(purturb[pur])] = list(dat_long['membrane.v'])
        data['rrc_'+str(purturb[pur])] = RRC
        data['apd_change_'+str(purturb[pur])] = apds

        #for ap in list(range(0,len(results['t']))):
        #    plt.plot(results['t'][ap], results['v'][ap], label = results['stims'][ap])
        #plt.legend()
        #plt.show()

    data = {**{'cond': cond[i]}, **data}

    return(data)

def my_mode(sample):
    c = Counter(sample)
    return [k for k, v in c.items() if v == c.most_common(1)[0][1]]

def apd_analysis(pur, df, type, ind):
    data_apd = {}
    for p in list(range(0, len(pur))):
        d = [np.array(eval(df['t_'+str(pur[p])+type][ind])), np.array(eval(df['v_'+str(pur[p])+type][ind]))]
        APD90s = []
        for i in list(range(0,4)):
            data = get_last_ap(d, i, type = 'part')
            APD90 = calc_APD(data['t'], data['v'], 90)
            APD90s.append(APD90)
        
        data_apd[str(pur[p])] = np.mean(APD90s)
    return(data_apd)

def plot_pop_study(df_cond, block, label, axs):

    base_data = []
    immune_data = []

    for i in list(range(0, len(block))):
        base_data.append((list(df_cond['abAP_'+str(block[i])]).count(1)/len(list(df_cond['abAP_'+str(block[i])])))*100)
        immune_data.append((list(df_cond['abAP_'+str(block[i])+'_i']).count(1)/len(list(df_cond['abAP_'+str(block[i])+'_i'])))*100)

    if block[1]<1:
        x_axis = [(1-val)*100 for val in block]
    else:
        x_axis = [(val)*100 for val in block]
    axs.plot(x_axis, base_data, '--ro', label = 'Baseline Data')
    axs.plot(x_axis, immune_data, '--bo', label = 'Immunized Data 1')
    axs.set_xlabel('IKr Block (%)', fontsize=30)
    axs.set_ylabel('Abnormal AP (%)', fontsize = 30)

    plt.legend()

    labels = ['label']+[label+'_'+str(p) for p in block]
    data = [['Baseline']+base_data, ['Immunized']+immune_data]

    df = pd.DataFrame(data, columns = labels)
    return(df)

def plot_cond_study(df_cond, block, label, x_label, model, axs, random_colors, best_idx):
    if block[1] <1:
        x_axis = [(1-val)*100 for val in block]
    else:
        x_axis = [(val)*100 for val in block]

    #base_ac_data = apd_analysis(pur, ikr, '_acute', 0)
    base_lo_data = apd_analysis(block, df_cond, '_long', 0)
    axs[0].plot(x_axis, list(base_lo_data.values()), '--ko', label = 'Baseline Data')
    axs[1].plot(x_axis,  [((i-list(base_lo_data.values())[0])/list(base_lo_data.values())[0])*100 for i in list(base_lo_data.values())], '--ko', label = 'Baseline Data')

    for ind in list(range(0, len(df_cond['v_1_long']))):
        ind_lo_data = apd_analysis(block, df_cond, '_long_i', ind)

        axs[0].plot(x_axis, list(ind_lo_data.values()), color = random_colors[ind], marker = 'o', label = 'Ind '+str(best_idx[ind]))
        axs[0].set_xlabel(x_label, fontsize = 30)
        axs[1].set_ylabel('Change in APD90 (%)', fontsize = 30)

        axs[1].plot(x_axis,  [((i-list(ind_lo_data.values())[0])/list(ind_lo_data.values())[0])*100 for i in list(ind_lo_data.values())], color = random_colors[ind], marker = 'o', label = 'Ind '+str(best_idx[ind]))
        axs[1].set_xlabel(x_label, fontsize = 30)
        axs[0].set_ylabel('APD (ms)', fontsize = 30)

    plt.legend()

    plt.savefig('./figures/'+label+'_study_'+model+'.png', dpi=300, trasparent=True)
    plt.show()
