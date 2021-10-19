
import numpy as np
import yaml
        
from ca3pyr_dhh import ca3pyrcell
from ca1pyr_dhh import ca1pyrcell
from pvbc import PVBC
from axoaxonic import AAC
from bis import BiS
from olm import OLM
from isccr import ISCCR
from iscck import ISCCK

from pvbc_v2 import PVBC_v2
from axoaxonic_v2 import AAC_v2
from bis_v2 import BiS_v2
from olm_v2 import OLM_v2

from septal import Septal
from neuron import h
from ca3_neuron_utils import create_netcon




class Circuit(object):
    
    def __init__(self, params_filepath, internal_pop2id, external_pop2id):
        self._read_params_filepath(params_filepath)
        self.internal_pop2id = internal_pop2id
        self.external_pop2id = external_pop2id

        self.neurons  = {}
        self.netstims = {}
        self.external_spike_times = {}
        
        self.lfp = None
    
    
    def _read_params_filepath(self, params_filepath):
        with open(params_filepath, 'r') as f:
            fparams = yaml.load(f, Loader=yaml.FullLoader)           
            self.params = fparams['Circuit']
            
            
    def build_cells(self):
        absolute_gid = 0
        cell_types   = self.params['cells']
        cell_numbers = self.params['ncells']
        for (i,ctype) in enumerate(cell_types):
            cidx = self.internal_pop2id[ctype]
            nc = cell_numbers[i]
            self.neurons[cidx] = {}
            for ctype_gid in range(nc):
                if ctype == 'ca3pyr':
                    self.neurons[cidx][ctype_gid] = ca3pyrcell(absolute_gid, 'B', 'CA3_Bakerb_marascoProp.pickle')
                elif ctype == 'ca1pyr':
                    self.neurons[cidx][ctype_gid]  = ca1pyrcell(absolute_gid)
                elif ctype == 'pvbc':
                    self.neurons[cidx][ctype_gid] = PVBC(absolute_gid)
                elif ctype == 'axoaxonic':
                    self.neurons[cidx][ctype_gid] = AAC(absolute_gid)
                elif ctype == 'cckbc':
                    self.neurons[cidx][ctype_gid] = CCKBC_v2(absolute_gid)
                elif ctype == 'bis':
                    self.neurons[cidx][ctype_gid] = BiS(absolute_gid)
                elif ctype == 'olm':
                    self.neurons[cidx][ctype_gid] = OLM_v2(absolute_gid)
                elif ctype == 'isccr':
                    self.neurons[cidx][ctype_gid] = ISCCR(absolute_gid)
                elif ctype == 'iscck':
                    self.neurons[cidx][ctype_gid] = ISCCK(absolute_gid)                  
                absolute_gid += 1
        if 'Septal' in self.params.keys():
            nc = self.params['Septal']['ncells']
            self.neurons['Septal'] = {}
            for ctype_gid in range(nc):
                sc = Septal(absolute_gid, self.params['Septal']['parameters'])
                self.neurons['Septal'][ctype_gid] = sc
                absolute_gid += 1
                
    def build_internal_netcons(self, internal_adj_matrices, seed=1e6):
        rnd = np.random.RandomState(seed=int(seed))
        src_population_ids = list(internal_adj_matrices.keys())
        for src_pop_id in src_population_ids:
            dst_population_ids = list(internal_adj_matrices[src_pop_id].keys())
            for dst_pop_id in dst_population_ids:
                adj_matrix = internal_adj_matrices[src_pop_id][dst_pop_id]
                src_neurons, dst_neurons = self.neurons[src_pop_id], self.neurons[dst_pop_id] # PYR to PVBC, for example
                src_gids = np.sort(list(src_neurons.keys()))
                dst_gids = np.sort(list(dst_neurons.keys()))
                if len(src_gids) != adj_matrix.shape[0]:
                    print('something is wrong in srcs...')
                    return
                if len(dst_gids) != adj_matrix.shape[1]:
                    print('something is wrong in dsts...')
                    return
                
                synapse_information = self.params['internal connectivity'][src_pop_id][dst_pop_id]['synapse']
                for i in range(adj_matrix.shape[0]):
                    for j in range(adj_matrix.shape[1]):
                        nconnections = adj_matrix[i,j]
                        if nconnections == 0: continue
                        compartments     = synapse_information['compartments']
                        rnd_compartments = rnd.randint(0, len(compartments), size=(nconnections,))
                        chosen_compartments = [compartments[ridx] for ridx in rnd_compartments]
                        
                        for con_num in range(nconnections):
                            compartment = chosen_compartments[con_num]
                            ncs = create_netcon(src_pop_id, dst_pop_id, src_neurons[i], dst_neurons[j], 
                                                synapse_information, compartment, self.params)
                            
                            dst_neurons[j].internal_netcons.append( (src_neurons[i].gid, ncs, compartment) )

    def build_external_netcons(self, src_pop_id, external_adj_matrices, external_spike_times, seed=1e7):
        seed = int(seed) + src_pop_id
        rnd  = np.random.RandomState(seed=seed)
        
        dst_population_ids = list(external_adj_matrices.keys())
        for dst_pop_id in dst_population_ids:
            if (src_pop_id, dst_pop_id) not in self.netstims: self.netstims[(src_pop_id, dst_pop_id)] = {}
            adj_matrix = external_adj_matrices[dst_pop_id]
            
            synapse_information = self.params['external connectivity'][src_pop_id][dst_pop_id]['synapse']
            for i in range(adj_matrix.shape[0]):
                for j in range(adj_matrix.shape[1]):
                    if (i,j) not in self.netstims[(src_pop_id, dst_pop_id)]: self.netstims[(src_pop_id, dst_pop_id)][(i,j)] = []
                    nconnections = adj_matrix[i,j]
                    if nconnections == 0: continue
                    compartments     = synapse_information['compartments']
                    rnd_compartments = rnd.randint(0, len(compartments), size=(nconnections,))
                    chosen_compartments = [compartments[ridx] for ridx in rnd_compartments]

                    
                    for con_num in range(nconnections):
                        vecStim = h.VecStim()
                        vec = h.Vector(external_spike_times[i])
                        vecStim.play(vec)
                        self.netstims[(src_pop_id,dst_pop_id)][(i,j)].append((vec, vecStim))
                        
                        compartment = chosen_compartments[con_num]
                        ncs = create_netcon(src_pop_id, dst_pop_id, None, self.neurons[dst_pop_id][j], 
                                            synapse_information, compartment, self.params, vecStim=vecStim)
                        if src_pop_id not in self.neurons[dst_pop_id][j].external_netcons: 
                            self.neurons[dst_pop_id][j].external_netcons[src_pop_id] = []
                        self.neurons[dst_pop_id][j].external_netcons[src_pop_id].append( (i, ncs, compartment) )
                        
    def build_septal_netcons(self, septal_adj_matrices, seed=1e8):
        rnd = np.random.RandomState(seed=int(seed))
        
        dst_population_ids = list(septal_adj_matrices.keys())
        for dst_pop_id in dst_population_ids:
            adj_matrix = septal_adj_matrices[dst_pop_id]
            
            synapse_information = self.params['Septal']['connectivity'][dst_pop_id]['synapse']
            for i in range(adj_matrix.shape[0]):
                for j in range(adj_matrix.shape[1]):
                    nconnections = adj_matrix[i,j]
                    if nconnections == 0: continue
                    compartments     = synapse_information['compartments']
                    rnd_compartments = rnd.randint(0, len(compartments), size=(nconnections,))
                    chosen_compartments = [compartments[ridx] for ridx in rnd_compartments]

                    for con_num in range(nconnections):
                        compartment = chosen_compartments[con_num]
                        ncs = create_netcon('Septal', dst_pop_id, self.neurons['Septal'][i], self.neurons[dst_pop_id][j], 
                                            synapse_information, compartment, self.params)
                        self.neurons[dst_pop_id][j].internal_netcons.append( (self.neurons['Septal'][i].gid, ncs, compartment) )
                       
    def build_ripples(self, weight, mean_time, Tmax, duration, fraction_place_active, place_cell_info, stim_freq, nspikes, delay=0., seed=1e4):
        
        self.ripple_netcons = []
        
        locs = place_cell_info['loc']
        gids = place_cell_info['gid']
        place_cells = [self.neurons[0][gid] for gid in gids]
        locs_sorted = np.argsort(locs)
        
        rnd = np.random.RandomState(seed=int(seed))
        ripple_start_times = []
        current_clock = delay
        while (current_clock < Tmax):
            good_sample = False
            while (not good_sample):
                ripple_time = rnd.exponential(mean_time)
                if ripple_time >= 500: good_sample = True
            ripple_start_times.append(current_clock + ripple_time)
            current_clock += ripple_time
            
        ripple_durations = np.random.randint(duration[0], duration[1], size=len(ripple_start_times))
        
        ### build ripple stims
        ncells_active = int(len(gids) * fraction_place_active)
        
        for i in range(len(ripple_start_times)):
            ripple_start, ripple_dur = ripple_start_times[i], ripple_durations[i]
            last_cell  = rnd.randint(ncells_active, len(gids))
            first_cell = last_cell - ncells_active
            #print(ripple_start_times[i], first_cell, last_cell)
            for (j, cell) in enumerate(np.arange(first_cell, last_cell)):
                ns = h.NetStim()
                ns.interval = 1000./stim_freq
                ns.start    = ripple_start + (j)
                ns.noise    = 0.1
                ns.number   = nspikes
                
                compartments = ['radTprox', 'radTmed', 'oriprox1', 'oriprox2']
                cidx = rnd.randint(0, len(compartments))
                synapse_information = {}
                synapse_information['type'] = ['AMPA', 'NMDA']
                synapse_information['weight1'] = weight
                synapse_information['tau1']    = 0.5
                synapse_information['tau2']    = 3.0
                synapse_information['e'] = 0.
                ncs = create_netcon('Ripple', 0, None, place_cells[cell],
                              synapse_information, compartments[cidx], self.params, vecStim=ns)
                self.ripple_netcons.append((ns, ncs))

                
#                 synapse_information['type'] = ['AMPA']
#                 synapse_information['weight1'] = 6.0e-4 
#                 synapse_information['tau1'] = 2.0
#                 synapse_information['tau2'] = 6.3
#                 synapse_information['e'] = 0.
#                 for pvbc_c in range(0):
#                     ridx = rnd.randint(0, len(self.neurons[1]))
#                     compartments = ['radM1', 'radM2', 'radT1', 'radT2']
#                     cidx = rnd.randint(0, len(compartments))
#                     ncs = create_netcon('Ripple', 1, None, self.neurons[1][ridx],
#                                         synapse_information, compartments[cidx], self.params, vecStim=ns)
#                     self.ripple_netcons.append((ns, ncs))

#         synapse_information['type'] = ['AMPA']
#         synapse_information['weight1'] = 7.5e-4 
#         synapse_information['tau1'] = 0.5
#         synapse_information['tau2'] = 3.0
#         synapse_information['e'] = 0.
#         for pvbc_id in range(len(self.neurons[1])):
#             for i in range(ncells_active):
#                 compartments = ['radM1', 'radM2', 'radT1', 'radT2']
#                 cidx         = rnd.randint(0, len(compartments))
#                 #ripple_stim  = rnd.randint(0, ncells_active)
#                 ns = self.ripple_netcons[i][0]
#                 ncs = create_netcon('Ripple', 1, None, self.neurons[1][pvbc_id],
#                                     synapse_information, compartments[cidx], self.params, vecStim=ns)
#                 self.ripple_netcons.append((ns, ncs))



                    
        return ripple_start_times
                

        

    def get_cell_spikes(self, group_id):
        neurons = self.neurons[group_id]
        spike_times = []
        for k in list(neurons.keys()):
            cell = neurons[k]
            spike_times.append(list(cell.spike_times))
        return spike_times
    
    def record_lfp(self, population_ids):
#         valid_synlocs = ['soma', 'oriensProximal', 'oriprox1', 'oriprox2', 'oriT1', 'oriT2', 'lucidum', 'radiatum', 'radT1', 'radT2', 'radM1', 'radM2']
        neurons = self.neurons
        self.lfp = []
        for pop_id in population_ids:
            current_neural_population = neurons[pop_id]
            for gid in current_neural_population.keys():
                current_neuron = current_neural_population[gid]
                for syntype in current_neuron.synGroups:
                    for synlocation in current_neuron.synGroups[syntype]:
                        #if synlocation in valid_synlocs:
                        synapses = current_neuron.synGroups[syntype][synlocation]
                        for pid in synapses.keys():
                            syns = synapses[pid]
                            for syn in syns:
                                curr = h.Vector()
                                curr.record(syn._ref_i)
                                self.lfp.append(curr)

        
        

        
        
