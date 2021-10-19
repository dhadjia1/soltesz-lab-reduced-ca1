' Author: Darian Hadjiabadi '

import numpy as np
from neuron import h, gui

class ca1pyrcell(object):

    def __init__(self, gid):
        self.gid        = gid
        self.prelist    = []
        self.all        = None
        self.soma       = None
        self.radTprox   = None
        self.radTmed    = None
        self.radTdist   = None
        self.lm_thick2  = None
        self.lm_medium2 = None
        self.lm_thin2   = None
        self.lm_thick1  = None
        self.lm_medium1 = None
        self.lm_thin1   = None
        self.oriprox1   = None
        self.oridist1   = None
        self.oriprox2   = None
        self.oridist2   = None
        self.axon       = None
        self.init()

    def init(self):
        self.topol()
        self.subsets()
        self.geom()
        self.biophys()
        self.geom_nseg()
        
        self.synGroups = None
        self.generate_synapses()
        
        self.internal_netcons = []
        self.external_netcons = {}
        self.spike_detector = h.NetCon(self.axon(0.5)._ref_v, None, sec=self.axon)
        self.spike_times = h.Vector()
        self.spike_detector.record(self.spike_times)
        

    def topol(self):
        self.soma       = h.Section(name='soma', cell=self)
        self.radTprox   = h.Section(name='radTprox', cell=self)
        self.radTmed    = h.Section(name='radTmed', cell=self)
        self.radTdist   = h.Section(name='radTdist', cell=self)
        self.lm_thick2  = h.Section(name='lm_thick2', cell=self)
        self.lm_medium2 = h.Section(name='lm_medium2', cell=self)
        self.lm_thin2   = h.Section(name='lm_thin2', cell=self)
        self.lm_thick1  = h.Section(name='lm_thick1', cell=self)
        self.lm_medium1 = h.Section(name='lm_medium1', cell=self)
        self.lm_thin1   = h.Section(name='lm_thin1', cell=self)
        self.oriprox1   = h.Section(name='oriprox1', cell=self)
        self.oridist1   = h.Section(name='oridist1', cell=self)
        self.oriprox2   = h.Section(name='oriprox2', cell=self)
        self.oridist2   = h.Section(name='oridist2', cell=self)
        self.axon       = h.Section(name='axon', cell=self)

        self.radTprox.connect(self.soma(1))
        self.radTmed.connect(self.radTprox(1))
        self.radTdist.connect(self.radTmed(1))
        self.lm_thick2.connect(self.radTdist(1))
        self.lm_medium2.connect(self.lm_thick2(1))
        self.lm_thin2.connect(self.lm_medium2(1))
        self.lm_thick1.connect(self.radTdist(1))
        self.lm_medium1.connect(self.lm_thick1(1))
        self.lm_thin1.connect(self.lm_medium1(1))
        self.oriprox1.connect(self.soma(0))
        self.oridist1.connect(self.oriprox1(1))
        self.oriprox2.connect(self.soma(1))
        self.oridist2.connect(self.oriprox2(1))
        self.axon.connect(self.soma(1))
        
        #self.basic_shape()

    def basic_shape(self):
        h.pt3dclear(sec=self.soma)
        h.pt3dadd(0, 0, 0, 1, sec=self.soma)
        h.pt3dadd(15, 0, 0, 1, sec=self.soma)
        
        h.pt3dclear(sec=self.radTprox)
        h.pt3dadd(15, 0, 0, 1, sec=self.radTprox)
        h.pt3dadd(15, 30, 0, 1, sec=self.radTprox)

        h.pt3dclear(sec=self.radTmed)
        h.pt3dadd(15, 30, 0, 1, sec=self.radTmed)
        h.pt3dadd(15, 60, 0, 1, sec=self.radTmed)
     
        h.pt3dclear(sec=self.radTdist)
        h.pt3dadd(15, 60, 0, 1, sec=self.radTdist)
        h.pt3dadd(15, 90, 0, 1, sec=self.radTdist)

        h.pt3dclear(sec=self.lm_thick2)
        h.pt3dadd(15, 90, 0, 1, sec=self.lm_thick2)
        h.pt3dadd(45, 105, 0, 1, sec=self.lm_thick2)

        h.pt3dclear(sec=self.lm_medium2)
        h.pt3dadd(45, 105, 0, 1, sec=self.lm_medium2)
        h.pt3dadd(75, 120, 0, 1, sec=self.lm_medium2)

        h.pt3dclear(sec=self.lm_thin2)
        h.pt3dadd(75, 120, 0, 1, sec=self.lm_thin2)
        h.pt3dadd(105, 135, 0, 1, sec=self.lm_thin2)
        
        h.pt3dclear(sec=self.lm_thick1)
        h.pt3dadd(15, 90, 0, 1, sec=self.lm_thick1)
        h.pt3dadd(-14, 105, 0, 1, sec=self.lm_thick1)

        h.pt3dclear(sec=self.lm_medium1)
        h.pt3dadd(-14, 105, 0, 1, sec=self.lm_medium1)
        h.pt3dadd(-44, 120, 0, 1, sec=self.lm_medium1)
  
        h.pt3dclear(sec=self.lm_thin1)
        h.pt3dadd(-44, 120, 0, 1, sec=self.lm_thin1)
        h.pt3dadd(-89, 135, 0, 1, sec=self.lm_thin1)
   
        h.pt3dclear(sec=self.oriprox1)
        h.pt3dadd(0, 0, 0, 1, sec=self.oriprox1)
        h.pt3dadd(-44, -29, 0, 1, sec=self.oriprox1)

        h.pt3dclear(sec=self.oridist1)
        h.pt3dadd(-44, -29, 0, 1, sec=self.oridist1)
        h.pt3dadd(-74, -59, 0, 1, sec=self.oridist1)

        h.pt3dclear(sec=self.oriprox2)
        h.pt3dadd(15, 0, 0, 1, sec=self.oriprox2)
        h.pt3dadd(60, -29, 0, 1, sec=self.oriprox2)

        h.pt3dclear(sec=self.oridist2)
        h.pt3dadd(60, -29, 0, 1, sec=self.oridist2)
        h.pt3dadd(105, -59, 0, 1, sec=self.oridist2)

        h.pt3dclear(sec=self.axon)
        h.pt3dadd(15, 0, 0, 1, sec=self.axon)
        h.pt3dadd(15, -149, 0, 1, sec=self.axon)

    def subsets(self):
        self.all = h.SectionList()
        self.all.wholetree(sec=self.soma)

    def geom(self):
        self.soma.L = 10
        self.soma.diam = 10

        self.radTprox.L = 100
        self.radTprox.diam = 4

        self.radTmed.L = 100
        self.radTmed.diam = 3

        self.radTdist.L = 200
        self.radTdist.diam = 2

        self.lm_thick2.L = 100
        self.lm_thick2.diam = 2
 
        self.lm_medium2.L = 100
        self.lm_medium2.diam = 1.5

        self.lm_thin2.L = 50
        self.lm_thin2.diam = 1

        self.lm_thick1.L = 100
        self.lm_thick1.diam = 2

        self.lm_medium1.L = 100
        self.lm_medium1.diam = 1.5
 
        self.lm_thin1.L = 50
        self.lm_thin1.diam = 1

        self.oriprox1.L = 100
        self.oriprox1.diam = 2
 
        self.oridist1.L = 200
        self.oridist1.diam = 1.5
 
        self.oriprox2.L = 100
        self.oriprox2.diam = 2

        self.oridist2.L = 200
        self.oridist2.diam = 1.5

        self.axon.L = 150
        self.axon.diam = 1

    def biophys(self):
        Rm = 20000
        gka_soma = 0.0075
        gh_soma = 0.00005
        self.soma.insert('hha2')
        for seg in self.soma:
            seg.hha2.gnabar = 0.007
            seg.hha2.gkbar = 0.007/5
            seg.hha2.gl = 0
            seg.hha2.el = -70
        self.soma.insert('pas')
        for seg in self.soma:
            seg.pas.g = 1. / Rm
        self.soma.insert('h2')
        for seg in self.soma:
            seg.h2.ghdbar = gh_soma
            seg.h2.vhalfl = -73
        #self.soma.insert('hNa')
        #for seg in self.soma:
        #    seg.hNa.gbar = 0.000043
        #    seg.hNa.gbar = 1.87e-5
        self.soma.insert('kap')
        for seg in self.soma:
            seg.kap.gkabar = gka_soma
        self.soma.insert('km')
        for seg in self.soma:
            seg.km.gbar = 0.06
        self.soma.insert('cal')
        for seg in self.soma:
            seg.cal.gcalbar = 0.0014/2
        self.soma.insert('cat')
        for seg in self.soma:
            seg.cat.gcatbar = 0.0001/2
        self.soma.insert('somacar')
        for seg in self.soma:
            seg.somacar.gcabar = 0.0003
        self.soma.insert('kca')
        for seg in self.soma:
            seg.kca.gbar = 5 * 0.0001
        self.soma.insert('mykca')
        for seg in self.soma:
            seg.mykca.gkbar = 0.09075
        self.soma.insert('cad')

        self.radTprox.insert('h2')
        for seg in self.radTprox:
            seg.h2.ghdbar = 2*gh_soma
            seg.h2.vhalfl = -81
        self.radTprox.insert('car')
        for seg in self.radTprox:
            seg.car.gcabar = 0.1 * 0.0003
 
        self.radTprox.insert('calH')
        for seg in self.radTprox:
            seg.calH.gcalbar = 0.1 * 0.00031635
        
        self.radTprox.insert('cat')
        for seg in self.radTprox:
            seg.cat.gcatbar = 0.0001

        self.radTprox.insert('cad')
        self.radTprox.insert('kca')
        for seg in self.radTprox:
            seg.kca.gbar = 5 * 0.0001

        self.radTprox.insert('mykca')
        for seg in self.radTprox:
            seg.mykca.gkbar = 2 * 0.0165

        self.radTprox.insert('km')
        for seg in self.radTprox:
            seg.km.gbar = 0.06

        self.radTprox.insert('kap')
        for seg in self.radTprox:
            seg.kap.gkabar = 2 * gka_soma

        self.radTprox.insert('kad')
        for seg in self.radTprox:
            seg.kad.gkabar = 0.0

        self.radTprox.insert('hha_old')
        for sseg in self.radTprox:
            seg.hha_old.gnabar = 0.007
            seg.hha_old.gkbar = 0.007 / 8.065
            seg.hha_old.el = -70

        self.radTprox.insert('pas')

        self.radTmed.insert('h2')
        for seg in self.radTmed:
            seg.h2.ghdbar = 4 * gh_soma
            seg.h2.vhalfl = -81
        self.radTmed.insert('car')
        for seg in self.radTmed:
            seg.car.gcabar = 0.1 * 0.0003
 
        self.radTmed.insert('calH')
        for seg in self.radTmed:
            seg.calH.gcalbar = 10 * 0.00031635
    
        self.radTmed.insert('cat')
        for seg in self.radTmed:
            seg.cat.gcatbar = .0001
            
        self.radTmed.insert('cad')
        self.radTmed.insert('kca')
        for seg in self.radTmed:
            seg.kca.gbar = 5 * 0.0001
        self.radTmed.insert('mykca')
        for seg in self.radTmed:
            seg.mykca.gkbar = 2 * 0.0165
        self.radTmed.insert('km')
        for seg in self.radTmed:
            seg.km.gbar = 0.06
        self.radTmed.insert('kap')
        for seg in self.radTmed:
            seg.kap.gkabar = 0.0
        self.radTmed.insert('kad')
        for seg in self.radTmed:
            seg.kad.gkabar = 4 * gka_soma
        self.radTmed.insert('hha_old')
        for seg in self.radTmed:
            seg.hha_old.gnabar = 0.007
            seg.hha_old.gkbar = 0.007 / 8.065
            seg.hha_old.el = -70
        self.radTmed.insert('pas')

        self.radTdist.insert('h2')
        for seg in self.radTdist:
            seg.h2.ghdbar = 7 * gh_soma
            seg.h2.vhalfl = -81

        self.radTdist.insert('car')
        for seg in self.radTdist:
            seg.car.gcabar = 0.1 * 0.0003
        self.radTdist.insert('calH')
        for seg in self.radTdist:
            seg.calH.gcalbar = 10 * 0.00031635
        self.radTdist.insert('cat')
        for seg in self.radTdist:
            seg.cat.gcatbar = 0.0001
        self.radTdist.insert('cad')
        self.radTdist.insert('kca')
        for seg in self.radTdist:
            seg.kca.gbar = 0.5 * 0.0001
        self.radTdist.insert('mykca')
        for seg in self.radTdist:
            seg.mykca.gkbar = 0.25 * 0.0165
        self.radTdist.insert('km')
        for seg in self.radTdist:
            seg.km.gbar = 0.06
        self.radTdist.insert('kap')
        for seg in self.radTdist:
            seg.kap.gkabar = 0.0
        self.radTdist.insert('kad')
        for seg in self.radTdist:
            seg.kad.gkabar = 6 * gka_soma
        self.radTdist.insert('hha_old')
        for seg in self.radTdist:
            seg.hha_old.gnabar = 0.007
            seg.hha_old.gkbar = 0.007 / 8.065
            seg.hha_old.el = -70
        self.radTdist.insert('pas')

        self.lm_thick2.insert('hha_old')
        for seg in self.lm_thick2:
            seg.hha_old.gnabar = 0.007
            seg.hha_old.gkbar = 0.007 / 8.065
            seg.hha_old.el = -70
            seg.hha_old.gl = 0
        self.lm_thick2.insert('pas')
        for seg in self.lm_thick2:
            seg.pas.g = 1 / 200000
        self.lm_thick2.insert('kad')
        for seg in self.lm_thick2:
            seg.kad.gkabar = 6.5 * gka_soma

        self.lm_medium2.insert('hha_old')
        for seg in self.lm_medium2:
            seg.hha_old.gnabar = 0.007
            seg.hha_old.gkbar = 0.007 / 8.065
            seg.hha_old.el = -70
            seg.hha_old.gl = 0
        self.lm_medium2.insert('pas')
        for seg in self.lm_medium2:
            seg.pas.g = 1./200000
        self.lm_medium2.insert('kad')
        for seg in self.lm_medium2:
            seg.kad.gkabar = 6.5 * gka_soma
        
        
        self.lm_thin2.insert('hha_old')
        for seg in self.lm_thin2:
            seg.hha_old.gnabar = 0.007
            seg.hha_old.gkbar = 0.007 / 8.065
            seg.hha_old.el = -70
            seg.hha_old.gl = 0
        self.lm_thin2.insert('pas')
        for seg in self.lm_thin2:
            seg.pas.g = 1./200000
        self.lm_thin2.insert('kad')
        for seg in self.lm_thin2:
            seg.kad.gkabar = 6.5 * gka_soma


        self.lm_thick1.insert('hha_old')
        for seg in self.lm_thick1:
            seg.hha_old.gnabar = 0.007
            seg.hha_old.gkbar = 0.007 / 8.065
            seg.hha_old.el = -70
            seg.hha_old.gl = 0
        self.lm_thick1.insert('pas')
        for seg in self.lm_thick1:
            seg.pas.g = 1 / 200000
        self.lm_thick1.insert('kad')
        for seg in self.lm_thick1:
            seg.kad.gkabar = 6.5 * gka_soma

        self.lm_medium1.insert('hha_old')
        for seg in self.lm_medium1:
            seg.hha_old.gnabar = 0.007
            seg.hha_old.gkbar = 0.007 / 8.065
            seg.hha_old.el = -70
            seg.hha_old.gl = 0
        self.lm_medium1.insert('pas')
        for seg in self.lm_medium1:
            seg.pas.g = 1./200000
        self.lm_medium1.insert('kad')
        for seg in self.lm_medium1:
            seg.kad.gkabar = 6.5 * gka_soma
        
        self.lm_thin1.insert('hha_old')
        for seg in self.lm_thin1:
            seg.hha_old.gnabar = 0.007
            seg.hha_old.gkbar = 0.007 / 8.065
            seg.hha_old.el = -70
            seg.hha_old.gl = 0
        self.lm_thin1.insert('pas')
        for seg in self.lm_thin1:
            seg.pas.g = 1./200000
        self.lm_thin1.insert('kad')
        for seg in self.lm_thin1:
            seg.kad.gkabar = 6.5 * gka_soma

        self.oriprox1.insert('h2')
        for seg in self.oriprox1:
            seg.h2.ghdbar = gh_soma
            seg.h2.vhalfl = -81
        self.oriprox1.insert('car')
        for seg in self.oriprox1:
            seg.car.gcabar = 0.1 * 0.0003
        self.oriprox1.insert('calH')
        for seg in self.oriprox1:
            seg.calH.gcalbar = 0.1 * 0.00031635
        self.oriprox1.insert('cat')
        for seg in self.oriprox1:
            seg.cat.gcatbar = 0.0001
        self.oriprox1.insert('cad')
        self.oriprox1.insert('kca')
        for seg in self.oriprox1:
            seg.kca.gbar = 5.0 * 0.0001
        self.oriprox1.insert('mykca')
        for seg in self.oriprox1:
            seg.mykca.gkbar = 2 * 0.0165
        self.oriprox1.insert('km')
        for seg in self.oriprox1:
            seg.km.gbar = 0.06
        self.oriprox1.insert('kap')
        for seg in self.oriprox1:
            seg.kap.gkabar = gka_soma
        self.oriprox1.insert('kad')
        for seg in self.oriprox1:
            seg.kad.gkabar = 0.0
        self.oriprox1.insert('hha_old')
        for seg in self.oriprox1:
            seg.hha_old.gnabar = 0.007
            seg.hha_old.gkbar = 0.007 / 8.065
            seg.hha_old.el = -70
        self.oriprox1.insert('pas')

        self.oridist1.insert('h2')
        for seg in self.oridist1:
            seg.h2.ghdbar = 2 * gh_soma
            seg.h2.vhalfl = -81
        self.oridist1.insert('car')
        for seg in self.oridist1:
            seg.car.gcabar = 0.1 * 0.0003
        self.oridist1.insert('calH')
        for seg in self.oridist1:
            seg.calH.gcalbar = 0.1 * 0.00031635
        self.oridist1.insert('cat')
        for seg in self.oridist1:
            seg.cat.gcatbar = 0.0001
        self.oridist1.insert('cad')
        self.oridist1.insert('kca')
        for seg in self.oridist1:
            seg.kca.gbar = 5 * 0.0001
        self.oridist1.insert('mykca')
        for seg in self.oridist1:
            seg.mykca.gkbar = 2 * 0.0165
        self.oridist1.insert('km')
        for seg in self.oridist1:
            seg.km.gbar = 0.06
        self.oridist1.insert('kap')
        for seg in self.oridist1:
            seg.kap.gkabar = gka_soma
        self.oridist1.insert('kad')
        for seg in self.oridist1:
            seg.kad.gkabar = 0.0
        self.oridist1.insert('hha_old')
        for seg in self.oridist1:
            seg.hha_old.gnabar = 0.007
            seg.hha_old.gkbar = 0.007 / 8.065
            seg.hha_old.el = -70
        self.oridist1.insert('pas')
        
        self.oriprox2.insert('h2')
        for seg in self.oriprox2:
            seg.h2.ghdbar = gh_soma
            seg.h2.vhalfl = -81
        self.oriprox2.insert('car')
        for seg in self.oriprox2:
            seg.car.gcabar = 0.1 * 0.0003
        self.oriprox2.insert('calH')
        for seg in self.oriprox2:
            seg.calH.gcalbar = 0.1 * 0.00031635
        self.oriprox2.insert('cat')
        for seg in self.oriprox2:
            seg.cat.gcatbar = 0.0001
        self.oriprox2.insert('cad')
        self.oriprox2.insert('kca')
        for seg in self.oriprox2:
            seg.kca.gbar = 5.0 * 0.0001
        self.oriprox2.insert('mykca')
        for seg in self.oriprox2:
            seg.mykca.gkbar = 2 * 0.0165
        self.oriprox2.insert('km')
        for seg in self.oriprox2:
            seg.km.gbar = 0.06
        self.oriprox2.insert('kap')
        for seg in self.oriprox2:
            seg.kap.gkabar = gka_soma
        self.oriprox2.insert('kad')
        for seg in self.oriprox2:
            seg.kad.gkabar = 0.0
        self.oriprox2.insert('hha_old')
        for seg in self.oriprox2:
            seg.hha_old.gnabar = 0.007
            seg.hha_old.gkbar = 0.007 / 8.065
            seg.hha_old.el = -70
        self.oriprox2.insert('pas')
    
        self.oridist2.insert('h2')
        for seg in self.oridist2:
            seg.h2.ghdbar = 2 * gh_soma
            seg.h2.vhalfl = -81
        self.oridist2.insert('car')
        for seg in self.oridist2:
            seg.car.gcabar = 0.1 * 0.0003
        self.oridist2.insert('calH')
        for seg in self.oridist2:
            seg.calH.gcalbar = 0.1 * 0.00031635
        self.oridist2.insert('cat')
        for seg in self.oridist2:
            seg.cat.gcatbar = 0.0001
        self.oridist2.insert('cad')
        self.oridist2.insert('kca')
        for seg in self.oridist2:
            seg.kca.gbar = 5 * 0.0001
        self.oridist2.insert('mykca')
        for seg in self.oridist2:
            seg.mykca.gkbar = 2 * 0.0165
        self.oridist2.insert('km')
        for seg in self.oridist2:
            seg.km.gbar = 0.06
        self.oridist2.insert('kap')
        for seg in self.oridist2:
            seg.kap.gkabar = gka_soma
        self.oridist2.insert('kad')
        for seg in self.oridist2:
            seg.kad.gkabar = 0.0
        self.oridist2.insert('hha_old')
        for seg in self.oridist2:
            seg.hha_old.gnabar = 0.007
            seg.hha_old.gkbar = 0.007 / 8.065
            seg.hha_old.el = -70
        self.oridist2.insert('pas')

        self.axon.insert('hha2')
        for seg in self.axon:
            seg.hha2.gnabar = 0.1
            seg.hha2.gkbar = .1 / 5
            seg.hha2.gl = 0
            seg.hha2.el = -70
        self.axon.insert('pas')
        for seg in self.axon:
            seg.pas.g = 1 / Rm
        self.axon.insert('km')
        for seg in self.axon:
            seg.km.gbar = 0.5 * 0.06

        for sec in self.all:
            sec.ek = -80
            sec.ena = 50
            sec.Ra = 150
            sec.cm = 1
            for seg in sec:
                seg.pas.e = -70.
                seg.pas.g = 1 / Rm

    def geom_nseg(self):
        lambda_f = h.lambda_f
        for seg in self.all:
            seg.nseg = int((seg.L/(0.1*lambda_f(100))+0.9)/2)*2+1

    def generate_synapses(self):
        self._make_syn_groups()

#     self.all        = None
#     self.soma       = None
#     self.radTprox   = None
#     self.radTmed    = None
#     self.radTdist   = None
#     self.lm_thick2  = None
#     self.lm_medium2 = None
#     self.lm_thin2   = None
#     self.lm_thick1  = None
#     self.lm_medium1 = None
#     self.lm_thin1   = None
#     self.oriprox1   = None
#     self.oridist1   = None
#     self.oriprox2   = None
#     self.oridist2   = None
#     self.axon       = None
    def _make_syn_groups(self):
        region_list = ['axon', 'soma', 'radTprox', 'radTmed', 'radTdist', 'lm_thick2', 'lm_medium2', 'lm_thin2', 'lm_thick1',
                      'lm_medium1', 'lm_thin1', 'oriprox1', 'oridist1', 'oriprox2', 'oridist2']
        self.synGroups = {} 
        
        self.synGroups['AMPA'] = {}
        for roi in region_list:
            self.synGroups['AMPA'][roi] = {}
                    
        self.synGroups['GABAA'] = {}
        for roi in region_list:
            self.synGroups['GABAA'][roi] = {}
            
        self.synGroups['GABAB'] = {}
        for roi in region_list:
            self.synGroups['GABAB'][roi] = {}
            
        self.synGroups['NMDA'] = {}
        for roi in region_list:
            self.synGroups['NMDA'][roi] = {}
            
        self.synGroups['STDPE2'] = {}
        for roi in region_list:
            self.synGroups['STDPE2'][roi] = {}
                      
        self.synGroups['STDPE2ASYM'] = {}
        for roi in region_list:
            self.synGroups['STDPE2ASYM'][roi] = {}  
            
            
    def get_syn_parameters(self, sec_choice, syn_type):
        params = {}
        if syn_type == 'NMDA':
            tau1 = 2.3
            tau2 = 100.0
            gNMDAmax = 1.0
            params = {'tcon': tau1, 'tcoff': tau2, 'gNMDAmax': gNMDAmax}
        elif syn_type == 'AMPA' or syn_type == 'STDPE2' or syn_type == 'STDPE2ASYM':
            tau1, tau2 = None, None
            e = 0.0
            tau1 = 0.5
            tau2 = 3.0
            params = {'e': e, 'tau1': tau1, 'tau2': tau2}
        return params
