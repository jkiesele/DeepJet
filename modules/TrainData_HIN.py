'''
Created on 21 Feb 2017

@author: jkiesele
'''
from TrainData import TrainData_Flavour, TrainData_simpleTruth



class TrainData_HIN(TrainData_Flavour, TrainData_simpleTruth):
    '''
    same as TrainData_deepCSV but with 4 truth labels: B BB C UDSG
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour.__init__(self)
        
        #unusual addtions because of reduced truth
        self.undefTruth=['isUnmatched']
        
        self.truthclasses=['isB','iC','isUDSG','isUnmatched']
        
        self.allbranchestoberead=[]
        self.registerBranches(self.undefTruth)
        self.registerBranches(self.truthclasses)
        self.registerBranches(['jet_pt','jet_eta'])
        
        
        
        self.addBranches(['jet_pt', 'jet_eta','ntk','hibin','trackSumJetDeltaR',
                          'trackSip2dSigAboveCharm','trackSip3dSigAboveCharm',
                          'trackSip2dValAboveCharm','trackSip3dValAboveCharm'])
       
        self.addBranches(['tkptrel',
                          'tkdr',
                          'tkip2d',
                          'tkip2dsig',
                          'tkip3d',
                          'tkipd3dsig',
                          'tkipdist2j',
                          'tkptratio',
                          'tkpparratio'],
                             1)#for now... (could be up to 70)
        
       
        
        

        self.addBranches(['svm', 
                              'svntrk', 
                              'svdl',
                              'svdls',
                              'svdl2d', 
                              'svdls2d', 
                              'svpt', 
                              'sve2e', 
                              'svdr2jet', 
                              'svptrel',  
                              'svtksumchi2',  
                              'svcharge',  
                              'svptrel',  
                              'svtkincone',  
                              'svmcorr'],
                             1)


        
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("jets")
        self.nsamples=tree.GetEntries()
        
        print('took ', sw.getAndReset(), ' seconds for getting tree entries')
        
        
        # split for convolutional network
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        #x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
        #                           self.branches[2],
        #                           self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            undef=Tuple['isUndefined']
            notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')
        
        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            #x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_cpf,x_sv]
        self.y=[alltruth]
    
