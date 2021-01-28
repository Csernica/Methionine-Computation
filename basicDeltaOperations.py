import numpy as np
import pandas as pd

##############################################################################################################################
###                                                                                                                        ###
###   This code takes care of basic manipulations between delta, ratio, and concentration space. These are three related   ###
###   but distinct ways of discussing the isotopic content at an atom.                                                     ###
###                                                                                                                        ###
###   For details, see lecture 3: http://web.gps.caltech.edu/classes/ge140a/Stable_Isotope_W19/Lecture.html                ###
###                                                                                                                        ###
##############################################################################################################################

#DEFINE STANDARDS
#Doubly isotopic atoms are given as standard ratios, i.e. PDB for carbon
STD_Rs = {"H": 0.00015576, "C": 0.0112372, "N": 0.003676, "17O": 0.0003799, "18O": 0.0020052,
         "33S":0.007895568,"34S":0.044741552,"36S":0.000105274}

def deltaToConcentration(atomIdentity,delta):
    '''
    Converts an input delta value for a given type of atom in some reference frame to a 4-tuple containing the 
    concentration of the unsubstituted, M+1, M+2, and all other versions of the atom.
    
    Inputs:
        atomIdentity: A string giving the isotope of interest
        delta: The input delta value. This must be in the VSMOW, PDB, or AIR standards for D, 13C, and 15N, respectively
        
    Outputs:
        The ratio for this delta value.
    '''
    if atomIdentity in 'HCN' or atomIdentity in ['D','13C','15N']:
        ratio = (delta/1000+1)*STD_Rs[atomIdentity]
        concentrationSub = ratio/(1+ratio)
        
        return (1-concentrationSub,concentrationSub,0,0)
    
    elif atomIdentity == '17O' or atomIdentity == 'O':
        r17 = (delta/1000+1)*STD_Rs['17O']
        delta18 = 1/0.52 * delta
        r18 = (delta18/1000+1)*STD_Rs['18O']
        
        o17 = r17/(1+r17+r18)
        o18 = r18/(1+r17+r18)
        o16 = 1-(o17+o18)
        
        return (o16,o17,o18,0)
    
    elif atomIdentity == '33S' or atomIdentity == 'S':
        r33 = (delta/1000+1)*STD_Rs['33S']
        delta34 = delta/0.515
        r34 = (delta34/1000+1)*STD_Rs['34S']
        delta36 = delta34*1.9
        r36 = (delta36/1000+1)*STD_Rs['36S']
        
        s33 = r33/(1+r33 + r34 + r36)
        s34 = r34/(1+r33+r34+r36)
        s36 = r36/(1+r33+r34+r36)

        s32 = 1-(s33+s34+s36)
        
        return (s32,s33,s34,s36)
                
    else:
        raise Exception('Sorry, I do not know how to deal with ' + atomIdentity)
    
def hFunction(atomIdentity, M1Value):
    '''
    Given the concentration of a site-specific M1 substitution (13C, 15N, 17O, 33S, D) determines the concentration of the 
    unsubstituted version (12C, 14N, 16O, 32S, H). For simple cases, this is 1-[13C]. For elements with 3+ isotopes, it 
    must include some information about the 18O, 34S, etc. This has been derived using the scaling laws in 
    deltaToConcentration. If we change the scaling laws, this must be changed as well. 
    
    Inputs:
        atomIdentity: A string giving the chemical element of interest
        M1Value: The concentration of the M1 substitution
        
    Outputs:
        unsub: The concentration of the unsubstituted isotope.
    '''
    if atomIdentity in 'HCN':
        return 1-M1Value
    
    #could condense this if necessary. 1.61 us right now. 
    elif atomIdentity == 'O':
        R17S = STD_Rs['17O']
        R18S = STD_Rs['18O']
        
        alpha = R17S / (0.52 * R18S)
        beta = R17S * 0.48 / 0.52
        
        unsub = ((-alpha -1) * M1Value + alpha) / (alpha + beta)
        
        return unsub

def concentrationToRatio(concentrationTuple):
    '''
    Gives the M1 ratio of an atom based on concentration
    '''
    return concentrationTuple[1]/concentrationTuple[0] 

def ratioToDelta(atomIdentity, ratio):
    '''
    Converts an input ratio for a given atom to a delta value.
    
    Inputs:
        atomIdentity: A string giving the isotope of interest
        ratio: The isotope ratio 
        
    outputs: 
        delta: The delta value for that isotope and ratio
    '''
    delta = 0
    if atomIdentity == 'D':
        atomIdentity = 'H'
        
    if atomIdentity in 'HCN' or atomIdentity in ['13C','15N']:
        #in case atomIdentity is 2H, 13C, 15N, take last character only
        delta = (ratio/STD_Rs[atomIdentity[-1]]-1)*1000
        
    elif atomIdentity == 'O' or atomIdentity == '17O':
        delta = (ratio/STD_Rs['17O']-1)*1000
        
    elif atomIdentity == '18O':
        delta = (ratio/STD_Rs['18O']-1)*1000
        
    elif atomIdentity == 'S' or atomIdentity == '33S':
        delta = (ratio/STD_Rs['33S']-1)*1000
        
    elif atomIdentity == '34S':
        delta = (ratio/STD_Rs['34S']-1)*1000
        
    elif atomIdentity == '36S':
        delta = (ratio/STD_Rs['36S']-1)*1000
        
    else:
        raise Exception('Sorry, I do not know how to deal with ' + atomIdentity)
        
    return delta