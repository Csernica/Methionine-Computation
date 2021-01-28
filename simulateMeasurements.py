import numpy as np
import pandas as pd

import basicDeltaOperations as op
import calcIsotopologues as ci

##############################################################################################################################
###                                                                                                                        ###
###   This code allows extracts the concentrations of isotopologues of interest from the dictionary of all isotopologues   ###
###   in order to predict the outcomes of meaurements. It also allows one to fragment the isotopologues to compute the     ###
###   outcome of fragment measurements.                                                                                    ###
###                                                                                                                        ###
###   The theory for this section is developed in the working M+N paper. Contact Tim for details                           ###
###                                                                                                                        ###
###   It assumes one has access to a dictionary with information about the isotopologues. See calcIsotopologues.py.        ###
###                                                                                                                        ###
##############################################################################################################################

#Gives an easy way to recover an isotope from an element and its cardinal mass representation. 
subDict = {'C':{'0':'','1':'13C'},
           'N':{'0':'','1':'15N'},
           'H':{'0':'','1':'D'},
           'O':{'0':'','1':'17O','2':'18O'},
           'S':{'0':'','1':'33S','2':'34S','4':'36S'}}

#An easy way to recover the mass of an isotope from an element and its cardinal mass representation
massDict = {'C':{'0':12,'1':13.00335484},
            'N':{'0':14.003074,'1':15.00010889},
            'H':{'0':1.007825032,'1':2.014101778},
            'O':{'0':15.99491462,'1':16.99913175,'2':17.9991596},
            'S':{'0':31.97207117,'1':32.9714589,'2':33.96786701,'4':35.9670807}}

def directMeasurement(bySub, allMeasurementInfo, massThreshold = 3):
    '''
    Simulates measurements with no fragmentation. Extracts the concentration of all isotopologues with mass below some threshold for easy reference. 
    
    Inputs:
        bySub: A dictionary with information about all isotopologues of a molecule, sorted by substitution. 
        allMeasurementInfo: A dictionary containing information from many types of measurements. 
        massThreshold: A mass cutoff; isotopologues with cardinal mass change above this will not be included.
        
    Outputs:
        allMeasurementInfo: A dictionary, updated to include information from direct measurements.
    '''
    if 'Full' not in allMeasurementInfo:
        allMeasurementInfo['Full'] = {}
        
    for sub, info in bySub.items():
    
        if info['Mass'][0] <= massThreshold:
            allMeasurementInfo['Full'][sub] = info['Conc']
            
    return allMeasurementInfo

def fragMult(z, y):
    '''
    Fragments an individual site of an isotopologue. z should be 1 or 'x'. 
    '''
    if z == 'x' or y == 'x':
        return 'x'
    else:
        return z*y
    
def expandFrag(multiAtomicFrag, number):
    '''
    Creates a "by position" depiction of a fragment from a "by site" depiction from a fragment. Expands the fragmentation vector according to the number of atoms at each site. For example, if I fragment [0,(0,1)] with fragmentation vector [0,1], I do so by applying the fragmentation vector [011] to the isotopologue [001], expanding the tuple. This function expands the fragmentation vector. 
    
    Inputs:
        multiAtomicFrag: condensed depiction of fragmentation vector.
        
    Outputs:
        expanded: expanded depiction of fragmentation vector
    '''
    expanded = []
    for i, v in enumerate(multiAtomicFrag):
        expanded += [v] * number[i]
    
    return expanded

def fragmentOneIsotopologue(expandedFrag, isotopologue):
    '''
    Applies the "by position" fragmentation vector to a condensed depiction of an isotopologue. Raises a warning if they are not the same length. Returns the condensed depiction of the isotopologue with "x" in positions that are lost.
    '''
    #important to raise this--otherwise one may inadvertantly fragment incorrectly. 
    if len(expandedFrag) != len(isotopologue):
           raise Exception("Cannot fragment successfully, as the fragment and the isotopologue you want to fragment have different lengths")
            
    a = [fragMult(x,y) for x, y in zip(expandedFrag, isotopologue)]
    
    if len(a) != len(isotopologue):
        raise Exception("Cannot fragment successfully, the resulting fragment has a different length than the input isotopologue.")
    
    return ''.join(a)

def fragmentIsotopologueDict(condensedIsotopologueDict, fragment):
    '''
    Applies the same fragmentation vector to all isotopologues of an input isotopologue dict and stores the results. This operation corresponds to the "fragmentation" operation from the M+N paper. 
    
    Inputs:
        condensedIsotopologueDict: A dictionary containing some set of isotopologues, often a M1, M2, ... set, keyed by their condensed depiction. 
        fragment: An expanded fragment, with indices corresponding to the condensed depiction of the isotopologue. 
        
    Outputs: 
        fragmentedDict: A dictionary where the keys are the condensed isotopologues after fragmentation (i.e. "0000x") and the values are the concentrations of those isotopologues. Note that this may combine isotopologues from the input dictionary which fragment in the same way; i.e. 001 and 002 both fragment to yield "00x". 
    '''
    
    fragmentedDict = {}
    for isotopologue, value in condensedIsotopologueDict.items():
        newIsotopologue = fragmentOneIsotopologue(fragment, isotopologue)
        if newIsotopologue not in fragmentedDict:
            fragmentedDict[newIsotopologue] = 0
        fragmentedDict[newIsotopologue] += value['Conc']
        
    return fragmentedDict
    
def computeSubs(isotopologue, IDs):
    '''
    Given a condensed depiction of an isotopologue, computes which substitutions are present. 
    
    Inputs:
        isotopologue: The condensed string depiction of an isotopologue
        IDs: The string of site elements, i.e. the output of strSiteElements
        
    Outputs:
        A string giving substitutions present in that isotopologue, separated by "-". I.e. "17O-17O"
    '''
    subs = []
    for i in range(len(isotopologue)):
        if isotopologue[i] != 'x':
            element = IDs[i]
            if subDict[element][str(isotopologue[i])] != '':
                subs.append(subDict[element][str(isotopologue[i])])
        
    return '-'.join(subs)

def predictMNFragmentExpt(allMeasurementInfo, MNDict, expandedFragList, fragKeys, df, abundanceThreshold = 0):
    '''
    A kind of 'do it all' function that predicts the results of several M+N experiements across a range of mass selected populations and fragments. It incorporates the preceding functions into a whole, so you can just call this and get results.
    
    Inputs:
        allMeasurementInfo: A dictionary containing information from many types of measurements. 
        MNDict: A dictionary where the keys are "M0", "M1", etc. and the values are dictionaries containing all isotopologues from the condensed dictionary with a specified cardinal mass difference. See massSelections function in calcIsotopologues.py
        expandedFragList: A list of expanded fragments, i.e. [[1, 1, 1, 1, 'x'], ['x', 1, 1, 1, 'x']]. See expandFrags function.
        fragKeys: A list of strings, indicating the identity of each fragment. I.e. ['54','42']
        df: A dataFrame containing information about the molecule.
        abundanceThreshold: Does not include measurements below a certain relative abundance, i.e. assuming they will not be successfully measured due to low abundance. 
        
    Outputs: 
        allMeasurementInfo: A dictionary, updated to include information from the M+N measurements. 
    '''
    siteElements = ci.strSiteElements(df)
    
    #For each population (M1, M2, M3) that we mass select
    for massSelection, MN in MNDict.items():
        #add a key to output dictionary
        if massSelection not in allMeasurementInfo:
            allMeasurementInfo[massSelection] = {}

        #For each fragment we will observe
        for j, fragment in enumerate(expandedFragList):

            #add a key to output dictionary
            if fragKeys[j] not in allMeasurementInfo[massSelection]:
                allMeasurementInfo[massSelection][fragKeys[j]] = {}

            #fragment the mass selection accordingly 
            fragmentedIsotopologues = fragmentIsotopologueDict(MN, fragment)

            #compute the absolute abundance of each substitution
            predictSpectrum = {}

            for key, item in fragmentedIsotopologues.items():
                subs = computeSubs(key, siteElements)

                if subs not in predictSpectrum:
                    predictSpectrum[subs] = {'Abs. Abundance':0}

                predictSpectrum[subs]['Abs. Abundance'] += item

            #compute relative abundance of each substitution
            totalAbundance = 0
            for key, item in predictSpectrum.items():
                totalAbundance += item['Abs. Abundance']

            for key, item in predictSpectrum.items():
                item['Rel. Abundance'] = item['Abs. Abundance'] / totalAbundance

            #cut off entries with relative abundance below some threshold
            shortSpectrum = {}
            totalAdjAbund = 0
            for x, v in predictSpectrum.items():
                if v['Rel. Abundance'] > abundanceThreshold:
                    shortSpectrum[x] = v
                    totalAdjAbund += v['Abs. Abundance']
                    
            for x, v in shortSpectrum.items():
                v['Adj. Rel. Abundance'] = v['Abs. Abundance'] / totalAdjAbund
                

            allMeasurementInfo[massSelection][fragKeys[j]] = shortSpectrum
            
    return allMeasurementInfo
        
def measurementToDf(allMeasurementInfo, bySub, fullMoleculeU = [], ratios = []):
    '''
    Takes relevant information from allMeasurementInfo and outputs it as a pandas dataframe. 
    
    Inputs:
        allMeasurementInfo: A dictionary containing information from many types of measurements. 
        bySub: A dictionary giving information about all isotopologues of a molecule, indexed by substituion. 
        fullMolecule: A list of full molecule U values to include. E.g. ['18O','13C','18O18O']
        ratios: A list of ratios between ion beams to measure from the parent population without fragmentation. E.g. ['D/34S','17O/34S','33S/34S','15N/34S','13C13C/34S',
         '18O/34S','13C33S/34S']
        
    Outputs:
        out: a pandas dataFrame including the desired information
    '''
    outputDict = {}
    ###Add full molecule U
    for key in fullMoleculeU:
        v = UValueBySubFromSub(bySub, key)
        outputDict[key + " U Value"] = v

    ###Add M+N Experiments
    MN = False
    for key, value in allMeasurementInfo.items():
        if key[0] == 'M':
            MN = True
            for fragment, data in value.items():
                for sub, abundance in data.items():
                    outputDict[key + " " + fragment + " " + sub] = abundance
                    
    ###Add ratios
    for UValue in ratios:
        num, den = UValue.split('/')
        fraction = allMeasurementInfo['Full'][num] / allMeasurementInfo['Full'][den]
        outputDict[UValue] = fraction
    
    if MN == False:
        out = pd.Series(outputDict)
    
    else:
        out = pd.DataFrame.from_dict(outputDict).T
    
    return out

def UValueBySubFromSub(dictionary, sub):
    '''
    Takes the "By Sub" dictionary type.    
    '''
    den = dictionary['']['Conc']
    num = dictionary[sub]['Conc']
    
    return num/den
    
    
def UValueBySubFromCondensed(dictionary, sub):
    '''
    inefficient but user-friendly. Takes the "By Condensed" dictionary type. 
    '''
    den = dictionary['000000000000000000000']['Conc']
    
    num = 0
    
    for i, v in dictionary.items():
        if v['Subs'] == sub:
            num += v['Conc']
            
    return num / den

def rpos(dictionary, pos):
    '''
    Calculates the R value for a substituion at a given position.
    '''
    #pos is "position", a string corresponding to the integer position in the shorthand version of the molecule
    r = 0
    num = 0
    den = 0
    for i, value in dictionary.items():
        if i[pos] == '1':
            num += value
            
        #Key point here: ratio is 17O/16O, not 17O/Other O. Easy to get tripped up here. 
        if i[pos] == '0':
            den += value
    rN1 = num/den
    
    return rN1

def Usite(dictionary, pos):
    '''
    Calculates the U value for a substitution at a given position. 
    '''
    zeros = '00000000000000000000'
    num = list(['0']*20)
    num[pos] = '1'
    numerator = ''.join(num)
    
    U = dictionary[numerator]/dictionary[zeros]

    return U

def nCr(n,r):
    '''
    n Choose r
    '''
    f = math.factorial
    return f(n) / f(r) / f(n-r)
    
def computeMass(isotopologue, IDs):
    '''
    Used to predict and generate spectra with exact masses. 
    '''
    mass = 0
    for i in range(len(isotopologue)):
        if isotopologue[i] != 'x':
            element = IDs[i]
            mass += massDict[element][str(isotopologue[i])]
        
    return mass
