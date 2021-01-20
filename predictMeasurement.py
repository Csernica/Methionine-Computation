###Deprecated as of 1/20/2021. Currently, to calculate measurements we calculate the set of all isotopologues and proceed. 

import numpy as np
import basicDeltaOperations as op
import pandas as pd
import M2Module as M2
import ClumpingModule as clump
import math

def defineFragmentReferenceDict(dataFrame,fragmentList):
    '''
    Creates a dictionary containing information about the 119.C, 119.N, 119.H vectors from a dataFrame with full
    fragment information
    '''
    
    vectorReferenceDict = {}
    isotopeSet = set()
    for element in dataFrame['element']:
        if element not in isotopeSet:
            isotopeSet.add(element)
    for isotope in isotopeSet:
        vectorReferenceDict[isotope] = {}
        for fragment in fragmentList:
            vectorReferenceDict[isotope][fragment] = np.asarray((dataFrame['element']==isotope) * dataFrame[fragment].values)
    return vectorReferenceDict

def calculateExpectedMeasurements(fA, VRD, fragmentList, notResolved = [], outputPandas = False):
    '''
    fA: Fractional Abundance Vector
    VRD: Vector Reference Dictionary
    '''

    outputDict = {}
    for key in VRD.keys():
        if key not in notResolved:
            outputDict[key] = {}
            for fragment in fragmentList:
                outputDict[key][fragment] = (fA * VRD[key][fragment]).sum()
        else:
            combinedKey = ''.join(notResolved)
            if combinedKey not in outputDict:
                outputDict[combinedKey] = {}

            for fragment in fragmentList:
                if fragment not in outputDict[combinedKey]:
                    outputDict[combinedKey][fragment] = (fA * VRD[key][fragment]).sum()

                else: 
                    outputDict[combinedKey][fragment] += (fA * VRD[key][fragment]).sum()
    
    if outputPandas:
        output = pd.DataFrame.from_dict(outputDict, orient = 'index',columns = fragmentList)
        return output
    else:
        return outputDict

def generateExpectedM1Measurements(dataFrame, fragmentList, notResolved = []):
    VRD = defineFragmentReferenceDict(dataFrame, fragmentList)
    op.updateDataFramePercentAbundance(dataFrame)
    fA = dataFrame['Fractional Abundance From Calculation'].values
    
    measurements = calculateExpectedMeasurements(fA, VRD,fragmentList, notResolved = notResolved, outputPandas = True)
    
    return measurements

def findAbsoluteConcentrations(M2Isotopologues, cVector, stoich):
    '''
    given a vector list of M2Isotopologues and a concentration vector for a molecule, finds the absolute
    concentration of each isotopologue
    '''
    
    absConc = []
    for isotopologue in M2Isotopologues:
        #site stoichiometry, not clumped stoichiometry. Calculates unsubstituted concentration
        conc = (cVector[0] ** stoich).prod()
        #Iterate through and add substitutions one by one
        for siteIndex in range(len(isotopologue)):
            subCode = isotopologue[siteIndex]
            if subCode == 0:
                continue
            if subCode == 1:
                conc /= cVector[0][siteIndex]
                conc *= cVector[1][siteIndex]
            if subCode == 2:
                conc /= cVector[0][siteIndex]**2
                conc *= cVector[1][siteIndex]**2
            if subCode == -1:
                conc /= cVector[0][siteIndex]
                conc *= cVector[2][siteIndex]

        absConc.append(conc)

    absConc = np.array(absConc)
    
    return absConc

def generateExpectedM2Measurements(dataFrame, inputFile, fragmentList, notResolved = []):
    '''
    Predicts the expected output of an M2 experiment
    '''
    #pull out of dataFrame for easy access
    elements = list(dataFrame['element'].values)
    atomID = list(dataFrame['atom ID'].values)
    stoich = list(dataFrame['Stoich'].values)
    refDeltas = dataFrame['Ref Deltas'].values
    
    #Generate the vector representation of the M2 isotopologues and track which substitutions they correspond to
    M2Output = M2.dataFrameToM2Matrix(dataFrame, inputFile, measurement = False)
    
    #Determine the concentration vector for the molecule
    concentration = [op.deltaToConcentration(x,y) for x, y in zip(elements, refDeltas)]
    cVector = np.transpose(concentration)

    #Determine the relative concentrations of the M2 isotopologues
    absConc = findAbsoluteConcentrations(M2Output['Isotopologues']['Matrix'], cVector, stoich)
    absConcStoich = absConc * M2Output['Isotopologues']['Clumped Stoich']
    relConcStoich = absConcStoich / absConcStoich.sum()
    relConc = relConcStoich / M2Output['Isotopologues']['Clumped Stoich']
    
    M2Population = {'Isotopologue':M2Output['Isotopologues']['Precise Identity'],
                    'Absolute Concentration': absConc,
                    'Relative Concentration': relConc}
    
    M2PopDf = pd.DataFrame.from_dict(M2Population)
    
    #Apply the M2 composition matrix to the relative concentraiton vector to predict observed measurements
    measurement = np.dot(M2Output['Composition Matrix'], relConc)
    
    M2Meas = {}
    
    numObservationsPerFragment = int((len(measurement)-1)/len(fragmentList))
    
    x = 1
    for fragment in fragmentList:
        #1 index because we include closure in measurement
        M2Meas[fragment] = measurement[x:x+numObservationsPerFragment]
        x += numObservationsPerFragment
        
    M2Meas['Order'] = M2Output['Full Order'][1: 1+numObservationsPerFragment]
    M2MeasDf = pd.DataFrame.from_dict(M2Meas)
    M2MeasDf.set_index('Order',inplace = True, drop = True)
    
    return M2PopDf, M2MeasDf

def elementSpecificUValue(dataFrame, element):
    '''
    Given a dataframe, calculates an element-specific U value for a given element. 
    '''
    if 'Ratios' not in dataFrame:
        dataFrame['Ratios'] = [op.concentrationToRatio(op.deltaToConcentration(x, y)) for x, y in zip(dataFrame['element'].values, dataFrame['Ref Deltas'].values)]
        
    cList = clump.defineConcentrationList(dataFrame)
    cVector = clump.defineConcVector(cList)
    unsub = clump.calcUnsub(cVector)

    #Get indices with that element
    subs = 0
    for i, v in dataFrame[dataFrame['element'] == element].iterrows():
        subs += clump.singleSub(cVector, i)

    UValue = subs / unsub

    return UValue


######################## NEW PREDICT VIA BIG A ##################################################

subDict = {'C':{'0':'','1':'13C'},
           'N':{'0':'','1':'15N'},
           'H':{'0':'','1':'D'},
           'O':{'0':'','1':'17O','2':'18O'},
           'S':{'0':'','1':'33S','2':'34S','4':'36S'}}

massDict = {'C':{'0':12,'1':13.00335484},
            'N':{'0':14.003074,'1':15.00010889},
            'H':{'0':1.007825032,'1':2.014101778},
            'O':{'0':15.99491462,'1':16.99913175,'2':17.9991596},
            'S':{'0':31.97207117,'1':32.9714589,'2':33.96786701,'4':35.9670807}}

def nCr(n,r):
    '''
    n Choose r
    '''
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def condenseStr(text):
    '''
    Condense the string depiction for easier calculation. When there are condensed depictions of isotopologues,
    they include something like "(0,0)" for multiatomic sites. This function strips those extra characters. 
    '''
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace(',', '')
    text = text.replace(' ', '')
    
    return text
    
def fragMult(z, y):
    '''
    Fragments an individual site of an isotopologue
    '''
    if z == 'x' or y == 'x':
        return 'x'
    else:
        return z*y
    
def intX(n):
    '''
    Interprets 'x' as 'x' and '1' as an integer; used to apply a fragmentation vector. 
    '''
    if n == 'x':
        return 'x'
    else:
        return int(n)
    
def expandFrag(frag, number):
    '''
    Creates an expanded description of a frament. For example, if I fragment [0,(0,1)] with fragmentation vector
    [0,1], I do so by applying the fragmentation vector [011] to the isotopologue [001], expanding the tuple. This
    expands the fragmentation vector to match the expanded tuple. 
    '''
    intermediate = []
    for i, v in enumerate(frag):
        intermediate += [v] * number[i]
    final = [intX(x) for x in intermediate]
    
    return final

def uEl(el, n):
    '''
    Returns the type of substitution, given a chemical element and shorthand notation. 
    '''
    if n == 0:
        return ''
    if n == 'x':
        return ''
    if el == 'C':
        if n == 1:
            return '13C'
    if el == 'H':
        if n == 1:
            return 'D'
    if el == 'O':
        if n == 1:
            return '17O'
        if n == 2:
            return '18O'
    if el == 'N':
        if n == 1:
            return '15N'
    if el == 'S':
        if n == 1:
            return '33S'
        if n == 2:
            return '34S'
        if n == 4:
            return '36S'
        
def computeMass(isotopologue, IDs):
    mass = 0
    for i in range(len(isotopologue)):
        if isotopologue[i] != 'x':
            element = IDs[i]
            mass += massDict[element][str(isotopologue[i])]
        
    return mass

def computeSubs(isotopologue, IDs):
    subs = []
    for i in range(len(isotopologue)):
        if isotopologue[i] != 'x':
            element = IDs[i]
            if subDict[element][str(isotopologue[i])] != '':
                subs.append(subDict[element][str(isotopologue[i])])
        
    return '/'.join(subs)

def UValueBySub(dictionary, sub):
    '''
    inefficient but user-friendly
    '''
    den = dictionary['000000000000000000000']['Conc']
    
    num = 0
    
    for i, v in dictionary.items():
        if v['Subs'] == sub:
            num += v['Conc']
            
    return num / den

def rpos(dictionary, pos):
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
    zeros = '00000000000000000000'
    num = list(['0']*20)
    num[pos] = '1'
    numerator = ''.join(num)
    
    U = dictionary[numerator]/dictionary[zeros]

    return U