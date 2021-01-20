import numpy as np
import pandas as pd

# DEFINE STANDARDS
#Doubly isotopic atoms are given as standard ratios, i.e. RPDB for carbon
STD_Rs = {"H": 0.00015576, "C": 0.0112372, "N": 0.003676, "17O": 0.0003799, "18O": 0.0020052,
         "33S":0.007895568,"34S":0.044741552,"36S":0.000105274}

def deltaToConcentration(atomIdentity,delta):
    '''
    Converts an input delta value for a given type of atom in some reference frame to a 4-tuple containing the 
    concentration of the unsubstituted, M+1, M+2, and all other versions of the atom.
    
    Inputs:
        atomIdentity: A string giving the type of atom, i.e. 'C'
        delta: The input delta value. This must be in the VSMOW, PDB, or AIR standards for D, 13C, and 15N, respectively
        
    Outputs:
        The ratio for this delta value.
    '''
    if atomIdentity in 'HCN':
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
    return concentrationTuple[1]/concentrationTuple[0] 

def ratioToDelta(atomIdentity, ratio):
    delta = 0
    if atomIdentity in 'HCN':
        delta = (ratio/STD_Rs[atomIdentity]-1)*1000
        
    elif atomIdentity == 'O':
        delta = (ratio/STD_Rs['17O']-1)*1000
        
    elif atomIdentity == '18O':
        delta = (ratio/STD_Rs['18O']-1)*1000
        
    elif atomIdentity == 'S':
        delta = (ratio/STD_Rs['33S']-1)*1000
        
    elif atomIdentity == '34S':
        delta = (ratio/STD_Rs['34S']-1)*1000
        
    elif atomIdentity == '36S':
        delta = (ratio/STD_Rs['36S']-1)*1000
        
    else:
        print('Sorry, I do not know how to deal with ' + atomIdentity)
        
    return delta
    
def extractConcTuples(concentrationTuples):
    concUnSubList = [x[0] for x in concentrationTuples]
    concSubList = [x[1] for x in concentrationTuples]
    
    return concUnSubList, concSubList
    
def probabilityM1(concUnSubList, concSubList,stoichiometryVector):
    unSubProbs = np.asarray(concUnSubList)**np.asarray(stoichiometryVector)
    probM1 = unSubProbs.prod() / concUnSubList* concSubList
    return probM1
    
def fractionalAbundance(probM1,stoichiometryVector):
    probM1Stoich = probM1 * stoichiometryVector
    fracAbund = probM1Stoich/probM1Stoich.sum()
    return fracAbund 

def deltasToFractionalAbundance(deltas, elements, stoichiometry):
    concentrationTuples = [deltaToConcentration(x,y) for x, y in zip(elements, deltas)]
    concUnSubList, concSubList = extractConcTuples(concentrationTuples)
    ratios = [concentrationToRatio(x) for x in concentrationTuples]
    probM1 = probabilityM1(concUnSubList, concSubList,stoichiometry)
    fA = fractionalAbundance(probM1, stoichiometry)
    return fA

def updateDataFramePercentAbundance(dataFrame):
    '''
    When initializing a new molecular dataframe or after updating delta values, this function calculates ratios,
    concentrations, probability unsubstituted, probability of a substitution, and percent abundance of substitutions
    for all sites. 
    
    Inputs:
        dataFrame: a molecular dataframe, consisting of at a minimum atom IDs, element IDs, and delta values
        
    Outputs:
        The same dataFrame, with entries for ratio, concentration, probability unsubstituted, probability of
        a substitution, and percent abundances of substitutions added or updated. 
    '''
    deltas = dataFrame['Ref Deltas'].values
    elements = dataFrame['element'].values
    stoichiometry = dataFrame['Stoich'].values
    
    concentrationTuples = [deltaToConcentration(x,y) for x, y in zip(elements, deltas)]
    concUnSubList, concSubList = extractConcTuples(concentrationTuples)
    ratios = [concentrationToRatio(x) for x in concentrationTuples]
    probM1 = probabilityM1(concUnSubList, concSubList,stoichiometry)
    fA = fractionalAbundance(probM1, stoichiometry)
    
    dataFrame['Ratio'] = ratios
    dataFrame['Concentration'] = concSubList
    dataFrame['Probability Unsubstituted'] = concUnSubList
    dataFrame['Probability M1'] = probM1
    dataFrame['Fractional Abundance From Calculation'] = fA
    
    return dataFrame

def singlySubstitutedStochastic(dataFrame):
    '''
    Used to predict output of "O" measurement. Generates probability of seeing a single substition at each site,
    as well as the probability of the unsubstituted. 
    '''
    deltas = dataFrame['Ref Deltas'].values
    elements = dataFrame['element'].values
    stoichiometry = dataFrame['Stoich'].values
    IDS = dataFrame['atom ID'].values
    output = {}

    concentrationTuples = [deltaToConcentration(x,y) for x, y in zip(elements, deltas)]

    for i in range(len(concentrationTuples)):
        prob = concentrationTuples[i][1]
        for j in range(len(concentrationTuples)):
            if j!= i:
                prob *= concentrationTuples[j][0]
        ID = IDS[i]
        prob *= stoichiometry[i]
        output[ID] = prob
    
    prob = 1
    for i in range(len(concentrationTuples)):
        prob *= concentrationTuples[i][0]
    output['Unsub'] = prob
    
    return output

def stochasticOValues(s):
    '''
    s is the dictionary of stochastic concentrations from singlySubstitutedStochastic
    '''
    OValues = {}
    for key, value in s.items():
        OValues[key] = s[key] / s['Unsub']
    return OValues
    
def errorWeightedMean(x, xerr, y, yerr):
    '''
    https://ned.ipac.caltech.edu/level5/Leo/Stats4_5.html
    
    Given two constraints on the same quantity, as numpy arrays, compute the error weighted mean.
    
    Outputs a numpy array for mean and error.
    Inputs:
        x, xerr, y, yerr: Numpy arrays, giving the values and error, respectively, for the two constraints
        
    Outputs:
        my, err: Numpy arrays, giving the values and error, respectively, for the output
    '''
    
    errWeightedMean = []
    errWeightedMeanStd = []
    for index in range(len(x)):
        num = 0
        den = 0
        num += x[index] / xerr[index]**2
        den += 1/ xerr[index]**2

        num += y[index] / yerr[index]**2
        den += 1/ yerr[index]**2

        mu = num/den
        err = (1/den)**(1/2)

        errWeightedMean.append(mu)
        errWeightedMeanStd.append(err)

    mu = np.array(errWeightedMean)
    err = np.array(errWeightedMeanStd)

    return mu, err   

def findAverageDelta(deltas, stoich, element):
    '''
    Takes some delta values for a given element, converts them to concentration space, finds the average concentration, and 
    returns them to delta space. 
    
    Inputs: 
        deltas: A numpy array
        stoich: A numpy array
        element: A string
        
    Outputs:
        delta: A float
    '''
    concentrations = [deltaToConcentration('C', x) for x in deltas]
    
    M0Conc = [c[0] for c in concentrations]
    M1Conc = [c[1] for c in concentrations]
    
    average0 = (stoich * M0Conc).sum() / stoich.sum()
    average1 = (stoich * M1Conc).sum() / stoich.sum()
    
    avgConc = (average0, average1, 0)
    averageR = concentrationToRatio(avgConc)
    delta = ratioToDelta('C',averageR)
    
    return delta

def findAverageDeltaDf(element, dataFrame, Header = 'Deltas'):
    '''
    Extracts the deltas and stoichiometry from site-specific data in a dataFrame for a given element, and finds the average delta. The "Header" inputcan be used in cases where deltas are calculated multiple ways, i.e. from and M1/EA or M1/O experiment, and therefore there are multiple columns with deltas in the dataFrame. 
    
    Inputs:
        element: A string
        dataFrame: A pandas dataFrame with a "deltas" column giving site-specific values. If there is also bulk information 
        in the dataFrame, this function will fail. 
        Header: A string, the column header to run the function on. 
        
    Outputs:
        average: A float, giving the average delta
    '''
    deltas = dataFrame[dataFrame['element'] == element][Header].values
    stoich = dataFrame[dataFrame['element'] == element]['Stoich'].values
    
    average = findAverageDelta(deltas, stoich, element)

    return average

def setElement(string):
    if 'H' in string:
        return 'H'
    
    if 'C' in string:
        return 'C'
    
    if 'N' in string:
        return 'N'
    
    if 'O' in string:
        return 'O'
    
    if 'S' in string:
        return 'S'