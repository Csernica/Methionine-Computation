import itertools
import copy
import math

import numpy as np
import pandas as pd

import basicDeltaOperations as op

##############################################################################################################################
###                                                                                                                        ###
###   This code calculates a dictionary giving all possible isotopologues of a molecule and their concentrations, based    ###
###   on input information about the sites and their isotopic composition.                                                 ###
###                                                                                                                        ###
###   The theory for this section is developed in the working M+N paper. Contact Tim for details                           ###
###                                                                                                                        ###
###   It assumes one has access to a dataframe specifying details about a molecule. See the demos.                         ###
###                                                                                                                        ###
##############################################################################################################################

#The possible substitutions, by cardinal mass, for each element. 
setsOfElementIsotopes = {'H':(0,1),'N':(0,1),'C':(0,1),'O':(0,1,2),'S':(0,1,2,4)}

def strSiteElements(df):
    '''
    Our dataframe may include multiatomic sites--for example, we may define site N1/N2 to include two nitrogens and site O3 to have one oxygen. It is useful to have a string where we can index in by position--i.e. "NNO"--to determine the chemical element at a given position. This function defines that string. 
    
    Inputs:
        df: A dataFrame containing information about the molecule.
        
    Outputs: 
        siteElements: A string giving the chemical element by position, expanding multiatomic sites. 
    '''
    elIDs = df['IDS'].values
    numberAtSite = df['Number'].values

    siteList = [(x,y) for x,y in zip(elIDs, numberAtSite)]
    siteElementsList = [site[0] * site[1] for site in siteList]
    siteElements = ''.join(siteElementsList)
    
    return siteElements

def calculateSetsOfSiteIsotopes(df):
    '''
    Every site has some set of possible isotopes. For single-atomic sites, this is equal to the set of element isotopes value for the relevant element. For multiatomic sites, it is given by a multinomial expansion of the set of element isotopes. For example, a nitrogen site with 2 atoms can have (00), (01), or (11) as possible isotopes. The number of ways to make these combinations are 1, 2, and 1 respectively. This function calculates the possible substitutions and multinomial coefficients. 
    
    Inputs:
        df: A dataFrame containing information about the molecule.
        
    Outputs: 
        setsOfSiteIsotopes: A list of tuples, where tuple i gives the possible combinations of substitutions at site i. 
        multinomialCoefficients: A list of tuples, where tuple i gives the multinomial coefficients of substitutions at site i. 
    '''
    elIDs = df['IDS'].values
    numberAtSite = df['Number'].values

    siteList = [(x,y) for x,y in zip(elIDs, numberAtSite)]

    #Determine the set of Site Isotopes for every site
    setsOfSiteIsotopes = []
    multinomialCoefficients = []

    for site in siteList:
        el = site[0]
        n = site[1]

        if n == 1:
            setsOfSiteIsotopes.append(setsOfElementIsotopes[el])
            multinomialCoefficients.append([1] * len(setsOfElementIsotopes[el]))

        else:
            siteIsotopes = tuple(itertools.combinations_with_replacement(setsOfElementIsotopes[el], n))
            setsOfSiteIsotopes.append(siteIsotopes)

            #Determining the multinomial coefficients takes a bit of work. There must be a more elegant way to do so.
            #First we generate all possible isotopic structures
            siteIsotopicStructures = itertools.product(setsOfElementIsotopes[el],repeat = n)

            #Then we sort each structure, i.e. so that [0,1] and [1,0] both become [0,1]
            sort = [sorted(list(x)) for x in siteIsotopicStructures]

            #Then we count how many times each structure occurs; i.e. [0,1] may appear twice
            counts = []
            for siteComposition in siteIsotopes:
                c = 0
                for isotopeStructure in sort:
                    if list(siteComposition) == isotopeStructure:
                        c += 1
                counts.append(c)

            #Finally, we expand the counts. Suppose we have a set of site Isotopes with [0,0], [0,1], [1,1] with 
            #multinomial Coefficients of [1,2,1], respectively. We want to output [[1,1],[2,2],[1,1]] rather than 
            #[1,2,1], because doing so will allow us to take advantage of the optimized itertools.product function
            #to calculate the symmetry number of isotopologues with many multiatomic sites.

            #One can check that by doing so, "multinomialCoefficients" and "setsOfSiteIsotopes", the output variables
            #from this section, have the same form. 
            processedCounts = [[x] * n for x in counts]

            multinomialCoefficients.append(processedCounts)
            
    return setsOfSiteIsotopes, multinomialCoefficients

def calcAllIsotopologues(setsOfSiteIsotopes, multinomialCoefficients):
    '''
    Compute all isotopologues (big A). For methionine: 995328, correct number based on hand computation. Takes ~8 seconds per loop. For much larger molecules, we will want to avoid this step and instead just calculate the MN populations we are most interested in. 
    
    Inputs:
        setsOfSiteIsotopes: A list of tuples, where tuple i gives the possible combinations of substitutions at site i. 
        multinomialCoefficients: A list of tuples, where tuple i gives the multinomial coefficients of substitutions at site i. 
        
    Outputs: 
        setOfAllIsotopologues: A list of tuples, where each tuple is an isotopologue of a molecule.
        symmetryNumbers: A list of ints, where int i gives the number of ways to construct isotopologue i. Follows same indexing as setOfAllIsotopologues. 
    '''
    i = 0
    setOfAllIsotopologues = []
    symmetryNumbers = []
    for isotopologue in itertools.product(*setsOfSiteIsotopes):
        setOfAllIsotopologues.append(isotopologue)

    #As setsOfSiteIsotopes and multinomialCoefficients are in the same form, we can use the optimized itertools.product
    #again to efficiently calculate the symmetry numbers
    for isotopologue in itertools.product(*multinomialCoefficients):
        flat = [x[0] if type(x) == list else x for x in isotopologue]
        n = np.array(flat).prod()

        symmetryNumbers.append(n)
                             
    return setOfAllIsotopologues, symmetryNumbers

def siteSpecificConcentrations(df):
    '''
    Calculates all site-specific concentrations and puts them in an array for easy access. Note that at present, it only works for C,N,O,S,H. If we add new elements, we may need to play with the structure of this function. 
    
    The basic structure of the array is: array[i][j] gives the concentration of an isotope with cardinal mass difference i at position j. 
    
    Inputs:
        df: A dataFrame containing information about the molecule.
        
    Outputs:
        concentrationArray: A numpy array giving the concentration of each isotope at each site. 
    '''
    elIDs = df['IDS'].values
    numberAtSite = df['Number'].values
    deltas = df['deltas'].values
    
    concentrationList = []
    for index in range(len(elIDs)):
        element = elIDs[index]
        delta = deltas[index]
        concentration = op.deltaToConcentration(element, delta)
        concentrationList.append(concentration)

    #put site-specific concentrations into a workable form
    unsub = []
    M1 = []
    M2 = []
    M3 = []
    M4 = []
    for concentration in concentrationList:
        unsub.append(concentration[0])
        M1.append(concentration[1])
        M2.append(concentration[2])
        M3.append(0)
        M4.append(concentration[3])

    concentrationArray = np.array(unsub), np.array(M1), np.array(M2), np.array(M3), np.array(M4)
    
    return concentrationArray

def calculateIsotopologueConcentrations(setOfAllIsotopologues, symmetryNumbers, concentrationArray):
    '''
    Puts information about the isotopologues of a molecule, their symmetry numbers, and concentrations of individual isotopes together in order to calculate the concentration of each isotopologue. Does so under the stochastic assumption, i.e. assuming that isotopes are distributed stochastically across all isotopologues.
    
    This is a computationally expensive step--ways to improve would be welcome. Takes ~20 seconds for methionine. For molecules where it is too expensive, it would be expedient to avoid calculating all isotopologues and only calculate the M1, M2, etc populations of interest. 
    
    Inputs:
        setOfAllIsotopologues: A list of tuples, where each tuple is an isotopologue of a molecule.
        symmetryNumbers: A list of ints, where int i gives the number of ways to construct isotopologue i. Follows same indexing as setOfAllIsotopologues. 
        concentrationArray: A numpy array giving the concentration of each isotope at each site. 
        
    Outputs:
        d: A dictionary where the keys are string representations of each isotopologue and the values are dictionaries. For example, a string could be '00100', where there is an M1 substitution at position 3 and M0 isotopes at all other sites. The value dictionaries include "Conc", or concentration, and "num", giving the number of isotopologues of that form. The sum of all concentrations should be 1.  
        
        The keys can be "expanded" strings, i.e. including multiple atomic sites in parentheses. For example, N1/N2 and O3 would appear as (0,1)0. 
    '''
    d = {}
    for i, isotopologue in enumerate(setOfAllIsotopologues):
        number = symmetryNumbers[i]
        isotopeConcList = []
        for index, value in enumerate(isotopologue):
            if type(value) == tuple:
                isotopeConc = [concentrationArray[sub][index] for sub in value]
                isotopeConcList += isotopeConc
            else:
                isotopeConc = concentrationArray[value][index]
                isotopeConcList.append(isotopeConc)      

        isotopologueConc = np.array(isotopeConcList).prod()

        string = ''.join(map(str, isotopologue))

        d[string] = {'Conc':isotopologueConc * number,'num':number}
        
    return d

def condenseStr(text):
    '''
    Takes the "expanded" string depictions, i.e. "(0,1)0" for multiatomic sites and transforms them into "condensed" depictions, i.e. "010". This makes it easy to pick out the element for a particular substitution, for example, by finding the index of the condensed depiction and looking at that same index in strSiteElements. 
    
    Inputs:
        text: A string, the "expanded" string depiction. 
        
    Outputs:
        text: A string, the "condensed" string depiction. 
    '''
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace(',', '')
    text = text.replace(' ', '')
    
    return text
 
def uEl(el, n):
    '''
    Returns the type of substitution, given a chemical element and cardinal mass of isotope.
    
    Inputs:
        el: A string, giving the element of interest
        n: An int, giving the cardinal mass of the isotope
        
    Returns: 
        A string identifying the isotope substitution. 
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
        
def calcCondensedDictionary(isotopologueConcentrationDict, df):
    '''
    Given the dictionary from calculateIsotopologueConcentrations, calculates another dictionary with more complete information. Takes the "expanded" string depictions i.e. "(0,1)0" to "condensed" depictions i.e. "010" and makes these the keys. Stores the expanded depictions, number, and concentration for each isotopologue, then additionally calculates their mass and relevant substitutions. 
    
    Computationally expensive; 20 seconds for methionine.
    
    An example entry from methionine for the unsubstituted isotopologue is shown below:  
    
    '000000000000000000000': {'Number': 1,
      'full': '00(0, 0)00000(0, 0, 0)(0, 0)(0, 0, 0)(0, 0)00',
      'Conc': 0.8906400439358315,
      'Mass': 0,
      'Subs': ''}
      
    Inputs: 
        isotopologueConcentrationDict: The output from calculateIsotopologueConcentrations. 
        df: A dataFrame containing information about the molecule.
        
    Outputs: 
        byCondensed: A new dictionary containing more complete information about the isotopologues. 
    '''
    siteElements = strSiteElements(df)
    
    byCondensed = {}
    for i, v in isotopologueConcentrationDict.items():
        condensed = condenseStr(i)
        byCondensed[condensed] = {}
        byCondensed[condensed]['Number'] = v['num']
        byCondensed[condensed]['full'] = i
        byCondensed[condensed]['Conc'] = v['Conc']
        byCondensed[condensed]['Mass'] = np.array(list(map(int,condensed))).sum()
        byCondensed[condensed]['Subs'] = ''.join([uEl(element, int(number)) for element, number in zip(siteElements, condensed)])
    
    return byCondensed

def calcSubDictionary(isotopologueConcentrationDict, df):
    '''
    Similar to the "byCondensed" dictionary, a more complete depiction of all isotopologues of a molecule. In this case, rather than index in by condensed string, index in by substitution--i.e., the key '17O' gives information for all isotopologues with the substituion '17O'. This is a better way to index into this information when we want to calculate results of mass spectrometry experiments. 
    
    Computationally expensive; 20 seconds for methionine. 
    
    An example entry from methionine is shown below.
    
    'D': {'Number': 12,
      'Full': ['00(0, 0)00000(0, 0, 0)(0, 0)(0, 0, 0)(0, 0)01',
       '00(0, 0)00000(0, 0, 0)(0, 0)(0, 0, 0)(0, 0)10',
       '00(0, 0)00000(0, 0, 0)(0, 0)(0, 0, 0)(0, 1)00',
       '00(0, 0)00000(0, 0, 0)(0, 0)(0, 0, 1)(0, 0)00',
       '00(0, 0)00000(0, 0, 0)(0, 1)(0, 0, 0)(0, 0)00',
       '00(0, 0)00000(0, 0, 1)(0, 0)(0, 0, 0)(0, 0)00'],
      'Conc': 0.0015953500722996194,
      'Mass': [1, 1, 1, 1, 1, 1],
      'Condensed': ['000000000000000000001',
       '000000000000000000010',
       '000000000000000000100',
       '000000000000000010000',
       '000000000000010000000',
       '000000000001000000000']},
      
    Inputs: 
        isotopologueConcentrationDict: The output from calculateIsotopologueConcentrations. 
        df: A dataFrame containing information about the molecule.
        
    Outputs: 
        bySub: A new dictionary containing more complete information about the isotopologues. 
    '''
    siteElements = strSiteElements(df)
    
    bySub = {}
    for i, v in isotopologueConcentrationDict.items():
        condensed = condenseStr(i)
        Subs = ''.join([uEl(element, int(number)) for element, number in zip(siteElements, condensed)])
        if Subs not in bySub:
            bySub[Subs] = {'Number': 0, 'Full': [],'Conc': 0, 'Mass': [], 'Condensed': []}
        bySub[Subs]['Number'] += v['num']
        bySub[Subs]['Full'].append(i)
        bySub[Subs]['Conc'] += v['Conc']
        bySub[Subs]['Mass'].append(np.array(list(map(int,condensed))).sum())
        bySub[Subs]['Condensed'].append(condensed)
        
    return bySub

def massSelections(condensedDictionary, massThreshold = 4):
    '''
    pulls out M0, M1, etc. populations from the condensed dictionary, up to specified threshold. 
    
    Inputs:
        condensedDictionary: A dictionary with information about all isotopologues, keyed by condensed strings. The output of calcCondensedDictionary.
        massThreshold: An int. Does not include populations with cardinal mass difference above this threshold. 
        
    Outputs:
        A dictionary where the keys are "M0", "M1", etc. and the values are dictionaries containing all isotopologues from the condensed dictionary with a specified cardinal mass difference. 
    '''
    MNDict = {}
    
    for i in range(massThreshold+1):
        MNDict['M' + str(i)] = {}
        
    for i, v in condensedDictionary.items():
        for j in range(massThreshold+1):
            if v['Mass'] == j:
                MNDict['M' + str(j)][i] = v
            
    return MNDict