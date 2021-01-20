import itertools
import numpy as np
import basicDeltaOperations as op
import pandas as pd
import copy
import math

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

setsOfElementIsotopes = {'H':(0,1),'N':(0,1),'C':(0,1),'O':(0,1,2),'S':(0,1,2,4)}

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

def strSiteElements(df):
    '''
    Create a string storing the chemical elements in condensed notation. I.e. 'C, (H, H)' becomes 'CHH'.
    This allows us to easily access the chemical element at a given site. 
    '''
    elIDs = df['IDS'].values
    numberAtSite = df['Number'].values

    siteList = [(x,y) for x,y in zip(elIDs, numberAtSite)]
    siteElementsList = [site[0] * site[1] for site in siteList]
    siteElements = ''.join(siteElementsList)
    
    return siteElements

def calculateSetsOfSiteIsotopes(df):
    '''
    Creates a list of tuples, where each tuple gives the shorthand form of a possible isotopologue. The list contains
    all possible isotopologues. 
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
    Compute all isotopologues (big A). For methionine: 663552, correct number based on hand computation. 268 ms to compute.

    If calculating numbers as well, 4.45 seconds. Compare to 1.57 seconds for calculating without multinomial coefficients. We lose here by using multiatomic sites as we need to track more things. However, the major time cost comes later, with the calculation of individual isotopologue concentrations. The extra seconds spent here are worth it. 
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
    Calculates site-specific concentrations from input delta values, using the op.deltaToConcentration function. Note that at present, it only works for C,N,O,S,H. If we add new elements, we may need to play with the structure of this function. 
    
    Outputs these as an array, where array[i][j] gives the concentration of an isotope with cardinal mass difference i at position j. 
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
    Compute concentrations for each isotopologue and put them into a dictionary
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

    