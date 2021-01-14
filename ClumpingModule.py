import basicDeltaOperations as op
import numpy as np

###Deprecated as of 9/28/2020...I redid the basic delta operations to have 4 entries rather than 3, allowing us to track 
###S36. This code may or may not work, have not checked.

def defineConcentrationList(dataFrame):
    '''
    Takes a dataFrame with O Values as Ratios and defines a list of tuples; each tuple is the concentration of 
    a given site, given as (Unsub, M1, Others)
    '''
    elementArray = dataFrame['element'].values
    
    if 'Ratios' not in dataFrame:
        ratioArray = [op.concentrationToRatio(op.deltaToConcentration(x, y)) for x, y in zip(dataFrame['element'].values, dataFrame['Ref Deltas'].values)]
        print("No calculated site-specific ratios found, using reference deltas")
    else:
        print("Found site-specific ratios, ignoring reference deltas")
        
        ratioArray = dataFrame['Ratios'].values
        
    concentrationList = []
    for index in range(len(elementArray)):
        element = elementArray[index]
        ratio = ratioArray[index]
        concentration = op.deltaToConcentration(element, op.ratioToDelta(element, ratio))
        concentrationList.append(concentration)
        
    return concentrationList

def defineConcVector(concentrationList):
    '''
    This is just np.transpose. But it turns out this is slightly faster than np.transpose for sizes up to methionine.
    '''
    unsub = []
    M1 = []
    MOther = []
    for concentration in concentrationList:
        unsub.append(concentration[0])
        M1.append(concentration[1])
        MOther.append(concentration[2])
    
    return np.array(unsub), np.array(M1), np.array(MOther)

def calcUnsubDf(dataFrame):
    concList = defineConcentrationList(dataFrame)
    v = defineConcVector(concList)
    
    return v[0].prod()

def calcUnsub(concVector):
    
    return concVector[0].prod()

def addSub(concentration, concVector, index):
    '''
    concentration = concentration without the substitution; i.e. this could be [Unsub]. A float.     
    '''
    concentration *= concVector[1][index]
    concentration /= concVector[0][index]
    return concentration

def addM2Sub(concentration, concVector, index):
    concentration *= concVector[2][index]
    concentration /= concVector[0][index]
    
    return concentration

def singleM2Sub(concVector, index):
    unsub = calcUnsub(concVector)
    M2 = addM2Sub(unsub, concVector, index)
    
    return M2

def singleSub(concVector, index):
    unsub = calcUnsub(concVector)
    single = addSub(unsub, concVector, index)
    
    return single

def doubleSub(concVector, index1, index2):
    single = singleSub(concVector, index1)
    double = addSub(single, concVector, index2)
    
    return double
    
def singleSubID(concVector, atomID, atomIDList):
    index = atomIDList.index(atomID)
    
    return singleSub(concVector, index)

def doubleSubID(concVector, atomID1, atomID2, atomIDList):
    index1 = atomIDList.index(atomID1)
    index2 = atomIDList.index(atomID2)
    
    return doubleSub(concVector, index1, index2)

def clumpedIndices(dataFrame, element1, element2):
    '''
    Given two elements where a clump exists, determines sets of substituted indices compatible with that clump. For example,
    if I have C3 and specify a CC clump, I may have substitutions at 1/2, 1/3, or 2/3. Returns these indices as a list of 
    tuples (e.g. [(1,2),(1,3),(2,3)]
    '''

    e1 = dataFrame[dataFrame['element'] == element1]
    e2 = dataFrame[dataFrame['element'] == element2]
    stoich1 = e1['Stoich']
    stoich2 = e2['Stoich']

    isotopologues = []
    for index1, value1 in stoich1.items():
        for iteration1 in range(int(value1)):

            #correct for stoichiometry > 1
            if value1 > 1 and element1 == element2:
                isotopologues += [(index1, index1)] * int((value1 - 1- iteration1))

            for index2, value2 in stoich2.items():
                if index2 > index1:
                    for iteration2 in range(int(value2)):
                        isotopologues.append((index1, index2))
    
    return isotopologues

def singleTotalConc(dataFrame, element):
    concList = defineConcentrationList(dataFrame)
    v = defineConcVector(concList)
    
    e1 = dataFrame[dataFrame['element'] == element]
    singleSubConc = 0
    for index, value in e1['Stoich'].items():
        conc = singleSub(v, index)
        singleSubConc += conc * value
        
    return singleSubConc
    
def clumpedConc(dataFrame, element1, element2):
    isotopologues = clumpedIndices(dataFrame, element1, element2)
    
    concList = defineConcentrationList(dataFrame)
    v = defineConcVector(concList)
    
    clumpedConc = 0
    for clumped in isotopologues:
        conc = doubleSub(v, clumped[0], clumped[1])
        clumpedConc += conc
    
    return clumpedConc
        
def clumpedOValue(dataFrame, element1, element2):
    clumped = clumpedConc(dataFrame, element1, element2)
    unsub = calcUnsubDf(dataFrame)
    
    return clumped / unsub

def processOrbiOClumped(dataFrame, clumpedO):
    '''
    dataFrame with no bulk values added
    '''
    
    output = {}
    for key, value in clumpedO['Sample'].items():
        elements = key.split('/')
        stochastic = clumpedOValue(OrbiM1, elements[0], elements[1])
        delta = 1000*(value['Measurement'] / stochastic - 1)
        output[key] = delta
        
    return output
    
def addOrbiOClumpedToOutput(dataFrame, clumpedO, outputDf):
    '''
    outputDf can have bulk values added
    '''
    for key, value in clumpedO['Sample'].items():
        element = key
        elements = key.split('/')
        stoich = len(clumpedIndices(dataFrame, elements[0], elements[1]))
        stochastic = clumpedOValue(dataFrame, elements[0], elements[1])
        delta = 1000*(value['Measurement'] / stochastic - 1)
        
        #correction to input data for Bremen Meeting, June 29
        #if element == 'C/C':
        #    concList = defineConcentrationList(dataFrame)
        #    v = defineConcVector(concList)
        #    unsub = calcUnsub (v)
        #    delta = 1000 * ((value['Measurement'] / unsub) / stochastic -1)      
        
        newRow = {'atom ID':"clumped " + key,'element':key, 'Stoich':stoich,'Weighted Deltas':delta}
        outputDf = outputDf.append(newRow,ignore_index = True)
        
    return outputDf

def addM2PlusClumpToOutput(dataFrame, clumpedOM2Plus, outputDf):
    '''
    outputDf can have bulk values added
    In fact, must be run after the bulk ratios have been computed
    
    Uses the O = R approximation
    '''
    unsub = calcUnsubDf(dataFrame)
    for key, value in clumpedOM2Plus['Sample'].items():
        elements = key.split('/')
        singleSub = singleTotalConc(dataFrame, elements[0])
        doubleSub = singleSub * outputDf[outputDf['element'] == elements[1]]['Ratios'].values
        
        stochastic = doubleSub / unsub
        delta = 1000*(value['Measurement'] / stochastic -1)
        
        newRow = {'atom ID': "clumped " + key, 'element': key, 'Weighted Deltas': delta.item()}
        outputDf = outputDf.append(newRow, ignore_index = True)
        
    return outputDf
        
def fragmentSingleConc(dataFrame, fragment, element):
    '''
    Given a specific fragment, i.e. the 56 fragment of methionine, and a specific element, i.e C, computes the
    total concentration of that element in that fragment. I.e., in methionine, it is the sum of the concentrations
    of the singly-substituted C isotopologues at those sites. (note this is subtly wrong, as it will also include
    the concentrations of any multiply-substituted isotopologues which lose the non-13C substitution during
    fragmentation)
    '''

    concList = defineConcentrationList(dataFrame)
    v = defineConcVector(concList)

    frag = dataFrame[dataFrame[fragment] == 1]
    elementFrag = frag[frag['element']== element]

    ratios = elementFrag['Ratios']
    stoich = elementFrag['Stoich']

    totalConc = 0

    for index, value in ratios.items():
        s = singleSub(v, index)
        totalConc += s * stoich[index]
    
    return totalConc

def allowedIndex(forbiddenIndices, isotopologue):
    for index in forbiddenIndices:
        if index in isotopologue:
            return False
    return True

def fragmentDoubleConc(dataFrame, fragment, element1, element2):
    '''
    Given a specific fragment, i.e. the 56 fragment of methionine, and two elements, computes the
    total concentration of those clumped elements in that fragment.  (note this is subtly wrong, as it will also include
    the concentrations of any multiply-substituted isotopologues which lose the non-13C substitution during
    fragmentation)
    '''
    allIsotopologues = clumpedIndices(dataFrame, element1, element2)
    forbiddenIndices = []
    for index, value in dataFrame[dataFrame[fragment] != 1]['element'].items():
        forbiddenIndices.append(index)
    culled = [x for x in allIsotopologues if allowedIndex(forbiddenIndices, x)]
    
    concList = defineConcentrationList(dataFrame)
    v = defineConcVector(concList)
    totalConc = 0
    for isotopologue in culled:
        toAdd = doubleSub(v, isotopologue[0],isotopologue[1])
        totalConc += toAdd
        
    return totalConc
                
def addM2ExperimentToOutput(dataFrame, M2FragDict, Output):
    '''
    Goes through an experiment where we isolate both M1 and M2 species of a fragment, compare the M2 to the 
    height of the M1, and use this to extract clumping information
    
    Approximate
    '''
    for fragment, info in M2FragDict['Sample'].items():
        for key, value in info.items():
            delta = None
            elements = key.split('/')

            #Clumped species
            if '-' in elements[0]:
                clumped = elements[0].split('-')
                singleSub = fragmentSingleConc(dataFrame, '56', elements[1])
                clumpedConc = value['Measurement'] * singleSub

                stochastic = fragmentDoubleConc(dataFrame, '56', clumped[0],clumped[1])
                unsub = calcUnsubDf(dataFrame)

                #This delta is approximate; have to look into the theory in more detail of how clumping affects what
                #unsub concentration we should use
                delta = 1000*((clumpedConc/unsub) / (stochastic/unsub) -1)
                ID = '/'.join(clumped)

                newRow = {'atom ID': "frag " + fragment + " " + ID, 'element': ID, 'Weighted Deltas': delta}
                Output = Output.append(newRow, ignore_index = True)

    return Output
                     