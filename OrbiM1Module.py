import M1EAModule as M1
import pandas as pd
import numpy as np
import perturbMeasurements as perturb
import basicDeltaOperations as op
import scipy.optimize as optimize

def getDeltas(bulkOrbiDict):
    '''
    Given Orbitrap Bulk measurements, turns them into delta values. 
    
    Works for stochastic distribution.
    '''
    output = {}
    for key, value in bulkOrbiDict['Sample'].items():
        element = key
        ratio = value['Measurement'] / basicInfo['element'].count(element)
        delta = op.ratioToDelta(element,ratio)
        output[element] = {'ratio':ratio,'delta':delta}
    
    return output
    
def computeM1Vector(M1Dict, dataFrame, notResolved = []):
    '''
    Solves the M1 percent abundance vector from M1 data
    '''  
    #Pull relevant info about measurement
    sampleMeasurement = M1Dict['Sample']
    fragmentList = M1Dict['Fragment List']

    #put input into a workable form
    composition, measurement = M1.pullOutCompositionAndMeasurement(dataFrame,sampleMeasurement, fragmentList, notResolved = notResolved)
    compositionUncertainty, uncertainty = M1.pullOutCompositionAndMeasurement(dataFrame,sampleMeasurement, fragmentList, notResolved = notResolved, uncertainty=True)

    #Check to see if the system is constrained
    solved = M1.checkIfNumpyLstSqFails(composition, measurement, dataFrame)

    sol = np.linalg.lstsq(composition,measurement,rcond=0)
    return sol[0]

def pullOutM1AbundanceByElement(M1Vector, element, basicInfo):
    '''
    Given an M1 percent abundance vector, and an element, pulls out the percent abundances corresponding to that
    element and puts them into a new vector
    '''
    elementSpecific = []
    for index in range(len(basicInfo['element'])):
        if basicInfo['element'][index] == element:
            #Add in stoichiometry here
            elementSpecific.append(M1Vector[index]*basicInfo['Stoich'][index])
            
    return elementSpecific
            
def normalizeVector(vector):
    '''
    Normalizes a vector so it sums to 1
    '''
    v = np.asarray(vector)
    v /= v.sum()
    
    return v

def computeSingleElementOValues(M1Vector, element, basicInfo,bulkOrbiDict):
    '''
    Given an M1 percent abundance vector and an Orbitrap Bulk measurement, computes the "O" values for each site.
    Then adds them to a dictionary. 
    '''
    elementSpecific = pullOutM1AbundanceByElement(M1Vector, element, basicInfo)
    x = normalizeVector(elementSpecific)
    x *= bulkOrbiDict['Sample'][element]['Measurement']
    
    return x

def computeSingleElementOValuesDict(M1Vector, element, basicInfo, bulkOrbiDict, dictionary = {}):
    '''
    computeSingleElementOValues with dictionary output
    '''
    x = computeSingleElementOValues(M1Vector, element, basicInfo, bulkOrbiDict)
    count = 0
    for index in range(len(basicInfo['element'])):
        if basicInfo['element'][index] == element:
            ID = basicInfo['atom ID'][index]
            dictionary[ID] = x[count]
            count += 1
    
    return dictionary
    
def computeOValues(basicInfo, bulkOrbiDict, M1Dict):
    '''
    Computes all O values based on a bulk Orbitrap measurement and an M1 measurement
    '''
    df = pd.DataFrame.from_dict(basicInfo)
    v = computeM1Vector(M1Dict, df)
    ODict = {}
    for element in set(basicInfo['element']):
        computeSingleElementOValuesDict(v, element, basicInfo, bulkOrbiDict, dictionary = ODict)
    return ODict

def defineElementList(basicInfo):
    '''
    Element list must be ordered; for this reason, sites must be in blocks (i.e. you can't write a CSV with 
    C-alpha, H-alpha, C-beta).
    '''
    elementList = []
    for element in basicInfo:
        if element not in elementList:
            elementList.append(element)
    return elementList

def perturbBulkOrbiDict(bulkOrbiDict):
    '''
    Perturbs the measurements in bulkOrbiDict by their uncertainty
    
    A little slower than not using a dictionary, but not much (~30 us vs ~24 us for direct on an array). The benefit
    is functions can work on either the perturbed or normal version. 
    '''
    perturbedOrbiDict = {'Sample':{}}
    for key, value in bulkOrbiDict['Sample'].items():
        perturbedOrbiDict['Sample'][key] = {'Measurement':0}
        perturbedOrbiDict['Sample'][key]['Measurement'] = np.random.normal(value['Measurement'],value['Uncertainty'])
        
    return perturbedOrbiDict

def computeOValuesWithUncertainty(inputFile,notResolved = [], N=1000):
    '''
    Repeats some parts of earlier functions, to improve computational time through the loop. 
    '''
    basicInfo = inputFile['basicInfo']
    M1Dict = inputFile['M1Dict']
    bulkOrbiDict = inputFile['bulkOrbiDict']
    
    dataFrame = pd.DataFrame.from_dict(basicInfo)
    results = {'fractionalAbundance':[],'OValues':[],'ratios':[],'deltas':[]}

    #START computeM1Vector
    sampleMeasurement = M1Dict['Sample']
    fragmentList = M1Dict['Fragment List']
    elements = basicInfo['element']
    elementList = defineElementList(basicInfo['element'])
    #Add 1 because elementSet comes from O measurements and they don't include unsub
    numIsotopes = len(elementList) + 1
    numFragments = len(fragmentList)

    #put input into a workable form
    composition, measurement = M1.pullOutCompositionAndMeasurement(dataFrame,sampleMeasurement, fragmentList, notResolved = notResolved)
    compositionUncertainty, uncertainty = M1.pullOutCompositionAndMeasurement(dataFrame,sampleMeasurement, fragmentList, notResolved = notResolved, uncertainty=True)

    #Check to see if the system is constrained
    solved = M1.checkIfNumpyLstSqFails(composition, measurement, dataFrame)

    for iteration in range(0,N):
        #perturb fractional abundances
        testArr =  perturb.perturbM1Measurement(measurement, uncertainty, numIsotopes, numFragments)

        sol = np.linalg.lstsq(composition,testArr,rcond=0)
        #END computeM1Vector
        #perturb bulk Orbitrap measurements
        perturbed = perturbBulkOrbiDict(bulkOrbiDict)
        
        thisRun = []
        for element in elementList:   
            v = computeSingleElementOValues(sol[0], element, basicInfo, perturbed)
            thisRun += list(v)

        ratios = np.array(thisRun) / np.array(basicInfo['Stoich'])
        deltas = [op.ratioToDelta(x,y) for x, y in zip(elements, ratios)]

        results['fractionalAbundance'].append(sol[0])
        results['OValues'].append(thisRun)
        results['ratios'].append(ratios)
        results['deltas'].append(deltas)
        
    return results

def processOrbiM1Results(dataFrame, results):
    fA = np.mean(np.array(results['fractionalAbundance']),axis=0)
    fAStd = np.std(np.array(results['fractionalAbundance']),axis=0)

    O = np.mean(np.array(results['OValues']),axis=0)
    OStd = np.std(np.array(results['OValues']),axis=0)

    ratios = np.mean(np.array(results['ratios']),axis=0)
    ratioStd = np.std(np.array(results['ratios']),axis=0)

    deltas = np.mean(np.array(results['deltas']),axis=0)
    deltasStd = np.std(np.array(results['deltas']),axis=0)

    dataFrame['Fractional Abundance'] = fA
    dataFrame['Fractional Abundance Error'] = fAStd

    dataFrame['O Values'] = O
    dataFrame['O Error'] = OStd

    dataFrame['Ratios'] = ratios
    dataFrame['Ratios Error'] = ratioStd

    dataFrame['Deltas'] = deltas
    dataFrame['Deltas Error'] = deltasStd
    
    return dataFrame

def addBulkToOutputDataFrame(basicInfo,dataFrame):
    '''
    Given the culled output dataFrame, calculates and adds bulk values
    '''
    
    elementList = defineElementList(basicInfo['element'])
    for element in elementList:
        deltas = dataFrame[dataFrame['element'] == element]['Weighted Deltas'].values
        stoich = dataFrame[dataFrame['element'] == element]['Stoich'].values
        bulk = np.dot(deltas, stoich) / stoich.sum()
        ratio = op.concentrationToRatio(op.deltaToConcentration(element, bulk))
        if element == 'O':
            elementStr = '17O'
        elif element == 'S':
            elementStr = '33S'
        else:
            elementStr = element
        newRow = {'atom ID': 'bulk ' + elementStr,'element':elementStr,'Ratios':ratio, 'Stoich':stoich.sum(),'Weighted Deltas':bulk}
        dataFrame = dataFrame.append(newRow,ignore_index = True)
        
    return dataFrame
        
def processOrbiM2Plus(bulkOrbiM2Plus, dataFrame):
    '''
    dataFrame from basicInfo
    '''
    output = {}
    for key, value in bulkOrbiM2Plus['Sample'].items():
        element = op.setElement(key)
        stoich = dataFrame[dataFrame['element'] == element]['Stoich'].sum()
        delta = op.ratioToDelta(key,value['Measurement'] / stoich)
        output[key] = (delta)
    
    return output

def addM2PlusDeltasToOutput(bulkOrbiM2Plus, dataFrame):
    
    M2PlusDeltas = processOrbiM2Plus(bulkOrbiM2Plus, dataFrame)
    for key, value in M2PlusDeltas.items():
        element = op.setElement(key)
        stoich = dataFrame[dataFrame['element'] == element]['Stoich'].sum()
        ratio = bulkOrbiM2Plus['Sample'][key]['Measurement'] / stoich
        newRow = {'atom ID': 'bulk ' + key,'element':key,'Ratios':ratio, 'Stoich':stoich,'Weighted Deltas':value}
        dataFrame = dataFrame.append(newRow,ignore_index = True)
        
    return dataFrame

def defineAnamolyDict(clumpIndices, clumpAnamolies):
    '''
    The "AnamolyDict" is a data object that ties specific site indices to a clumped anamoly of some amount. The
    anamoly is defined as the deviation in concentration for this site relative to the stochastic distribution.
    
    Inputs:
        clumpIndices: A list of lists, where each interior list includes the indices of the substitutions for 
        a given clump. Ex: [[0,1],[0,2]] where we are considering two clumped anamolies, one at sites 0 and 1 
        and one at sites 0 and 2. 
        
        clumpAnamolies: A list with the same length as clumpIndices. The numerical value, in concentration 
        space, of the anamolies. Ex: [0.00001,-0.000005] where the anamoly at 0,1 is 0.00001 above the stochastic
        distribution and the anamoly at 0,2 is -0.000005 below the stochastic distribution
    
    Outputs:
        A dictionary tying the clumpIndices to the net clumpAnamolies. Ex: {'0':0.000005,'1':0.00001,'2':-0.000005}
    '''
    anamolyDict = {}
    for i, anamoly in enumerate(clumpIndices):
        siteIndex1 = anamoly[0]
        siteIndex2 = anamoly[1]
        
        if str(siteIndex1) not in anamolyDict:
            anamolyDict[str(siteIndex1)] = clumpAnamolies[i]
        else:
            anamolyDict[str(siteIndex1)] += clumpAnamolies[i]
            
        if str(siteIndex2) not in anamolyDict:
            anamolyDict[str(siteIndex2)] = clumpAnamolies[i]
        else:
            anamolyDict[str(siteIndex2)] += clumpAnamolies[i]
            
    return anamolyDict

def OValuesToRValues(dataFrame, targetArray, clumpDict):
    '''
    Computes R Values from O Values and clumping information, i.e. does not follow the stochastic distribution.
    '''
    #This part of the function takes our information into a form the optimizer can understand. The optimizer
    #must take a one-dimensional array. We here vary two types of parameters--the concentration at any site, and
    #the clumped anamoly at specified sites. We begin by making a reasonable "guess" about the values of these
    #parameters, and feed them into a combined array. I.e. the array size is M + N, where M is the number of sites
    #and N is the number of clumped anamolies we're allowing. 

    #Preliminary assignments
    elements = dataFrame['element'].values
    numSites = len(elements)
    OValues = dataFrame['O Values'].values

    #start with a guess for concentration, assuming the stochastic distribution, i.e. O Values = Ratios.
    RatioGuess = [op.ratioToDelta(x, y) for x, y in zip(elements, OValues)]
    concGuess = [op.deltaToConcentration(x, y)[1] for x, y in zip(elements, RatioGuess)]

    #Clumped Information
    clumpGuess = []
    clumpedIndices = []
    for key, value in clumpDict.items():
        clumpGuess.append(value['amount'])
        clumpedIndices += value['indices']

    #Starting array
    startingGuess = concGuess + clumpGuess
    
    #Next we define a function to minimize
    def toMinimize(inputList):
    
        output = []
        M1 = np.array(inputList[:numSites])
        clumpAnamolies = np.array(inputList[numSites:])

        M0 = np.array([op.hFunction(x, y) for x, y in zip(elements, M1)])

        stochUnsub = M0.prod()
        Unsub = stochUnsub + clumpAnamolies.sum()

        #break this off into its own function
        anamolyDict = defineAnamolyDict(clumpedIndices, clumpAnamolies)

        #Now, compute O Values for singly-substituted sites
        for index in range(len(M1)):
            siteConc = stochUnsub * M1[index] / M0[index]
            if str(index) in anamolyDict:
                siteConc -= anamolyDict[str(index)]

            OValue = siteConc / Unsub
            output.append(OValue)

        #compute Ratios for multiply-substituted isotopologue
        for i, anamoly in enumerate(clumpedIndices): 
            siteIndex1 = anamoly[0]
            siteIndex2 = anamoly[1]

            isotopologueConc = stochUnsub * M1[siteIndex1] / M0[siteIndex1] * M1[siteIndex2] / M0[siteIndex2]

            isotopologueConc += clumpAnamolies[i]

            RValue = isotopologueConc / Unsub
            output.append(RValue)

        outputArray = np.array(output)    

        Objective = 1000*((outputArray - targetArray)**2).sum()

        return Objective
    
    #we then minimize this function
    sol = optimize.minimize(toMinimize, startingGuess ,method = 'Powell')
    
    M1 = sol['x'][:numSites]
    M0 = np.array([op.hFunction(x, y) for x, y in zip(elements, M1)])
    RValues = M1 / M0
    deltas = [op.ratioToDelta(x, y) for x, y in zip(elements, RValues)]
    
    dataFrame['Ratios'] = RValues
    dataFrame['Ratios Error'] = 'N/A'
    dataFrame['Deltas'] = deltas
    dataFrame['Deltas Error'] = 'N/A'
    
    return sol['x']