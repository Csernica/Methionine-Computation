import numpy as np
import basicDeltaOperations as op
import pandas as pd
import M2Module as M2
import ClumpingModule as clump

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

def generateExpectedM2Measurements(dataFrame, inputFile, notResolved = []):
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
    print("CONCENTRATION")
    print(concentration)
    cVector = np.transpose(concentration)
    print("CVECTOR")
    print(cVector)

    #Determine the relative concentrations of the M2 isotopologues
    absConc = findAbsoluteConcentrations(M2Output['Isotopologues']['Matrix'], cVector, stoich)
    absConcStoich = absConc * M2Output['Isotopologues']['Clumped Stoich']
    relConcStoich = absConcStoich / absConcStoich.sum()
    relConc = relConcStoich / M2Output['Isotopologues']['Clumped Stoich']
    print("HEADINGS")
    print(M2Output['Isotopologues']['Precise Identity'])
    print("ABSCONC")
    print(absConc)
    print("REL CONC")
    print(relConc)

    #Apply the M2 composition matrix to the relative concentraiton vector to predict observed measurements
    measurement = np.dot(M2Output['Composition Matrix'], relConc)
    
    return M2Output['Isotopologues']['Precise Identity'], M2Output['Full Order'], relConc, measurement

def calculateOValue(dataFrame, element):
    '''
    Given a dataframe, calculates predicted O Values for a given element. 
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

    OValue = subs / unsub

    return OValue