#A set of functions to deal with unresolved heterogeneous sites. These allow one to specify a list of heterogeneous sites
#to combine, then solves the system for these unresolved sites. It takes the Percent molar abundance from the solved sites
#and fills it in in the molecularDf. 

#The notation introduced here is preserved for the rest of the unresolved site modules. In particular, we refer to a 
#"molecularDf" as the pandas dataframe containing only homogeneous sites (i.e. each row contains only 1 element) while a
#"condensedDf" is a pandas dataframe with some heterogeneous sites (i.e. a combined C/O site). 

import M1Module as M1
import numpy as np

def combineSites(toCombine, molecularDf, completeMeasurementDict):
    '''
    Given a list of atomIds in a dataframe, combines these into one; this will allow us to check if a restricted
    version of the system is constrained.
    
    toCombine: A list of sites to combine, i.e. ['C-carbonyl','O-4']. If one has to create multiple sets of heterogeneous 
    sites, run this function multiple times. 
    
    '''
    fragmentList = completeMeasurementDict['Fragment List']
    
    combinedDf = molecularDf.copy()
    combineStr = ''

    elementStr = ''

    for atomID in toCombine:
        combineStr += atomID
        combineStr += '/'

        index = list(molecularDf['atom ID']).index(atomID)

        elementStr += molecularDf['element'][index]
        elementStr += '/'

        combinedDf.drop(index, inplace=True)

    toAdd = {'atom ID': combineStr[:-1], 'element': elementStr[:-1], 'Stoich': 1, 'Ref Deltas': 0,
            'Equivalence':1}

    for fragment in fragmentList:
        value = False
        for atomID in toCombine:
            index = list(molecularDf['atom ID']).index(atomID)
            newValue = molecularDf[fragment][index]

            if value != False and newValue != value:
                print("attempting to combine sites that fragment differently, rethink your approach")
            else:
                value = newValue

        toAdd[fragment] = value

    combinedDf = combinedDf.append(toAdd, ignore_index = True)
    
    return combinedDf

def moveFiToMolecularDataFrame(molecularDf, condensedDf):
    '''
    Given a molecular dataframe with homogeneous sites only and a condensed dataframe that combines heterogeneous
    sites and has calculated Fi successfully, pulls the successful Fi calculation into the molecular Df, 
    tracking which heterogeneous sites depend on each other. When condensed sites are split their 
    Fi is split equally.
    '''

    fullAtomID = list(molecularDf['atom ID'])
    count = 0
    equivalenceList = [0] * len(fullAtomID)
    fAList = [0] * len(fullAtomID)
    stoichList = list(molecularDf['Stoich'])

    for row, value in condensedDf.iterrows():

        IDList = value['atom ID'].split('/')
        if len(IDList) > 1:
            count += 1
            equivalenceID = str(count) 
        else: 
            equivalenceID = '0'

        for ID in IDList:
            #Pull out index and update equivalence
            index = fullAtomID.index(ID)
            equivalenceList[index] = equivalenceID

            #If this site is not condensed, pull information directly. If it is condensed, split it and determine
            #the fi at the site either by splitting equally or accoridng to a rule
            if len(IDList) > 1:
                fAList[index] = value['Fractional Abundance From Measurement'] / (len(IDList)) 

            else: 
                fAList[index] = value['Fractional Abundance From Measurement'] * value['Stoich']

    molecularDf['Equivalence'] = equivalenceList
    molecularDf['Fractional Abundance From Measurement'] = fAList / molecularDf['Stoich']
    
    return molecularDf

def solveCondensedAndUpdateMolecularDf(condensedDf, molecularDf, completeMeasurementDict, notResolved = []):
    '''
    Given a condensed dataframe, attempts to solve the system. If solved successfully, passes that solution
    to the molecular dataFrame. Doesn't return anything, but updates the condensed and molecular dataframes in place. 
    '''
    #Pull relevant info about measurement
    sampleMeasurement = completeMeasurementDict['Sample']
    bulkMeasurementList = completeMeasurementDict['Bulk Sample']
    fragmentList = completeMeasurementDict['Fragment List']

    #try to solve combined dataframe
    composition, measurement = M1.pullOutCompositionAndMeasurement(condensedDf,sampleMeasurement, fragmentList, notResolved = notResolved)
    compositionUncertainty, uncertainty = M1.pullOutCompositionAndMeasurement(condensedDf,sampleMeasurement,fragmentList, notResolved = notResolved, uncertainty=True)

    solved = M1.checkIfNumpyLstSqFails(composition, measurement, condensedDf)

    #If I did solve it, move that information to the molecular DataFrame
    if solved:
        solution = np.linalg.lstsq(composition, measurement, rcond=0)

        condensedDf['Fractional Abundance From Measurement'] = solution[0]

        moveFiToMolecularDataFrame(molecularDf, condensedDf)
    
    return None