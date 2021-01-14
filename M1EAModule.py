import sys
import scipy.optimize as optimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sy
import collections
import time
import random
import basicDeltaOperations as op
import calculateGamma
import itertools
import perturbMeasurements as perturb
import unresolvedHelpers as un

def permuteNotResolved(notResolved):
    '''
    Given a list of unresolved substitutions, ['C', 'H', 'O'], generates a list containing these every permutation of 
    these joined as strings: ['C/H/O', 'C/O/H', 'H/C/O', 'H/O/C', 'O/C/H', 'O/H/C']. This allows us to check whether a
    certain unresolved site containing only a subset of these elements (i.e. a CO fragment) should be considered part of 
    the COH group. 
    '''
    permutations = list(itertools.permutations(notResolved))
    permuteNotResolved = ['/'.join(tups) for tups in permutations] 
    return permuteNotResolved

def checkInPermuteNotResolved(isotope, permuteNotResolved):
    '''
    Check if an isotope is in the permuteNotResolved list
    '''
    for string in permuteNotResolved:
        if isotope in string:
            return True
    return False
        
    
def defineListOfIsotopes(dataFrame, notResolved):
    '''
    Given a dataframe and a list of substitutions which are not resolved, constructs an ordered list of the 
    observed species of interest. Each entry in the output corresponds to an observed ion beam for the experiment.
    The order is arbitrary, but there must be an order, to make sure the composition matrix maps onto the measured
    values properly (i. pullOutCompositionAndMeasurement).
    Ex:
    
    notResolved: ['C','O']
    
    output: ['C/O','N','H']
    '''
    combined = '/'.join(notResolved)
    permutations = permuteNotResolved(notResolved)

    isotopeOrder = []
    if combined != '':
        isotopeOrder.append(combined)

    for isotope in dataFrame['element'].values:
        if checkInPermuteNotResolved(isotope, permutations) == False: 
            if isotope not in isotopeOrder:
                isotopeOrder.append(isotope)
    
    return isotopeOrder

def pullOutCompositionAndMeasurement(dataFrame,measurementDict, fragmentList, notResolved = [], uncertainty = False):
    '''
    Take the composition matrix from the pandas dataframe and the measured values from input data to prepare for
    monte carlo. The key role of this function is keeping the composition matrix and measurement vectors in the
    right isotope order
    '''
    #By default, the composition matrix has a separate line for each element. Specifying that two elements are not
    #resolved combines them into a single line. isotopeOrder specifies the lines; each entry in isotopeOrder 
    #corresponds to a measured ion beam (for a single fragment). I.e. it should be ['C/O','N'] if you looked at C, O,
    #and N and couldn't resolve C and O. 
    combined = '/'.join(notResolved)
    isotopeOrder = defineListOfIsotopes(dataFrame, notResolved)

    compositionMatrix = []
    #The first row sets the sum of pM1 = 1
    compositionMatrix.append(list(dataFrame['Stoich'].values))

    #The same function is used to construct the uncertainty vector as well. In this case, the uncertainty of pM1 is 
    #set to 0. Reusing the function ensures the Measurement and Uncertainty data are processed in the same order. 
    Measurement = []
    if uncertainty:
        Measurement.append(0)
    else:
        Measurement.append(1)

    #For each fragment, specifies the fragment vector for each observed element. See the powerpoint. Example: For C2N
    #where the 26 fragment is given by [0,1,1], splits this into 26.C [0,1,0] and 26.N [0,0,1], vectors corresponding
    #to the actual measured quantities

    for fragment in fragmentList:
        for isotopeGroup in isotopeOrder:
            #vector is initially zero
            vector = np.zeros(len(dataFrame['element'].values),dtype = float)

            listForm = isotopeGroup.split('/')
            permutations = permuteNotResolved(listForm)

            for row, value in dataFrame.iterrows():
                #Check if the element is in this group
                if checkInPermuteNotResolved(value['element'], permutations) == True:
                    #if it is, add it to the vector
                    toAdd = value[fragment] * value['Stoich']
                    vector[row] += toAdd

            compositionMatrix.append(list(vector))

            for key in measurementDict.keys():
                if checkInPermuteNotResolved(key, permutations) == True:
                    if uncertainty == False:
                        Measurement.append(measurementDict[key][fragment]['Measurement'])
                    else:
                        Measurement.append(measurementDict[key][fragment]['Uncertainty'])
                        
        #At the end of each fragment, add the unsubstituted composition and measurement
        vector = (1-dataFrame[fragment].values) * dataFrame['Stoich'].values
        compositionMatrix.append(list(vector))
        
        if uncertainty == False:
            Measurement.append(measurementDict['Unsub'][fragment]['Measurement'])
        else:
            Measurement.append(measurementDict['Unsub'][fragment]['Uncertainty'])

    return compositionMatrix, Measurement

def computeUnknowns(Composition, Abundance, Measurement):
    '''
    A general function which takes sympy matrices and attempts to solve the linear system of equations defined
    by CA = M. The key benefits here are 1) One can assign variables in any of C, A, or M, as opposed to the 
    numpy matrix solvers which only allows A to vary (this i.e. allows one to use known abundances to solve for
    unknown composition parameters) and 2) if one is only allowing A to vary, and A is underconstrained, this 
    will tell you which variables are constrained or unconstrained (np.linalg.lstsq can only say the whole system
    is constrained or unconstrained)
    
    '''
    #Determine which variables we will solve for
    variables = []
    for item in Composition:
        if type(item) == sy.symbol.Symbol:
            variables.append(item)
    for item in Abundance:
        if type(item) == sy.symbol.Symbol:
            variables.append(item)
    for item in Measurement:
        if type(item) == sy.symbol.Symbol:
            variables.append(item)
    
    #Generate the system of equations
    eqMatrix = sy.Matrix(np.dot(Composition,Abundance))
    eqList = []
    for row in range(len(eqMatrix)):
        eq = eqMatrix[row] - Measurement[row]
        eqList.append(eq)
    
    #Solve the system of equations and save the solutions
    solutions = sy.solve(eqList,variables,dict=True)
    print(solutions)
        
    replacements = []
    
    if solutions == []:
        print('Sorry, no analytical solution found. This is likely because some variables are overconstrained \nwhile others are underconstrained. You can either look at your problem manually or run the numerical solver.')
        return [[],[],[],[]]
        
    for item in solutions[0].items():
        replacements.append((item[0],item[1]))
        
    #Substitute the solutions into new output matrices    
    CompositionFilled = Composition.subs(replacements)
    AbundanceFilled = Abundance.subs(replacements)
    MeasurementFilled = Measurement.subs(replacements)
    
    solved = True
    #Raise an error if there are still variables, i.e. the system was not solved
    for item in CompositionFilled:
        if type(item) == sy.symbol.Symbol:
            print("Oops, we didn't solve for everything")
            solved = False
    for item in AbundanceFilled:
        if type(item) == sy.symbol.Symbol:
            print("Oops, we didn't solve for everything")
            solved = False
    for item in MeasurementFilled:
        if type(item) == sy.symbol.Symbol:
            print("Oops, we didn't solve for everything")
            solved = False
    
    return CompositionFilled, AbundanceFilled, MeasurementFilled, solved

def checkIfNumpyLstSqFails(Composition, Measurement, dataFrame):
    rankTarget = len(Composition[0])
    rank = np.linalg.lstsq(Composition, Measurement,rcond=0)[2]
    
    if rank < rankTarget:
        print("Your system is not fully constrained")
        solution = checkHowUnconstrainedSystemFails(Composition, Measurement)
        unsolved = []
        atomIDs = dataFrame['atom ID'].values
        
        for item in list(solution[1]):
            if type(item) == sy.numbers.Float:
                continue
            else:
                index = list(solution[1]).index(item)
                unsolved.append(atomIDs[index])

        print("Solution did not constrain")
        print(unsolved)
        
        return False
    
    else:
        print("System is constrained, you're good to go")
        return True 

def checkHowUnconstrainedSystemFails(Composition, Measurement):

    CompositionMatrix = sy.Matrix(Composition)
    MeasurementMatrix = sy.Matrix(Measurement)

    length = len(Composition[0])

    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = sy.symbols('a b c d e f g h i j k l m n o p q r s t u v w x y z')
    variableList = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z]

    #If you have more than 26 variables this will fail; you should explictly initialize more sympy variables
    Abundance = sy.Matrix(variableList[:length])

    sol = computeUnknowns(CompositionMatrix, Abundance, MeasurementMatrix)

    return sol

def pullOutRelevantFi(singleElementIndices, fullFiArray):
    '''
    Given a list giving indices of interest and an array containing the whole molecule fractional abundances 
    pulls out the fractional abundances at the indices of interest to create a condensed array. This allows us 
    to i.e. pull out an array of fractional abundances at only carbon sites to calculate gamma using a carbon 
    bulk measurement
    '''
    shortList = []
    for index in range(len(fullFiArray)):
        if index in singleElementIndices:
            shortList.append(fullFiArray[index])
    shortArray = np.array(shortList)
    
    return shortArray
    
def solveEntireSystemThroughDeltas(dataFrame,M1Dict, EADict, molecularDf = pd.DataFrame(), constraintDict = {}, notResolved = [], N=1000):
    '''
    A function which takes M1 data (in M1 percent abundance space) and EA data (in delta space) and attempts to solve for
    site-specific delta values. Only valid under the stochastic assumption
    
    molecular Df is an option for "unresolved" cases; i.e. where we can't distinguish between two heterogeneous sites
    (maybe a C and an O) but one of them (C) is used to calculate gamma, (as we have a bulk C measurement). Read the 
    unresolved modules to get a better idea of what this does. In this case, the "dataFrame" variable stands in for the
    "condensedDf" in the unresolved site code. 
    '''
    #Pull relevant info about measurement
    sampleMeasurement = M1Dict['Sample']
    bulkMeasurementList = EADict['Sample']
    fragmentList = M1Dict['Fragment List']
    numIsotopes = len(M1Dict['Sample'].keys())
    numFragments = len(fragmentList)
    elements = dataFrame['element'].values
    singleElementIndices = dataFrame[dataFrame['element'] == bulkMeasurementList[0][0]].index.values
    singleElementStoich = dataFrame[dataFrame['element'] == bulkMeasurementList[0][0]]['Stoich'].values
    
    #If we are unresolved, then we include a "molecular" dataframe in addition to the "condensed" dataframe. This uses
    #the molecular Dataframe to introduce information as needed
    if molecularDf.empty == False:
        m = un.condensedToMolecularIndices(molecularDf, dataFrame)
        condensedStoich = dataFrame['Stoich'].values
        elements = molecularDf['element'].values
        molecularStoich = molecularDf['Stoich'].values
        
        singleElementIndices = molecularDf[molecularDf['element'] == bulkMeasurementList[0][0]].index.values
        singleElementStoich = molecularDf[molecularDf['element'] == bulkMeasurementList[0][0]]['Stoich'].values
        
        unresolved = un.defineUnresolvedDict(constraintDict, molecularDf)
        
        sol = un.findUnresolvedFiErrorBounds(molecularDf, M1Dict, constraintDict)
        
        un.updateUnresolved(unresolved, sol, molecularDf)
        
    #put input into a workable form
    composition, measurement = pullOutCompositionAndMeasurement(dataFrame,sampleMeasurement, fragmentList, notResolved = notResolved)
    compositionUncertainty, uncertainty = pullOutCompositionAndMeasurement(dataFrame,sampleMeasurement, fragmentList, notResolved = notResolved, uncertainty=True)
    
    #Check to see if the system is constrained
    solved = checkIfNumpyLstSqFails(composition, measurement, dataFrame)
    
    #Pull out relevant items from dataframe into np.arrays for faster computation. Bulk measurement is as a list
    #to eventually expand to multiple bulk measurements
    
    stoichArray = dataFrame['Stoich'].values
    
    #Used to calculate Gamma, tracks the indices and stoichiometry for only the bulk element of interest
    if bulkMeasurementList == []:
        raise Exception("No bulk measurement provided")

    #Store Results
    results = {'fractionalAbundance':[],'bulkDelta':[],'gamma':[],'ratiosFromGamma':[],'deltas':[]}
    for iteration in range(N):
        #Perturb Fractional Abundances 
        #Uncorrelated Errors
        #testArr = np.random.normal(measurement,uncertainty)
        #Correlated Errors
        testArr =  perturb.perturbM1Measurement(measurement, uncertainty, numIsotopes, numFragments)
            
        #Perturb Bulk Measurement in delta space, and convert to ratio space
        testBulk = np.random.normal(bulkMeasurementList[0][1],bulkMeasurementList[0][2])
        testBulkRatio = op.concentrationToRatio(op.deltaToConcentration(bulkMeasurementList[0][0],testBulk))
        
        #Perturb multiplier
        if molecularDf.empty == False: 
            multiplier = un.defineMultiplier(unresolved, len(molecularStoich))
            
        #Solve Fractional Abundances
        sol = np.linalg.lstsq(composition,testArr,rcond=0)
        molecularFiSingleAtom = sol[0]

        if molecularDf.empty == False: 
            condensedFiSingleAtom = molecularFiSingleAtom.copy()
            molecularFiSingleAtom = un.condensedToMolecularValues(condensedFiSingleAtom, m, multiplier)
        
        testFiSingleAtom = pullOutRelevantFi(singleElementIndices, molecularFiSingleAtom)

        #Solve for gamma
        gamma = calculateGamma.minimizeSingleGammaNumerical(testFiSingleAtom, singleElementStoich, testBulkRatio)

        #calculate new ratios from fi and gamma
        ratiosFromGamma = molecularFiSingleAtom / (gamma)
        deltas = [op.ratioToDelta(x,y) for x, y in zip(elements, ratiosFromGamma)]
        
        results['fractionalAbundance'].append(molecularFiSingleAtom)
        results['bulkDelta'].append(testBulk)
        results['gamma'].append(gamma)
        results['ratiosFromGamma'].append(ratiosFromGamma)
        results['deltas'].append(deltas)
        
        
    return results

def processResults(dataFrame, results):
    '''
    calculates means and error from the results dictionary and updates dataframe
    '''
    fA = np.mean(np.array(results['fractionalAbundance']),axis=0)
    fAStd = np.std(np.array(results['fractionalAbundance']),axis=0)

    bulk = np.mean(np.array(results['bulkDelta']),axis=0)
    bulkStd = np.std(np.array(results['bulkDelta']),axis=0)

    gamma = np.mean(np.array(results['gamma']),axis=0)
    gammaStd = np.std(np.array(results['gamma']),axis=0)

    ratios = np.mean(np.array(results['ratiosFromGamma']),axis=0)
    ratioStd = np.std(np.array(results['ratiosFromGamma']),axis=0)

    deltas = np.mean(np.array(results['deltas']),axis=0)
    deltasStd = np.std(np.array(results['deltas']),axis=0)
    
    dataFrame['Fractional Abundance From Calculation'] = fA
    dataFrame['Fractional Abundance From Calculation Error'] = fAStd

    dataFrame['gamma'] = gamma
    dataFrame['gamma Error'] = gammaStd

    dataFrame['Ratios From Gamma'] = ratios
    dataFrame['Ratios From Gamma Error'] = ratioStd

    dataFrame['Deltas from Gamma'] = deltas
    dataFrame['Deltas From Gamma Error'] = deltasStd
    
    return dataFrame