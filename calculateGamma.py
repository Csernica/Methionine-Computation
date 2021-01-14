import scipy.optimize as optimize
import numpy as np
import pandas as pd
import basicDeltaOperations as op
import sympy as sy

def calculateGammaAnalytic(dataFrame, bulkMeasurementTuple):
    '''
    Deprecated--I find this analytic solution yields the same value in less time than the numerical, and is not conducive to 
    1000+ repititions for propagation of error, so I no longer use it. 
    '''

    REA = op.concentrationToRatio(op.deltaToConcentration(bulkMeasurementTuple[0],bulkMeasurementTuple[1]))
    
    g = sy.symbols('g',real=True)
    
    k = dataFrame[dataFrame['element'] == bulkMeasurementTuple[0]]['Stoich'].values.sum()
    
    expr = 0 
    
    for index, value in dataFrame.iterrows():
        if value['element'] == bulkMeasurementTuple[0]:
            fi = value['Fractional Abundance From Measurement']/ value['Stoich']
            toAdd =  fi/g / (1+fi/g)
            
            toAdd *= value['Stoich']
            
            expr += toAdd
    expr *= 1/k
    
    fullExpr = expr/(1-expr)
    
    sol = sy.solve(fullExpr - REA)
    
    positive = []
    
    for solution in sol:
        if solution >= 0:
            positive.append(solution)
            
    return positive[0]

def minimizeSingleGammaNumerical(fiArray, stoichArray, bulkRatio):
    '''
    Given some number of fractional abundances, stoichiometries, and bulkRatios, solves for gamma
    '''
    k = stoichArray.sum()
    def gammaFunction(gamma):
        '''
        Sets an internal function, to be minimized
        '''
        #Gamma cannot be negative; return a high result to force the solver to consider positive values
        if gamma <= 0:
            return 100

        #construct the bulk expression
        expr = 0 
        for index in range(len(fiArray)):
            fi = fiArray[index]

            toAdd =  fi/gamma / (1+fi/gamma)
            toAdd *=  stoichArray[index]
            expr += toAdd

        expr *= 1/k

        fullExpr = expr/(1-expr)

        residual = 1000*(bulkRatio - fullExpr)**2

        return residual

    #optimize gamma and store result
    sol = optimize.minimize(gammaFunction, 5,method = 'Powell')

    return sol['x'].item()

def calculateGammaNumerical(dataFrame, bulkMeasurementList, N = 1000):
    #Define array to keep track of output gammas
    gammaArray = np.zeros((N))

    #Pull out relevant items from dataframe into np.arrays for faster computation
    singleElementDf = dataFrame[dataFrame['element'] == bulkMeasurementList[0][0]]

    fiArray = (singleElementDf['Fractional Abundance From Measurement'] / singleElementDf['Stoich']).values
    fiUncertaintyArray = (singleElementDf['Fractional Abundance From Measurement Uncertainty'] / np.sqrt(singleElementDf['Stoich'])).values
    stoichArray = singleElementDf['Stoich'].values
    length = len(stoichArray)

    #Start Monte Carlo fit
    for iteration in range(N):

        #Generate fractional abundances and bulk measurements within uncertainty.
        testFi = np.zeros(length)
        testBulk = np.random.normal(bulkMeasurementList[0][1],bulkMeasurementList[0][2])
        testBulkRatio = op.concentrationToRatio(op.deltaToConcentration(bulkMeasurementList[0][0],testBulk))

        for index in range(len(testFi)):
            testFi[index] = np.random.normal(fiArray[index],fiUncertaintyArray[index])
    
        #Run the fit
        gamma = minimizeSingleGammaNumerical(testFi, stoichArray, testBulkRatio)
        
        gammaArray[iteration] = gamma
        
    mean = np.mean(gammaArray, axis = 0)
    std = np.std(gammaArray, axis = 0)


    return mean, std