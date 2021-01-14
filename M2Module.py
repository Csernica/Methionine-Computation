import numpy as np
import pandas as pd
import sympy as sy
import M1EAModule as M1
import itertools
import ClumpingModule as clump
import basicDeltaOperations as op
import copy

def getClumpedIsotopologues(atomIDList, stoich):
    '''
    Returns vector representations of Clumped M1 isotopologues, where i.e. [1, 0, 1, 0] indicates a clump with an M1 substitution
    at indices 0 and 2 and [2, 0, 0, 0] indicates a clump with two substitutions at index 0. 
    '''
    clumpedIsotopologues = []
    clumpedStoichList = []
    
    for item in itertools.combinations((atomIDList),2):
        row = [0] * len(atomIDList)
        index1 = atomIDList.index(item[0])
        index2 = atomIDList.index(item[1])

        row[index1] = 1
        row[index2] = 1
        clumpedStoich = stoich[index1] * stoich[index2]

        clumpedIsotopologues.append(row)
        clumpedStoichList.append(clumpedStoich)

        #Add in clumps with multiple substitutions at the same site
        #Add in the first time you pass index1
        if stoich[index1] > 1 and index2 == index1 + 1:
            row = [0] * len(atomIDList)
            row[index1] = 2
            clumpedIsotopologues.append(row)
            n = stoich[index1] - 1
            number = n * (n+1) / 2
            clumpedStoichList.append(number)

        #and just in case the double sub is the last one
        if stoich[index2] > 1 and index2 == len(atomIDList):
            row = [0] * len(atomIDList)
            row[index2] = 2
            clumpedIsotopologues.append(row)
            n = stoich[index2] - 1
            number = n * (n+1) / 2
            clumpedStoichList.append(number)  
        
    return clumpedIsotopologues, clumpedStoichList
    
def getSingleM2Isotopologues(elementList, stoich):
    '''
    Returns vector representation of singly substited M2 isotopologues, where [0, 0, -1, 0] indicates an M2 substitution at index
    2. We use '-1' to show this to avoid ambiguity with clumped substitutions at the same M1 site (for example, if a site has
    stoichiometry = 2). 
    '''
    singleM2Isotopologues = []
    singleM2Stoich = []
    for index in range(len(elementList)):
        row = [0] * len(elementList)
        if elementList[index] == 'O' or elementList[index] == 'S':
            row[index] = -1

            singleM2Isotopologues.append(row)
            singleM2Stoich.append(stoich[index])
            
    return singleM2Isotopologues, singleM2Stoich

def defineM2VectorRepr(atomIDList, elementList, stoich):
    '''
    Defines a vector representation of the isotopologues introduced to the Orbitrap by a M2 experiment. Clumped M1 isotopologues
    are represented as [1, 0, 1, 0] or [2, 0, 0, 0] for clumps at different or the same sites, respectively, while single M2
    isotopologues are represented as [0, 0, -1, 0]. 
    
    Does NOT yet work for multiple stoichiometries. I decided to continue on so I don't spend a day building the edge cases 
    and realize I have to throw out the whole approach. 
    '''
    M2Isotopologues = {'Matrix':[],'Clumped Stoich':[]}
    clumped = getClumpedIsotopologues(atomIDList, stoich)
    M2 = getSingleM2Isotopologues(elementList, stoich)
    
    M2Isotopologues['Matrix'] += clumped[0]
    M2Isotopologues['Matrix'] += M2[0]
    
    M2Isotopologues['Clumped Stoich'] += clumped[1]
    M2Isotopologues['Clumped Stoich'] += M2[1]
    
    return M2Isotopologues
    
def getOneSubstitution(number, element, double = False):
    '''
    A helper function, which ties the vector representation of substitutions to a string indicating which 
    substitutions exist
    
    Fails in edge case where I have 13C/13C and fragment such that I see < 1. Double deals with this.
    '''        
    if number == 0:
        return ''
    
    if double == True:
        if element == 'C':
            return '13C/13C'
        if element == 'N':
            return '15N/15N'
        if element == 'O':
            return '17O/17O'
        if element == 'S':
            return '33S/33S'
        if element == 'H':
            return 'D/D'
  
    if element == 'C':
        if 0 <= number <= 1:
            return '13C'
        if 1 < number <= 2:
            return '13C/13C'
        
    if element == 'N':
        if 0 <= number <= 1:
            return '15N'
        if 1 < number <= 2:
            return '15N/15N'
        
    if element == 'H':
        if 0 <= number <= 1:
            return 'D'
        if 1 < number <= 2:
            return 'D/D'
        
    if element == 'O':
        if 0 <= number <= 1:
            return '17O'
        if 1 < number <= 2:
            return '17O/17O'
        if -1<=  number < 0:
            return '18O'
        
    if element == 'S':
        if 0 <= number <= 1:
            return '33S'
        if 1 < number <= 2:
            return '33S/33S'
        if -1 <= number < 0:
            return '34S'
    
    else:
        print("I don't know how to deal with " + element)
        print("check " + element + str(number)) 
        return None

def getSubsAndLocations(MatrixRep, atomID, elements, dictOutput = False):
    '''
    Takes a vector representation of substitutions and returns i.e. "13C C-1/15N N-2", a precise description of which 
    substitutions occur where for that vector
    
    dictOutput instead returns a dictionary where each precise string is keyed to each vector representation. This is 
    a useful data format for determining stochastic distributions based on the precise strings. 
    '''
    PreciseStrings = []
    
    if dictOutput:
        preciseStringsDict = {}
        
    for vector in MatrixRep:
        l = [getOneSubstitution(x, y) for x, y in zip(vector, elements)]
        Precise = [x + " " + y for x, y in zip(l, atomID) if x != '']
        output = '   |   '.join(Precise)
        PreciseStrings.append(output)
        
        if dictOutput:
            preciseStringsDict[output] = vector
    
    if dictOutput:
        return preciseStringsDict
    
    return PreciseStrings
    
def getSubstitutions(vectorRep, elements, double = False):
    '''
    Takes a vector representing the substitutions present in an isotopologue and a list of elements and outputs
    which substitutions are present
    '''
    l = [getOneSubstitution(x, y, double) for x, y in zip(vectorRep, elements)]
    output = '/'.join(filter(None, l))
    return output

def getSubstitutionStrings(isotopologueMatrix, elements):
    '''
    Where the isotopologue Matrix contains information about the isotopologues in vector format
    '''
    output = []
    for vector in isotopologueMatrix:
        double = False
        if 2 in list(vector):
            double = True
        output.append(getSubstitutions(vector, elements))
    return output

def calcStochasticIsotopologues(molecularDf, M2Dict):
    '''
    Takes a dictionary with information about the M2 Isotopologues, as well as the molecular dataframe, and 
    adds the stochastic abundance of each M2 Isotopologue to the M2 dictionary. Also adds a "type", saying if 
    a particular M2 is a multiply or singly substituted one. 
    '''
    stochastic = []
    types = []
    cList = clump.defineConcentrationList(molecularDf)
    cVector = clump.defineConcVector(cList)

    for isotopologue in M2Dict['Matrix']:

        if 1 in isotopologue:
            subIndices = [site for site, sub in enumerate(isotopologue) if sub == 1]
            conc = clump.doubleSub(cVector, subIndices[0], subIndices[1])
            types.append("Clumped")

        if 2 in isotopologue:
            subIndex = isotopologue.index(2)
            conc = clump.doubleSub(cVector, subIndex, subIndex)
            types.append("Clumped")

        if -1 in isotopologue:
            subIndex = isotopologue.index(-1)
            conc = clump.singleM2Sub(cVector, subIndex)
            types.append("O")

        stochastic.append(conc)

    ORValues = np.array(stochastic) / clump.calcUnsub(cVector)

    M2Dict['Stochastic'] = ORValues
    M2Dict['Types'] = types    
    
    return M2Dict

def defineIsotopologueDict(dataFrame):
    atomID = list(dataFrame['atom ID'])
    elements = list(dataFrame['element'])
    stoich = list(dataFrame['Stoich'])
    
    M2Isotopologues = defineM2VectorRepr(atomID, elements, stoich)
    M2Isotopologues['Composition'] = getSubstitutionStrings(M2Isotopologues['Matrix'], elements)
    M2Isotopologues['Precise Identity'] = getSubsAndLocations(M2Isotopologues['Matrix'], atomID, elements)
    M2Isotopologues = calcStochasticIsotopologues(dataFrame, M2Isotopologues)
    
    return M2Isotopologues
    
def defineCompositionDict(isotopologueStrings):
    '''
    A step in constructing the M2 Composition Matrix. This looks at which substitutions were introduced by a 
    M+2 measurement, and computes all possible observed isotope peaks. For example, if we introduce CO into
    an M+2 measurement, we get 13C/17O and 18O. The possible observed species are Unsub, 13C, 17O, 13C/17O, and 18O,
    depending on what is lost on fragmentation. 
    '''
    compositionDict = {}
    length = len(isotopologueStrings)
    for sub in isotopologueStrings:
        if sub not in compositionDict:
            compositionDict[sub] = [0] * length
        split = sub.split('/')
        if split[0] not in compositionDict:
            compositionDict[split[0]] = [0] * length
        if len(split) > 1:
            if split[1] not in compositionDict:
                compositionDict[split[1]] = [0] * length
    compositionDict['Unsub'] = [0] * length
    
    return compositionDict

def fillCompositionDict(compositionDict, fragmentedStrings, clumpedStoich):
    '''
    This function determines which substitutions actually were observed in a fragment of an M+2 experiment, and 
    indicates which isotopologues are observed in which ion beams.
    '''
    for index in range(len(fragmentedStrings)):
        if fragmentedStrings[index] == '':
            compositionDict['Unsub'][index] += 1 * clumpedStoich[index]
        else:
            compositionDict[fragmentedStrings[index]][index] += 1 * clumpedStoich[index]

    return compositionDict

def defineMeasurementVectorOrder():
    '''
    The order of measured values in the measurement matrix will always be the same: M2 subs, Clumped species, M1 
    subs, and Unsub. This defines that order, and now allows C, N, O, H, and S. More species may be added later.     
    '''
    M2Subs = ['18O','34S']
    M1Subs = ['13C','15N','17O','D','33S']
    M1Combos = list(itertools.combinations(M1Subs, 2))
    M1SelfClumps = [x + '/' + x for x in M1Subs]
    M1Clumps = ['/'.join(x) for x in M1Combos]

    Order = M2Subs + M1SelfClumps + M1Clumps + M1Subs + ['Unsub']
    
    return Order

def fillM2CompositionMatrix(compositionDict, clumpedStoich):
    '''
    Fills the M2 composition matrix from the M2 Composition dictionary. Takes rows out from the dictionary
    in order and adds them to the matrix output. Also outputs a list giving the order.
    '''  
    Order = defineMeasurementVectorOrder()
    output = []
    outputOrder = []
    for tag in Order:
        if tag in compositionDict:
            output.append(compositionDict[tag])
            outputOrder.append(tag)
            
    return output, outputOrder

def constructM2Matrix(isotopologueStrings, fragmentedStrings, clumpedStoich):
    '''
    Given the list of all M2 isotopologues introduced via the M+2 experiment, and a list of which substitutions these
    isotopologues contain following fragmentation, identifies which isotopologues contribute to which ion beams. 
    '''
    c = defineCompositionDict(isotopologueStrings)
    c2 = fillCompositionDict(c, fragmentedStrings, clumpedStoich)
    matrix, order = fillM2CompositionMatrix(c2, clumpedStoich)
    
    return matrix, order

def dataFrameToM2Matrix(dataFrame, inputFile, measurement = True):
    '''
    Generate the "M2 Composition Matrix" from basic information about the molecule and its fragmentation
    '''
    M2Output = {}
    #pull out of dataFrame for easy access
    elements = list(dataFrame['element'].values)
    atomID = list(dataFrame['atom ID'].values)
    stoich = list(dataFrame['Stoich'].values)
    fragmentList = inputFile['M1Dict']['Fragment List']
    measurementDict = inputFile['M2Dict']['Sample']
    
    x = defineIsotopologueDict(dataFrame)
    
    M2Complete = [x['Clumped Stoich']]
    MeasurementVector = [1]
    M2CompleteOrder = ['Closure']
    
    for fragment in fragmentList:
        #Generate the vector representation following fragmentation and which substitutions they correspond to
        fragmented = x['Matrix'] * dataFrame[fragment].values
        fragmentedStrings = getSubstitutionStrings(fragmented, elements)

        #Use this information to generate a M2 composition matrix, and which substitutions they correspond to
        M2Matrix, M2MatrixOrder = constructM2Matrix(x['Composition'], fragmentedStrings, x['Clumped Stoich'])
        
        #Uses this information to add measurements to the measurement vector. Combining all of this into one function makes
        #the function more complex, but means we only have to iterate once.
        if measurement == True:
            fragmentMeasurement = []
            for substitution in M2MatrixOrder:
                try:
                    fragmentMeasurement.append(measurementDict[substitution][fragment]['Measurement'])
                except:
                    print(substitution + " does not appear in your input file, but is theoretically observable. A measurement of 0 has been added to your measurement vector for fragment " + fragment)
                    fragmentMeasurement.append(0)
            
            MeasurementVector += fragmentMeasurement

        M2Complete += M2Matrix
        M2CompleteOrder += M2MatrixOrder

        
    M2Output = {'Isotopologues' : x, 'Composition Matrix' : M2Complete, 'Full Order' : M2CompleteOrder, 'Single Fragment Order' : M2MatrixOrder, 'Measurement' : MeasurementVector}
    
    return M2Output

def checkIfM2Solved(dataFrame, M2Output):
    '''
    Runs np.linalg.lstsq a single time, checking if the M2 system is constrained by determining the rank of the 
    solution. If the system is constrained, prints a message saying so. If not, attempts to determine which 
    isotopologues have not been solved for
    '''
    
    rankTarget = len(M2Output['Isotopologues']['Precise Identity'])
    rank = np.linalg.lstsq(M2Output['Composition Matrix'], M2Output['Measurement'], rcond =0)[2]

    if rank < rankTarget:
        print("Your M2 System is not fully constrained")
        solution = M1.checkHowUnconstrainedSystemFails(M2Output['Composition Matrix'], M2Output['Measurement'])
        unsolved = []

        for item in list(solution[1]):
            if type(item) == sy.numbers.Float:
                continue
            else:
                index = list(solution[1]).index(item)
                unsolved.append(M2Output['Isotopologues']['Precise Identity'][index])

        print("Solution did not constrain")
        print(unsolved)

        return False

    else:
        print("M2 System is constrained, you're good to go")
        return True
    
def rref(B, tol=1e-8, augMatrix = False, debug=False):
    '''
    Adapted from https://gist.github.com/sgsfak/77a1c08ac8a9b0af77393b24e44c9547
    '''
    A = B.copy()
    rows, cols = A.shape
    r = 0
    pivots_pos = []
    row_exchanges = np.arange(rows)
    
    #Don't eliminate final column
    if augMatrix == True:
        rangeTarget = cols - 1
    else: 
        rangeTarget = cols
        
    for c in range(rangeTarget):
        if debug: 
            print("Now at row", r, "and col", c, "with matrix:")
            print(A)

    ## Find the pivot row:
        pivot = np.argmax (np.abs (A[r:rows,c])) + r
        m = np.abs(A[pivot, c])
        
        if debug: 
            print("Found pivot"), m, "in row", pivot
        
        #if m is aproximately 0, say it is really 0 to avoid floating point nonsense
        if m <= tol:
            A[r:rows, c] = np.zeros(rows-r)
            if debug: 
                print("All elements at and below (", r, ",", c, ") are zero.. moving on..")
            
        else:
      ## keep track of bound variables
            pivots_pos.append((r,c))

            if pivot != r:
            ## Swap current row and pivot row
                A[[pivot, r], c:cols] = A[[r, pivot], c:cols]
                row_exchanges[[pivot,r]] = row_exchanges[[r,pivot]]

            if debug: 
                print("Swap row", r, "with row", pivot, "Now:")
                print(A)

            ## Normalize pivot row
            A[r, c:cols] = A[r, c:cols] / A[r, c]
            if debug:
                print("Normalized pivot row")
                print(A)

            ## Eliminate the current column
            v = A[r, c:cols]
            ## Above (before row r):
            if r > 0:
                ridx_above = np.arange(r)
                A[ridx_above, c:cols] = A[ridx_above, c:cols] - np.outer(v, A[ridx_above, c]).T
                
            if debug:
                print("Elimination above performed:")
                print(A)
                
            ## Below (after row r):
            if r < rows-1:
                ridx_below = np.arange(r+1,rows)
                A[ridx_below, c:cols] = A[ridx_below, c:cols] - np.outer(v, A[ridx_below, c]).T
                
                if debug: 
                    print("Elimination below performed:")
                    print(A)
                    
                r += 1
            ## Check if done
            
        if r == rows:
            break;
    return (A, pivots_pos, row_exchanges)

def solveByGaussianElimination(M2Output, debug = False):
    '''
    Better than np.linalg.lstsq for solving underconstrained systems, this method will reveal *which* isotopologues
    are unresolved by explicitly performing a Gauss-Jordan elimination on the augmented matrix. It outputs the 
    results as a pandas dataFrame explicitly tying the sets of isotopologues to their constrained sums. 
    '''
    #Construct the augmented matrix
    comp = np.array(M2Output['Composition Matrix'],dtype=float)
    meas = np.array(M2Output['Measurement'],dtype = float)
    AugMatrix = np.column_stack((comp, meas))
    
    #solve by Gauss Jordan
    solve = rref(AugMatrix, augMatrix = True, debug=debug)
    
    #Take everything but the final column, which is just the answer
    solution = solve[0][:,:-1]
    
    #Check which isotopologues correspond to which measurements in the answer, and explicitly track them
    uniqueAnswers = []
    stochasticValues = []
    types = []
    composition = []
    rank = len(solve[1])
    for i in range(len(solution)):
        stoch = 0
        t = None
        c = None 
        if i >= rank:
            break
        rowIsotopologues = []
        for j in range(len(solution[i])):
            if solution[i][j] == 1:
                rowIsotopologues.append(M2Output['Isotopologues']['Precise Identity'][j])
                
                stoch += M2Output['Isotopologues']['Stochastic'][j]
                
                if t == None:
                    t = M2Output['Isotopologues']['Types'][j]
                elif t != M2Output['Isotopologues']['Types'][j]:
                    t = "Mixed"
                    
                if c == None:
                    c = M2Output['Isotopologues']['Composition'][j]
                elif c != M2Output['Isotopologues']['Composition'][j]:
                    c = 'Mixed'
                    print("may not be able to calculate gamma properly, as some sites are unresolved")
                
        uniqueAnswers.append(rowIsotopologues)
        stochasticValues.append(stoch)
        types.append(t)
        composition.append(c)
        

    #take the measured values
    values = solve[0][:rank,-1]
    
    condensed = [' & '.join(x) for x in uniqueAnswers]

    #output as dataFrame
    output = {}
    output['Types'] = types
    output['Composition'] = composition
    output['M2 Percent Abundance'] = values
    output['Stochastic'] = stochasticValues
    
    dfOutput = pd.DataFrame.from_dict(output)
    dfOutput.index = condensed
    
    return dfOutput, solve

def findM2Gamma(M2Solution, inputFile, key):
    '''
    Calculates the M2 gamma for a given key, i.e. 18O or 13C/13C, a certain M+2/unsub measurement. The key should
    appear in the "Composition" column of M2Solution, and be the same as the key used in the input csv file. 
    '''
    combinedAbundance = M2Solution[M2Solution['Composition'] == key]['M2 Percent Abundance'].values.sum()
    gamma = inputFile['bulkOrbiM2Plus']['Sample'][key]['Measurement'] / combinedAbundance
    M2Solution['M2 gamma'] = gamma
    
    return M2Solution

def computeM2ORValues(M2Solution):
    '''
    Given an M2 gamma, update the M2 output dataframe to include clumped and site-specific delta values
    '''
    M2Solution['OR Values'] = M2Solution['M2 Percent Abundance'] * M2Solution['M2 gamma']
    
    #calculate clumped deltas
    clumpedDeltas = [1000*(x/y-1) for x, y in zip(M2Solution['OR Values'].values, M2Solution['Stochastic'].values)]
    clumpedCulled = []
    for i in range(len(clumpedDeltas)):
        if M2Solution['Types'].values[i] == 'Clumped':
            clumpedCulled.append(clumpedDeltas[i])
        else:
            clumpedCulled.append('N/A')
      
    #calculate site specific deltas
    deltas = []
    for i, v in M2Solution.iterrows():
        if v['Types'] == 'O':
            if "18O" in i:
                delta = op.ratioToDelta('18O',v['OR Values'])
                deltas.append(delta)

        else:
            deltas.append('N/A')

    M2Solution['Deltas'] = deltas
    M2Solution['Clumped Deltas'] = clumpedCulled
    
    return M2Solution

def getPreciseIdentityIndices(preciseID, atomIds):
    '''
    Given a precise ID, i.e. "13C C-1 | 15N N-3 & 13C C-2 | 15N N-3", outputs the indices of the atoms making up
    this ID as a list of lists, with one interior list for each clump. I.e. in this case, the output is 
    [[0,2],[1,2]], where [0,2] corresponds to "13C C-1 | 15N N-3". 
    '''
    contributingClumps = preciseID.split(' & ')
    totalIndices = []
    for clump in contributingClumps:
        #get indices (could make this its own function)
        clumpIndices = []
        longAtomIds = clump.split('   |   ')
        for atom in longAtomIds:
            shortId = atom.split(' ')[1]
            index = atomIds.index(shortId)
            clumpIndices.append(index)
            
        totalIndices.append(clumpIndices)
        
    return totalIndices

def defineClumpDict(M1dataFrame, M2dataFrame, threshold = 5):
    '''
    After one iteration of the M2 solver is run, this searches the M2Solution to find any nonstochastic clumps, 
    by looking at which delta values lie outside of a given threshold. It compiles those clumps and their indices
    into a dictionary, which it feeds into the O value --> R value solver routine. This allows the O-->R routine to
    use information about these clumps to fit the M1 data. 
    Inputs:
        M1dataFrame: The dataframe containing information about M1 fractional abundances
        M2dataFrame: The M2 solution dataframe, which gives M2 percent abundances and clumped delta values
        threshold: A float, giving the amount in delta space which clumps must be over to consider them. 
        
    Outputs: 
        clumpDict. A structure like {'C-1/C-2': {'indices': [0, 1], 'amount': 5.158508633117321e-06},
 'C-1/N-3': {'indices': [0, 2], 'amount': -1.0450462761999357e-05}}, where C-1/C-2 and C-1/N-3 here are clumps that
         had clumped delta values of more than the threshold. 
         RValues: A list of the R Values for the relevant clumps
    '''
    atomIds = list(M1dataFrame['atom ID'].values)
    clumpDict = {}
    RValues = []
    for row, value in M2dataFrame.iterrows():
        #avoid N/A results
        if value['Clumped Deltas'] != 'N/A':
            if abs(value['Clumped Deltas']) > threshold:
                RValues.append(value['OR Values'])
                indices = getPreciseIdentityIndices(row, atomIds)
                
                #This is only an estimate of the clumped anamoly; the clumped anamoly is defined as [A]_actual - [A]_stoch
                #while here we have ([A]/[Unsub])_actual - ([A]/[Unsub])_stoch.
                amountEstimate = value['Stochastic'] - value['OR Values']

                clumpDict[row] = {'indices': indices, 'amount': amountEstimate}
    return clumpDict, RValues

def defineTargetArrayAndClumpDict(M1dataFrame, M2dataFrame, threshold = 5):
    '''
    Defines data to initialize the O-->R ratio solver. These include the clumped dictionary, which tracks which
    clumps are important and what their estimated magnitude is, and the target array, an array of all site specific
    O and relevant clumped R values, which the O-->R solver will fit for. 
    '''
    OValues = list(M1dataFrame['O Values'])
    
    clumpDict, RTargets = defineClumpDict(M1dataFrame, M2dataFrame, threshold = threshold)
    
    target = np.array(OValues + RTargets)
    
    return clumpDict, target

def M1M2Iterator(inputFile, threshold = 5):
    '''
    Solves an M1 system and uses it to solve an M2 system. Then checks where clumps exist over a specified
    threshold delta value. If they do, resolves the M1 system allowing there to be clumps at these sites. Uses
    the new M1 system to solve the M2 system again. Repeats this process until the clumps over a specific threshold
    are constant, then finishes the routine. 
    '''
    #Solve M1
    M1Input = pd.DataFrame.from_dict(inputFile['basicInfo'])
    solution = computeOValuesWithUncertainty(inputFile,notResolved = [], N=1)
    M1dataFrame = processOrbiM1Results(M1Input, solution)
    
    #Solve M2
    M2Output = dataFrameToM2Matrix(M1dataFrame, inputFile, measurement = True)
    sol = solveByGaussianElimination(M2Output)
    M2Solution = sol[0]
    findM2Gamma(M2Solution,inputFile, '18O')
    computeM2ORValues(M2Solution)
    
    oldClumpDict = {}
    newClumpDict = {'no':False}
    
    for i in range(100):
        print('iteration')
        print(i)
        #update M1
        clumpDict, targetArray = defineTargetArrayAndClumpDict(df, M2Solution, threshold = threshold)
        if clumpDict.keys() == oldClumpDict.keys():
            break

        oldClumpDict = copy.deepcopy(clumpDict)

        sol = OValuesToRValues(M1dataFrame, targetArray, clumpDict)

        M2Output = dataFrameToM2Matrix(M1dataFrame, inputFile, measurement = True)
        sol = solveByGaussianElimination(M2Output)
        M2Solution = sol[0]
        findM2Gamma(M2Solution,inputFile, '18O')
        computeM2ORValues(M2Solution)
    
    return M1dataFrame, M2Solution