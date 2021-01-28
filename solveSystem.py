import numpy as np
import pandas as pd
import sympy as sy

import basicDeltaOperations as op

def GJElim(Matrix, augMatrix = False, store = False):
    M = Matrix.copy()
    rows, cols = M.shape

    r = 0
    c = 0
    
    if augMatrix == True:
        colLimit = cols - 1
    else:
        colLimit = cols
        
    rank = 0
    storage = []
    while r < rows and c < colLimit:
        if store:
            storage.append(M.copy())
        #If there is a nonzero entry in the column, then pivot and eliminate. 
        if True in (M[r:,c]!=0):
            pivotRow = (M[r:,c]!=0).argmax(axis=0) + r
            rank += 1

            M[[r, pivotRow]] = M[[pivotRow, r]]

            M[r] = M[r]/ M[r,c]

            for i in range(1,rows-r):
                M[r+i] -= (M[r+i,c]/M[r,c] * M[r])

            for j in range(0,r):
                M[j] -= M[j,c]/M[r,c] * M[r]
                
            r += 1

        c += 1

    if store:
        storage.append(M.copy())
        
    return M, rank, storage

def computeMNUValues(MNSolution, key, df, applyUMN = True):
    '''
    Given an MN U value, update the MN output dataframe to include clumped and site-specific delta values
    
    Key should be "MN", i.e. "M1".
    '''
    if applyUMN:
        MNSolution['U Values'] = MNSolution[key + ' Percent Abundance'] * MNSolution["U" + key]
    
    #calculate clumped deltas
    clumpedDeltas = [1000*(x/y-1) for x, y in zip(MNSolution['U Values'].values, MNSolution['Stochastic U'].values)]
    clumpedCulled = []
    for i in range(len(clumpedDeltas)):
        if '|' in MNSolution.index[i]:
            if np.abs(clumpedDeltas[i]) < 10**(-10):
                clumpedCulled.append(0)
            else:
                clumpedCulled.append(clumpedDeltas[i])
        else:
            clumpedCulled.append('N/A')
      
    #calculate site specific deltas
    deltas = []
    for i, v in MNSolution.iterrows():
        if '|' not in i:
            n = 0
            contributingAtoms = i.split(' & ')
            for atom in contributingAtoms:
                ID = atom.split(' ')[1]
                #df.index gives atom IDs, .index() function returns associated index
                indexOfId = list(df.index).index(ID)
                n += df['Number'][indexOfId]


            siteSpecificR = v['U Values'] / n
            delta = op.ratioToDelta(v['Composition'],siteSpecificR)
            deltas.append(delta)
            
        else:
            deltas.append('N/A')

    MNSolution['Deltas'] = deltas
    MNSolution['Clumped Deltas'] = clumpedCulled
    
    return MNSolution

def checkSolutionIsotopologues(solve, Isotopologues, massKey, fullSolve = False):
    '''
    Given a solution to an augmented matrix which associates isotopologues with their MN Percent abundance,
    recovers the identity of each isotopologue from its column in the augmented matrix. Computes the stochastic
    abundance of each constraint, as well as their compositions and number. 
    
    Inputs:
        solve: The output of GJElim
        Isotopologues: A dataFrame containing the MN population of Isotopologues
        massKey: 'M1', 'M2', etc. 
        fullSolve: True if we include multiple fragments, False otherwise
        
    Outputs:
        a Pandas dataFrame
    '''
    #Take everything but the final column, which is just the answer
    solution = solve[0][:,:-1]
    rank = solve[1]
    
    uniqueAnswers = []
    stochasticValues = []
    composition = []
    number = []
    if fullSolve == False:
        MatrixRows = []

    for i in range(len(solution)):
        if fullSolve == False:
            MatrixRows.append(solution[i])
        stoch = 0
        c = None

        if i >= rank:
            break

        rowIsotopologues = []
        n = 0

        #The solutions *should* all be integer values...if they are not, something weird is happening.
        for j in range(len(solution[i])):
            if solution[i][j] not in list(range(-15,15)):
                print("WARNING: Something unanticipated is going on with row reduction. You need to check the matrix.")
                print(solution[i][j])
                   
            if solution[i][j] > 0:
                string = ""
            elif solution[i][j] < 0:
                string = "MINUS "           
            
            if solution[i][j] != 0:
                #Take the int to simplify things
                sol = int(solution[i][j])
                n += 1

                if sol != 1:
                    rowIsotopologues.append(string + str(sol) + " " + Isotopologues['Precise Identity'][j])
                else:
                    rowIsotopologues.append(Isotopologues['Precise Identity'][j])

                stoch += sol*Isotopologues['Stochastic U'][j]

                if c == None:
                    c = Isotopologues['Composition'][j]
                elif c != Isotopologues['Composition'][j]:
                    c = c + " & " + Isotopologues['Composition'][j]

        uniqueAnswers.append(rowIsotopologues)
        stochasticValues.append(stoch)
        composition.append(c)
        number.append(n)

    #take the measured values
    values = solve[0][:rank,-1]

    condensed = [' & '.join(x) for x in uniqueAnswers]

    #output as dataFrame
    output = {}

    if fullSolve == False:
        output['Matrix Row'] = MatrixRows
        output[massKey +' Percent Abundance'] = values
        
    elif fullSolve == True:
        output['U Values'] = values

    output['Stochastic U'] = stochasticValues
    output['Composition'] = composition
    output['Number'] = number
    
    dfOutput = pd.DataFrame.from_dict(output)
    dfOutput.index = condensed
    
    return dfOutput

