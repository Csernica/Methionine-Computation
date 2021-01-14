import itertools
import basicDeltaOperations as op
import numpy as np
import scipy.optimize as optimize
import M1EAModule as M1
import calculateGamma

def defineCombinedSiteList(molecularDf):
    '''
    Given a molecular DataFrame with some heterogeneous equivalent sites, tracks these in a list of dictionaries. 
    A helper function to define a useful intermediate. 
    
    Example Output: 
            [{'atom IDs': ['C-carbonyl', 'O-4'],
              'Indices': [0, 3],
              'Stoich': [1.0, 1.0],
              'equivalenceNumber': 1,
              'combinedFi': 0.41510663600000025},
             {'atom IDs': ['C-alpha', 'H-alpha'],
              'Indices': [1, 5],
              'Stoich': [1.0, 2.0],
              'equivalenceNumber': 2,
              'combinedFi': 0.433260305}]
    '''
    CSL = []
    numberOfEquivalences = int(max(set(molecularDf['Equivalence'].values)))
    
    for equivalence in range(0,numberOfEquivalences):
        eqNumber = equivalence + 1
        site = {'atom IDs':[],'Indices':[],'Stoich':[],'equivalenceNumber':eqNumber,'combinedFi':0}
        for row, value in molecularDf.iterrows():
            if int(value['Equivalence']) == eqNumber:
                site['atom IDs'].append(value['atom ID'])
                site['Indices'].append(row)
                site['Stoich'].append(value['Stoich'])
                site['combinedFi'] += value['Fractional Abundance From Measurement'] * value['Stoich']
        
        CSL.append(site)
        
    return CSL

def modifyFiOfUnresolvedSites(fractionalAbundance, combinedSiteList, listOfConstants):
    '''
    Given an array of fractional abundances for a whole molecule, supposing that some sites are individually 
    unconstrained but sum up to a constrained value, generates a new array of fractional abundances. Determines
    the new values by making the unconstrained values to be a fraction of the whole value. The sum of 
    listOfConstants should always be <= 1. The length of listOfConstants is len(coupledIndices)-1. See sample
    calculation in powerpoint.
    
    Inputs:
        fractionalAbundance: np array giving the old fractional abundance
        combinedSiteList: see above
        equivalenceNumber':1,'combinedFi' : 0.257754}]
        listOfConstants: List of lists, one list for each entry in combinedSiteList. The length of each list is 
        equal to the number of atoms IDs - 1. 
    '''
    #make a new object to store updated fi
    newFi = np.array(fractionalAbundance) 
    
    #Set combined abundance and the indices which will vary
    for index in range(len(combinedSiteList)): 
        combinedAbundance = combinedSiteList[index]['combinedFi']   
        coupledIndices = combinedSiteList[index]['Indices']
        stoich = combinedSiteList[index]['Stoich']
        
        siteConstants = listOfConstants[index]
            
        for siteIndex in range(len(siteConstants)):
            newFi[coupledIndices[siteIndex]] = siteConstants[siteIndex] * combinedAbundance
            #newFi /= stoich[siteIndex]
        
        lastIndex = coupledIndices[-1]

        newFi[lastIndex] =(1-np.array(siteConstants).sum()) * combinedAbundance
        #newFi[lastIndex] /= stoich[-1]
    
    return newFi


        
def defineDirectory(combineSiteList):
    '''
    Given the combined site list, defines a list which gives the number of free parameters in each site. I.e, if 
    a site includes C, O, and H, it has 2 free parameters. 
    
    Ex Input:[{'atom IDs': ['C-carbonyl', 'O-4'],
              'Indices': [0, 3],
              'Stoich': [1.0, 1.0],
              'equivalenceNumber': 1,
              'combinedFi': 0.41510663600000025},
             {'atom IDs': ['C-alpha', 'H-alpha'],
              'Indices': [1, 5],
              'Stoich': [1.0, 2.0],
              'equivalenceNumber': 2,
              'combinedFi': 0.433260305}]
              
    Ex Output: [1,1]
    '''
    directory = []
    for entry in combineSiteList:
        directory.append(len(entry['Indices'])-1)
        
    return directory

def getListOfConstants(array, directory):
    '''
    Helper function to fit an input delta value. The powell optimization routine in findUnresolvedFiForSpecificDelta
    takes an array; we want to make it a list of lists, where each list corresponds to a single unresolved site
    and contains a number of values equal to the number of atoms contributing to that site - 1. Directory specifies
    how many elements are in each interior list. 
    
    Ex) 
    array = np.array([0.25,0.5,0.75])
    directory = [2,1]
    outputList = [[0.25, 0.5], [0.75]]
    
    for a molecule with 2 separate unresolved sites, the first containing 3 atoms and the second containing two.
    '''
    outputList = []
    start = 0
    for index in range(len(directory)):
        toAdd = []
        for site in range(start, start + directory[index]):
            toAdd.append(array[site])
        start += directory[index]
        outputList.append(toAdd)
        
    return outputList

def findUnresolvedFiForSpecificDelta(molecularDf, targetDeltaList, bulkMeasurementList):
    '''
    Take a molecular dataframe with one or more unresolved sites and a known bulk Measurement. The user specifies a
    target delta value for one of the atoms in an unresolved site, and calculates the fi for each atom in that
    unresolved site and that particular delta value. 
    
    I.e. suppose we know fi for all but a C and O of a carbonyl group, and we know C+O. This allows us to set the 
    delta value of O to i.e. -100 and calculate what the fi of C and O are in that case. 
    
    Inputs:
        molecularDataFrame: A full molecule dataFrame
        targetDeltaList: A list of tuples, i.e. [('O-5',100),('H-7, 100')] The first is the atom ID and second 
        is the delta value. 
        bulkMeasurementList: [('C',5,0.1)]. Element, value, error. 
        
    Outputs: The complete solution to our optimization, which minimizes a parameter (or list of parameters if more
    than 2 atoms contribute to the site) going between 0 and 1, the fraction each atom contributes to the total 
    fractional abundance of the site. 
    '''
    #Check our approach is valid
    for target in targetDeltaList:
        if int(molecularDf[molecularDf['atom ID'] == target[0]]['Equivalence'].values[0]) == 0:
            print("The atom ID you are targeting has no equivalent sites")

    #Pull information from dataFrame; this is because interacting with the dataframe is slow, so we don't want to 
    #do it 1000 times during the optimization. But all of these variables are just things in the dataFrame
    atomIDList = list(molecularDf['atom ID'])
    singleElementIndices = molecularDf[molecularDf['element'] == bulkMeasurementList[0][0]].index.values
    singleElementStoich = molecularDf[molecularDf['element'] == bulkMeasurementList[0][0]]['Stoich'].values
    elements = molecularDf['element'].values
    stoich = molecularDf['Stoich'].values
    #fullFi has single atom fractional abundances; i.e. multiply by stoich to get the real fi of a site with stoich != 1
    fullFiSingleAtom = molecularDf['Fractional Abundance From Measurement'].values
    fullFiStoich = fullFiSingleAtom * stoich


    #Convert bulk measurement to ratio space
    testBulkRatio = op.concentrationToRatio(op.deltaToConcentration(bulkMeasurementList[0][0],bulkMeasurementList[0][1]))

    #Define other calculation intermediates
    CSL = defineCombinedSiteList(molecularDf)
    directory = defineDirectory(CSL)
    numParameters = np.array(directory).sum()

    #Define function to minimize
    def minimizeHelper(inputArray):
        '''
        Takes a np.array of length equal to the number of equivalent sets of sites. I.e. if C-O of one carboxyl and 
        C-O of another carboxyl arre both distinct from each other, the array is length 2. Currently, the list of 
        list of inputArray is for eventual incorporation of multiple exclusive equivalent sites. 

        The time to run this function is dominated by the minimizeSingleGammaNumerical Function (which itself is a
        minimization routine). 
        '''
        #We take stoichiometry into account here, as the combined fractional abundances in CSL do
        listOfConstants = getListOfConstants(inputArray, directory)
        newFiStoich = modifyFiOfUnresolvedSites(fullFiStoich, CSL, listOfConstants)

        #When we solve for gamma, we want single atom fi again
        newFiSingleAtom = newFiStoich / stoich
        #Solve for gamma, first by pulling out only the fi for the element of interest, then solving
        testFiSingleAtom = M1.pullOutRelevantFi(singleElementIndices, newFiSingleAtom)

        gamma = calculateGamma.minimizeSingleGammaNumerical(testFiSingleAtom, singleElementStoich, testBulkRatio)

        #Calculate delta values
        ratiosFromGamma = newFiSingleAtom / (gamma)
        deltas = [op.ratioToDelta(x,y) for x, y in zip(elements, ratiosFromGamma)]

        objective = 0
        for target in targetDeltaList:
            index = atomIDList.index(target[0])
            objective += 10**9*(deltas[index] - target[1])**2

        return objective

    #minimize difference between target delta and solved delta
    bounds = [(0,1)] * numParameters
    #Ran into issues previously with this not converging in some cases as it decreased too slowly; if this
    #happens, try increasing the constant to multiply objective by
    solution = optimize.minimize(minimizeHelper,np.zeros(numParameters),method='trust-constr',bounds = bounds)

    #save solution; special instructions if only one variable
    if numParameters == 1:
        l = [solution['x'].item()]
        listOfConstants = [l]
    else:
        listOfConstants = getListOfConstants(solution['x'],directory)

    #save Fi
    newFiStoich = modifyFiOfUnresolvedSites(fullFiStoich, CSL, listOfConstants)

    return solution, newFiStoich

def setConstraints(constraintDict):
    '''
    Given a dictionary of boundaries, generates all combinations of boundary points. If there are very many
    unknown boundaries, we should deal with this differently...
    
    Input Ex: {'O-4':(-100,100),'H-alpha':(-100,100)}
    
    Output Ex: [[('O-4', -100), ('H-alpha', -100)]
            [('O-4', -100), ('H-alpha', 100)]
            [('O-4', 100), ('H-alpha', -100)]
            [('O-4', 100), ('H-alpha', 100)]]
    '''
    output = []
    length = len(constraintDict.keys())
    if length > 7:
        print("You should revisit how constraints are set")
    s = set([0,1])
    l = list(itertools.product(s,repeat=length))
    for tup in l:
        targetDeltaList = []
        index = 0
        for key, value in constraintDict.items():
            target = (key, value[tup[index]])
            index += 1
            targetDeltaList.append(target)
        output.append(targetDeltaList)
    return output

def findUnresolvedFiErrorBounds(molecularDf, completeMeasurementDict, constraintDict):
    '''
    Given a molecularDf with some unresolved sites and a dictionary of constraints, dertermines the bounds for
    the relative Abundance Parameter in each unresolved site. 
    
    constraintDict = {'O-4': (-90, 110)}
    '''
    constraintList = setConstraints(constraintDict)
    relAbundanceParameters = []
    for targetDeltaList in constraintList:
        sol = findUnresolvedFiForSpecificDelta(molecularDf, targetDeltaList, completeMeasurementDict['Bulk Sample'])
        relAbundanceParameters.append(sol[0]['x'])
        
    low = min(relAbundanceParameters)
    high = max(relAbundanceParameters)

    return low, high

def condensedToMolecularIndices(molecularDf, condensedDf):
    '''
    Input a molecular and a condensed dataframe. Each atom ID has an index (row number) in both the molecular
    and condensed dataframes. This function defines a list with length of the molecular dataframe. Each entry of
    the list corresponds to an index of the molecular dataframe; i.e., the 0th entry corresponds to the 0th
    atom ID in the molecular dataframe. The value gives the index in the condensed dataframe. I.e., if the 0th
    entry is 2, this says that the 0th element in the molecular dataframe appears in the 2nd row of the condensed
    dataframe. 
    
    Difficult to explain but easy to see in action; I recommend you initialize a dataFrame, condense it using
    combine Sites, then run this function and see what happens. 
    '''
    #Split condensed Df atom IDs into a list of lists; each interior list gives each of the atom IDs at that site
    condensedDfSites = []
    for combinedSite in condensedDf['atom ID'].values:
        x = combinedSite.split('/')
        condensedDfSites.append(x)
    
    #For each atom ID in molecular Df, check where it appears in condensed Df. Atom IDs must be unique!
    condensedToMolecularIndices = []
    for atomID in molecularDf['atom ID'].values:
        for combinedSite in condensedDfSites:
            if atomID in combinedSite:
                condensedToMolecularIndices.append(condensedDfSites.index(combinedSite))
                
    return condensedToMolecularIndices

def condensedToMolecularValues(condensedFi, condensedToMolecularIndices, multiplier = []):
    '''
    Takes a list of fractional abundances from condensed sites and takes them to a list of molecular fractional 
    abundances.
    '''
    if list(multiplier) == []:
        multiplier = np.ones(len(condensedToMolecularIndices))
    
    molecularFiStoich = np.zeros(len(condensedToMolecularIndices))
    for molecularIndex in range(len(condensedToMolecularIndices)):
        molecularFiStoich[molecularIndex] = condensedFi[condensedToMolecularIndices[molecularIndex]] 

    molecularFiStoich *= multiplier
    
    return molecularFiStoich

def defineUnresolvedDict(constraintDict, molecularDf):
    '''
    Given a constraint dict, defines a dictionary specifying which atom IDs will be varied and which indices
    their equivalent sites encompass
    '''
    unresolved = {}
    for key in constraintDict.keys():
        index = list(molecularDf['atom ID']).index(key)
        unresolved[str(index)] = {'indices':[], 'Stoich':[]}
        equivalenceNumber = list(molecularDf[molecularDf['atom ID'] == key]['Equivalence'])[0]
        for row, value in molecularDf.iterrows():
            if value['Equivalence'] == equivalenceNumber:
                unresolved[str(index)]['indices'].append(row)
                unresolved[str(index)]['Stoich'].append(value['Stoich'])
    return unresolved

def perturbUniformVector(unresolved, siteIndex):
    '''
    Given a lower bound and upper bound vector, perturb uniformly between bounds 
    '''
    site = unresolved[siteIndex]
    low = site['lowVector']
    high = site['highVector']
    newMeas = np.random.uniform(np.array(low),np.array(high))
    #renormalize
    newMeas /= newMeas.sum()

    return newMeas

def updateUnresolved(unresolved, solution, molecularDf):
    '''
    Given a solution to unresolved Fi Error Bounds, adds the solution to the unresolved dictionary
    '''
    for key in unresolved.keys():
        index = int(key)
        site = unresolved[str(index)]
        site['lowVector'] = list(solution[0])
        site['lowVector'].append(1-np.array(site['lowVector']).sum())
        site['highVector'] = list(solution[1])
        site['highVector'].append(1-np.array(site['highVector']).sum())
        
    return unresolved

def defineMultiplier(unresolved, numMolecularSites):
    '''
    Defines a numpy array giving the constants to multiply the fractional abundance of each site by; this is a helper
    function for going from a condensed solved state to a molecular solved state
    '''
    multiplier = np.ones(numMolecularSites)
    for index in range(numMolecularSites):
        if str(index) in unresolved:
            site = unresolved[str(index)]
            perturb = perturbUniformVector(unresolved,str(index))
            for unresolvedIndex in range(len(site['indices'])):
                multiplier[site['indices'][unresolvedIndex]] = perturb[unresolvedIndex]
                multiplier[site['indices'][unresolvedIndex]] /= site['Stoich'][unresolvedIndex]
    return multiplier
