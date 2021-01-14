import csv

def cleanList(dirtyList, name, makeFloat = False):
    '''
    Remove all empty strings from a list, and printing a warning if any were present
    '''
    #if x means if x is not the empty string, (assuming we only have strings)
    if makeFloat:
        cleaned = [float(x) for x in dirtyList if x]
    else:
        cleaned = [x for x in dirtyList if x]
    if len(cleaned) < len(dirtyList):
        print("Warning: Culled empty strings from " + name + "; it may just be whitespace, but you may have left something blank")
    return cleaned

def parseInput(InputFile):  
    '''
    Takes an input CSV in our standard format and puts all the information in a dictionary. The basic functionality is that it
    searches the first column of the csv looking for keywords corresponding to different types of measurements. If it finds
    one, it adds it to the respective dictionary. Some information must be given in blocks, including M1 measurements and 
    M+2 fragments. In general, CSVs should be prepared with all of the same types of measurements in the same block, to be 
    safe. 
    
    The types of information this stores are:
        basicInfo = element IDs, stoichiometries, equivalence numbers, and fragmentation data
        M1Dict = Information about M1 experiments, i.e. when singly substituted isotopologues are isolated and fragmented.
                 Data are given in percent molar abundance space.
        EADict = Elemental Analyzer data. Given in delta space. 
        bulkOrbiDict = O values for M1 isotopologues, given as O values. 
        bulkOrbiM2Plus = O values for M2 and above isotopologues (18O, 34S, 36S), given as O values.
        clumpedO = clumped species consisting solely of M1 substitutions, i.e. 13C15N, 13CD
        clumpedOM2Plus = clumped species consisting of some M2 and above substitutions, i.e. 13C34S
        M2FragDict = Measurements of the M1 and M2 beams of a fragment. For example, I may fragment methionine and look
                     at the 56 beam, and measure only the singly and doubly substiuted beams. Note this assumes you isolated
                     the FULL molecular ion, not only the M1 beam!
        
    '''
    basicInfo = {}
    M1Dict = {'Standard':{},'Sample':{}}
    M2Dict = {'Standard':{},'Sample':{}}
    EADict = {'Standard':[],'Sample':[]}
    bulkOrbiDict = {'Standard':{},'Sample':{}}
    bulkOrbiM2Plus = {'Standard':{},'Sample':{}}
    clumpedO = {'Standard':{},'Sample':{}}
    clumpedOM2Plus = {'Standard':{},'Sample':{}}
    M2FragDict = {'Standard':{},'Sample':{}}
    
    onStandard = False
    onSample = False
    onSampleM2 = False
    onSampleM2 = False
    onM2Frag = False

    with open(InputFile, mode='r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            if row[0] == 'Atom IDs':
                atomIDs = row[2:]
                basicInfo['atom ID'] = cleanList(atomIDs, 'atom IDs')

            if row[0] == 'Element IDs':
                ElementIDs = row[2:]
                basicInfo['element'] = cleanList(ElementIDs, 'Element IDs')

            if row[0] == 'Stoichiometry':
                Stoichiometry = row[2:]
                basicInfo['Stoich'] = cleanList(Stoichiometry, 'Stoichiometry', makeFloat = True)

            if row[0] == 'Equivalence':
                Equivalence = row[2:]
                basicInfo['Equivalence'] = cleanList(Equivalence, 'Equivalence', makeFloat = True)

            if row[0] == 'Orbi':
                basicInfo[row[1]] = []
                fragment = row[2:]
                basicInfo[row[1]] = cleanList(fragment, 'fragment ' + row[1], makeFloat = True)

            if row[0] == 'Standard Composition':
                deltas = row[2:]
                basicInfo['Ref Deltas'] = cleanList(deltas, 'Standard Composition', makeFloat = True)

            if row[0] == 'Standard':
                onStandard = True
                fragList = cleanList(row[1:], row[0])
                continue

            if onStandard == True and row[0] != '':
                measureList = row[1:]

                for index in range(len(fragList)):
                    if index % 2 == 0:
                        if row[0] not in M1Dict['Standard']:
                            M1Dict['Standard'][row[0]] = {}
                        if fragList[index] not in M1Dict['Standard'][row[0]]:
                            M1Dict['Standard'][row[0]][fragList[index]] = {'Measurement':0,'Uncertainty':0}
                        M1Dict['Standard'][row[0]][fragList[index]]['Measurement'] = float(measureList[index])

                    if index % 2 == 1:
                        M1Dict['Standard'][row[0]][fragList[index-1]]['Uncertainty'] = float(measureList[index])

            else:
                onStandard = False

            if row[0] == 'Sample':
                onSample = True
                fragList = cleanList(row[1:], row[0])
                continue

            if onSample == True and row[0] != '':
                measureList = row[1:]
                for index in range(len(fragList)):
                    if index % 2 == 0:
                        if row[0] not in M1Dict['Sample']:
                            M1Dict['Sample'][row[0]] = {}
                        if fragList[index] not in M1Dict['Sample'][row[0]]:
                            M1Dict['Sample'][row[0]][fragList[index]] = {'Measurement':0,'Uncertainty':0}
                        M1Dict['Sample'][row[0]][fragList[index]]['Measurement'] = float(measureList[index])

                    if index % 2 == 1:
                        M1Dict['Sample'][row[0]][fragList[index-1]]['Uncertainty'] = float(measureList[index])
            else:
                onSample = False
                
            if row[0] == 'Sample M2':
                onSampleM2 = True
                fragList = cleanList(row[1:], row[0])
                continue

            if onSampleM2 == True and row[0] != '':
                measureList = row[1:]
                for index in range(len(fragList)):
                    if index % 2 == 0:
                        if row[0] not in M2Dict['Sample']:
                            M2Dict['Sample'][row[0]] = {}
                        if fragList[index] not in M2Dict['Sample'][row[0]]:
                            M2Dict['Sample'][row[0]][fragList[index]] = {'Measurement':0,'Uncertainty':0}
                        M2Dict['Sample'][row[0]][fragList[index]]['Measurement'] = float(measureList[index])

                    if index % 2 == 1:
                        M2Dict['Sample'][row[0]][fragList[index-1]]['Uncertainty'] = float(measureList[index])
            else:
                onSampleM2 = False
                
            if row[0] == "EA Standard":
                EADict["Standard"].append((row[1], float(row[2]), float(row[3])))  
                
            if row[0] == "EA Sample":
                EADict["Sample"].append((row[1], float(row[2]), float(row[3])))
                
            if row[0] == "Bulk Orbi Standard":
                bulkOrbiDict['Standard'][row[1]] = {'Measurement':float(row[2]), 'Uncertainty':float(row[3])}
                
            if row[0] == "Bulk Orbi Sample":
                bulkOrbiDict['Sample'][row[1]] = {'Measurement':float(row[2]), 'Uncertainty':float(row[3])}
                
            if row[0] == "Bulk Orbi M2+ Sample":
                bulkOrbiM2Plus['Sample'][row[1]] = {'Measurement':float(row[2]), 'Uncertainty':float(row[3])}
                
            if row[0] == "M+2 Fragment":
                onM2Frag = True
                fragment = row[1]
                M2FragDict['Sample'][fragment] = {}
                continue
            
            if onM2Frag == True and row[0] != '':
                M2FragDict['Sample'][fragment][row[0]] = {'Measurement':float(row[1]), 'Uncertainty':float(row[2])}      

    #remove sigmas from fragment list
    del fragList[1::2]
    M1Dict['Fragment List'] = fragList
    
    InputFile = {'basicInfo':basicInfo,'M1Dict':M1Dict,'M2Dict':M2Dict,'EADict':EADict,'bulkOrbiDict':bulkOrbiDict,
                 'bulkOrbiM2Plus':bulkOrbiM2Plus, 'clumpedO':clumpedO,'clumpedOM2Plus':clumpedOM2Plus,
                 'M2FragDict':M2FragDict}
    
    return InputFile