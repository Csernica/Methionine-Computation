###A short module for perturbing proportional vectors. This may need to be expanded substantially at some point. 
import numpy as np

def perturbProportionalVector(testMeas,testUncert):
    '''
    A simple function that has deep importance in our routine. 
    
    The idea is: we have some measured quantities, m_1, m_2, ..., m_n with the sum of m_i = 1. This could be 
    a measurement of a given fragment, a vector in fractional abundance space, or proportional contributions to 
    an unresolved site. We also have some uncertainties, delm_1, delm_2, ..., delm_n. We want to propagate these
    uncertainties through with a monte carlo approach; i.e. we perturb the m vector N times and use it to 
    calculate some quantity. 
    
    If we perturb each m_i according to a normal distribution, we miss the fact that errors in m_i are correlated. 
    
    To do this properly, we could sample vectors assuming they follow a Dirichlet distribution. However, it is 
    challenging to define a dirichlet distribution based on our measured uncertainties. This would be a good
    opportunity for an improvement to our model. 
    
    Currently, we perturb each m_i individaully, then normalize such that the sum of m_i equals 1 (actually, the
    old sum; this is simpler for measurements as we don't report the fractional abundance of the unsubstituted. We
    could add this in. It would require building a bit more infrastructure on the pullOutCompositionAndMeasurement
    function to allow us to perturb after pullOutCompositionAndMeasurement, to keep things fast for 1000s of 
    computations, but also remove the unsubstituted vector from our measurement matrix). 
    
    This is reasonable approximation, but could run into trouble if some m_i is perturbed such that m_i < 0, a 
    nonphysical result. Using the Dirichlet distribution would remove this possibility. 
    
    To address this, we print a warning if anything is perturbed below 0. I expect that in routine use this will
    be rare.
    
    (see http://mayagupta.org/publications/FrigyikKapilaGuptaIntroToDirichlet.pdf)
    
    testMeas and testUncert are numpy arrays of the same length.     
    '''
    newMeas = np.random.normal(testMeas,testUncert)
    if min(newMeas) < 0:
        print("WARNING: A measurement iteration went below 0. The normalization method does not account for this situation")
    newMeas /= newMeas.sum()
    
    return newMeas

def perturbM1Measurement(M1Measurement, M1Uncertainty, numIsotopes, numFragments):
    '''
    Take an M1 measurement matrix, output of pullOutCompositionAndMeasurement and perturb the measurement
    with normalization. To do so, it will skip the first element (closure) then pull sections of length 
    numIsotopes; each section corresponds to one peak. It will perturb this, and continue through all sections. 
    Inputs:
        M1Measurement: An M1 measurement vector (list) from pullOutCompositionAndMeasurement
        M1Uncertainty: An M1 uncertainty vector (list) from pullOutCompositionAndMeasurement
        numIsotopes: The number of isotopes observed for each peak
        numFragments: The number of peaks observed
    
    Outputs:
        perturbedMeasurement: A list, an M1 measurement vector perturbed based on the input uncertainties. 
    '''
    perturbedMeasurement = M1Measurement.copy()
    #Track row of the measurement vector 
    row = 1
    for peak in range(0, numFragments):
        peakArray = np.array(M1Measurement[row:row+numIsotopes])
        peakUncertainty = np.array(M1Uncertainty[row:row+numIsotopes])
        perturbedMeasurement[row:row+numIsotopes] = perturbProportionalVector(peakArray, peakUncertainty)
        row += numIsotopes
    
    return perturbedMeasurement
        