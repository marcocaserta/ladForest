'''
/***************************************************************************
 *   copyright (C) 2016 by Marco Caserta                                   *
 *   marco dot caserta at ie dot edu                                       *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
'''

from __future__ import division
import sys, getopt
import cplex
import unittest
import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import random
import bisect
from sklearn.cross_validation import KFold
from datetime import datetime
import editdistance


nFolds                  = 10
withPrinting            = 0

nForestCycle            = -1
forestDepth             = -1
PHI                     = -1.0
#PHI  = 0.05  # fuzziness value

# BCW
pos = 4
neg = 2

EPSI = 0.0000001


# Parse command line
def parseCommandLine(argv):
    global inputfile
    global nForestCycle
    global forestDepth
    global PHI
    
    try:
        opts, args = getopt.getopt(argv, "hi:d:c:f:", ["help","ifile=","depth=","cycles=","fuzziness="])
    except getopt.GetoptError:
        print "Command Line Erorr. Usage : test.py -i <inputfile> -d <depth> - c <cycles> -f <fuzziness>"
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print "Usage : test.py -i <inputfile> -d <depth> - c <cycles> -f <fuzziness>"
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-d", "--depth"):
            forestDepth = int(arg)
        elif opt in ("-c", "--cycles"):
            nForestCycle = int(arg)
        elif opt in ("-f", "--fuzziness"):
            PHI = float(arg)

            
class Patterns:
    '''
    This class defines the data structure used to collect the patterns. Each
    pattern is defined by two lists:
    - the index of the literals/attributes included in the patter
    - the sign of the literal.

    For example, consider:
    index = [ 0, 1, 5]
    sign  = [ 1, 1, 0]
    This is the pattern x_1x_2\bar{x}_5.

    The structure differentiates bewteen the list of positive patterns (matrices
    "posP" and "signP") and the list of negative patterns ("negP" and "signN").

    We also store the coverage of each patter, i.e., the percentage of points
    of the same class covered by that pattern.
    '''
    def reset(self):
        self.posP      = []
        self.signP     = []
        self.negP      = []
        self.signN     = []
        self.coverageP = []
        self.coverageN = []
        
    def getNPos(self): # returns the number of positive points
        return (len(self.posP))    
    def getNNeg(self): # returns the number of positive points
        return (len(self.negP))    

    
class Stats:
    '''
    Statistics of the pattern population. 
    '''
    def __init__(self):
        self.accuracy     = [] # the accuracy of the LAD theory defined by the set of patterns
        self.nPos         = [] # nr of positive patterns
        self.avgLengthPos = [] # average length (# literals) of positive patterns
        self.nNeg         = [] # nr of negative patterns
        self.avgLengthNeg = [] # average length (# literals) of negative patterns
# END of Class Stats  ==========================================================

class Instance: 
    '''
    Used to read and store the dataset. The structure is initialized by creating
    a matrix "A", which holds the value of each attribute for each observation in
    the dataset, and a vector "y", which contains the class each observation
    belongs to. We also store the number of observations belonging to each class
    (nPos and nNeg).
    '''
    # initialize instance read from inputfile
    def __init__(self):

        self.A2       = [] # to implement formulation that minimized the nr of uncovered
        self.A        = [] # the original observation/attribute matrix (not binarized)
        self.Abin     = [] # the binarized attribute matrix
        self.y        = [] # the class of each observation
        self.nBinAttr = 0
        self.nPos     = 0  # nr. of points belonging to class "positive"
        self.nNeg     = 0  # nr. of points belonging to class "negative"
        self.support  = [] # support set, i.e., list of selected binary attributes

    def readDataSet(self, inputfile):
        '''
        Read the dataset from a disk file (inputfile).
        '''
        with open(inputfile) as f:
            data = f.readline()
            self.nObs, self.nAttr = [int(v) for v in data.split()]

            for i in range(self.nObs):
                data = f.readline().split()
                ai   = [int(data[k]) for k in range(1, self.nAttr+1)]
                self.A.append(ai)
                self.y.append(int(data[self.nAttr+1]))
                if int(data[self.nAttr+1]) == pos:
                    self.nPos = self.nPos + 1
                else:
                    self.nNeg = self.nNeg + 1
# END of Class Instance ========================================================

                    
def splitData(dataset, set_index):
    '''
    Take the "dataset" and extract from it the observations (rows) included in
    the list "set_index".

    The function returns a new structure of type Instance(), called "subset".
    '''
    subset = Instance()
    subset.nObs  = len(set_index)
    subset.nAttr = dataset.nAttr
    for i in set_index:
        subset.A.append(dataset.A[i])
        subset.y.append(dataset.y[i])
        if dataset.y[i] == pos:
            subset.nPos = subset.nPos + 1
        else:
            subset.nNeg = subset.nNeg + 1

    return subset
            

def isFeasibleProblem(dataset):
    '''
    Determine whether the set of binary attributes selected for the current dataset
    leads to a feasible problem. A problem is feasible if every two observations
    belonging to different classes can be distinguished.
    '''
    negObs   = [k for k in range(dataset.nObs) if dataset.y[k] == neg]
    
    for i in range(dataset.nObs):
        if dataset.y[i] == pos:
            for k in negObs:
                atLeastOne = 0
                for j in range(dataset.nBinAttr):
                    if dataset.Abin[i][j] != dataset.Abin[k][j]:
                        atLeastOne = 1
                        break
                        
                if atLeastOne == 0:
                    return -1
    return 1

    
def createSampleForest(dataset, set_row, set_col):
    '''
    Create the data structure for the LAD forest run.
    '''
    
    #print("selecting cols ", set_col)
    subset          = Instance()
    subset.nObs     = len(set_row)
    subset.nBinAttr = len(set_col)

    for i in set_row:
        auxRow = [dataset.Abin[i][j] for j in set_col]
        subset.Abin.append(auxRow)
        subset.y.append(dataset.y[i])
        if dataset.y[i] == pos:
            subset.nPos += 1
        else:
            subset.nNeg += 1
            
    return subset

    
def computeStatistics(inp):
    '''
    Get statistics of the original data structure, i.e., before binarization.
    '''
    # compute mean and standard deviation of each column
    npA = np.array(inp.A)

    #return means, stdevs
    return [npA[:,j].mean() for j in range(inp.nAttr)], [npA[:,j].std()  for j in range(inp.nAttr)]
        

def dataBinarizationPhase(inp, means, stdevs, pTau):

    '''
    Describe the strategy used for data binarization.
    
    sns.set()
    fig = plt.figure(figsize=(6,6))
    '''
    
    # sort values w.r.t. each attribute
    npA = np.array(inp.A)
    cuts = []

    for j in range(inp.nAttr):
        
        tau = pTau*stdevs[j]
        
        '''
        posO = [npA[k,j] for k in range(inp.nObs) if inp.y[k] == pos]
        negO = [npA[k,j] for k in range(inp.nObs) if inp.y[k] == neg]

        plt.subplot(3,3, j)
        plt.hist([posO, negO], label=["pos","neg"], alpha=0.5)
        plt.title("Attribute {0}".format(j))
        plt.legend()
        '''
        '''
        ps = pandas.Series([i for i in posO])
        ccPos = ps.value_counts()
        ps = pandas.Series([i for i in negO])
        ccNeg = ps.value_counts()
        print("POSITIVES ", ccPos)
        print("NEGATIVES ", ccNeg)
        input("aka")
        '''

        cutpoints    = []
        index        = [i for i in range(inp.nObs)]
        sortedI      = [i for (k,i) in sorted(zip(npA[:,j], index))]
        currentClass = inp.y[sortedI[0]]
        currentValue = npA[sortedI[0], j]
            
        k = 0
        while (k < inp.nObs):    
            currentClass = inp.y[sortedI[k]]
            currentValue = npA[sortedI[k], j]
            w = k + 1
            while ((abs(inp.y[sortedI[w]] - currentClass) < EPSI) & (w < inp.nObs-1)):
                w = w + 1

            if (w == inp.nObs-1):
                break

            if (abs(currentValue - npA[sortedI[w],j]) > tau):
                cutpoints.append((currentValue + npA[sortedI[w],j])/2.0)
                #print("new set of cutpoints :: ", cutpoints)
                
            k = w
                
        cuts.append(cutpoints)       
        #print(cutpoints)

    #plt.show()
        
    #fig.savefig("output.png")
    Abin = createBinaryMatrix(inp, cuts)
    nBinAttr = len(Abin[0])
        
    return cuts, Abin, nBinAttr
    
        

def createBinaryMatrix(inp, cuts):
    '''
    Create binary matrix for the attributes.
    The binary matrix is stored in inp.Abin[][]
    '''

    Abin = []
    for i in range(inp.nObs):
        binRow = []
        for j in range(inp.nAttr):
            pos = bisect.bisect(cuts[j], inp.A[i][j])
            binRow = binRow + [1]*pos + [0]*(len(cuts[j])-pos)

        # add binary row
        Abin.append(binRow)

    return Abin


def patternGenerationMIP2(inp, alphaI, mainClass, otherClass, timeLimit=60, solLimit=9999,
                          withPrinting=True, display=4):
    '''
    Implementation aimed at minimizing the number of uncovered observations.
    '''
    
    nMain    = len(mainClass)
    nOther   = len(otherClass)
    nSupport = len(inp.A2[0])
    n        = len(inp.support)

    if (2*n != nSupport):
        print("problem in MIP2")
        exit(156)

    alphaSet = [k for k in range(nSupport) if inp.A2[alphaI][k] == 1]
        
    cpx = cplex.Cplex()

    # define obj function direction
    cpx.objective.set_sense(cpx.objective.sense.minimize)

    # define variables x_j
    cpx.variables.add(obj   = [0]*nSupport,
                      lb    = [0]*nSupport,
                      ub    = [1]*nSupport,
                      types = ["B"]*nSupport)
    
    # define variables y_i
    cpx.variables.add(obj   = [1.0]*nMain,
                      lb    = [0.0]*nMain,
                      ub    = [n]*nMain,
                      types = ["C"]*nMain)


    # variables w_i
    cpx.variables.add(obj   = [0]*nOther,
                      lb    = [0.0]*nOther,
                      ub    = [1.0]*nOther,
                      types = ["C"]*nOther)
    # variable d

    cpx.variables.add(obj   = [0.0],
                      lb    = [1.0],
                      ub    = [n],
                      types = ["C"])

    # create indexes (progressive values. Each var needs to have a unique identifier)
    x_var = range(nSupport)
    y_var = range(nSupport, nSupport+nMain)
    w_var = range(nSupport+nMain, nSupport+nMain+nOther)
    d_var = range(nSupport+nMain+nOther, nSupport+nMain+nOther+1)


    # pattern degree "d"    
    index = [x_var[j] for j in alphaSet]
    value = [1.0      for j in alphaSet]
    index.append(d_var[0])
    value.append(-1.0)
    d_constr = cplex.SparsePair(ind=index, val=value)
    cpx.linear_constraints.add(lin_expr = [d_constr],
                               senses   = ["E"],
                               rhs      = [0.0])
    

    # covering constraint
    progr = 0
    for i in mainClass:
        index = [x_var[j]     for j in alphaSet]
        value = [inp.A2[i][j] for j in alphaSet]
        index.append(y_var[progr])
        value.append(1.0)
        index.append(d_var[0])
        value.append(-1.0)
        covering_constr = cplex.SparsePair(ind=index, val=value)
        cpx.linear_constraints.add(lin_expr = [covering_constr],
                                   senses   = ["G"],
                                   rhs      = [0.0])
        progr += 1

    # fuzziness constraint 1
    progr = 0    
    for i in otherClass:
        index = [x_var[j]     for j in alphaSet]        
        value = [inp.A2[i][j] for j in alphaSet]
        index.append(w_var[progr])
        value.append(-1.0)
        index.append(d_var[0])
        value.append(-1.0)
        fuzziness_constr = cplex.SparsePair(ind=index, val=value)
        cpx.linear_constraints.add(lin_expr = [fuzziness_constr],
                                   senses   = ["L"],
                                   rhs      = [-1.0])
        progr += 1

    # fuzziness constraint 2
    index = [w_var[i] for i in range(nOther)]
    value = [1.0      for i in range(nOther)]
    max_fuzziness = cplex.SparsePair(ind=index, val=value)
    cpx.linear_constraints.add(lin_expr = [max_fuzziness],
                               senses   = ["L"],
                               rhs      = [PHI*nOther])


    try:
        cpx.parameters.mip.interval.set(100) # how often to print info
        cpx.parameters.timelimit.set(timeLimit)
        cpx.parameters.mip.limits.solutions.set(solLimit)
        cpx.parameters.mip.display.set(display)
        
        cpx.solve()
        
    except CplexSolverError as e:
        print("Exception raised during solve: " + e)
    else:
        # get solution
        solution = cpx.solution
        #if withPrinting:
        nCovered = solution.get_objective_value()
        
        #print ("*** *** *** ub[{0:4d}] = {1:10.2f} with solution status = {2:20s}".\
        #       format(0, nCovered, solution.status[solution.get_status()]))

        # get pattern and sign
        pattern = []
        sign    = []
        for j in range(n):
            if solution.get_values(x_var[j]) > 1.0 - cpx.parameters.mip.tolerances.integrality.get():
                pattern.append(inp.support[j])
                sign.append(inp.Abin[alphaI][inp.support[j]])
        for j in range(n, nSupport ):
            if solution.get_values(x_var[j]) > 1.0 - cpx.parameters.mip.tolerances.integrality.get():
                pattern.append(inp.support[j-n])
                sign.append(inp.Abin[alphaI][inp.support[j-n]])
        '''        
        print("PATTERN : ", pattern)
        print("SIGN    : ", sign)
        input("aka")
        '''
        
        # remove covered point of main class
        # VERIFY
        ySolNeg = []        # points not covered when y_i > 0
        for i in range(nMain):
            if solution.get_values(y_var[i]) > cpx.parameters.mip.tolerances.integrality.get():
                ySolNeg.append(mainClass[i])

        #print("Number of UNCOVERED points of MAIN class : ", len(ySolNeg))
        #print("List of uncovered points ", ySolNeg)        
        nUncovered = 0
        for i in mainClass:
            if [inp.Abin[i][k] for k in  pattern] != sign:
                nUncovered += 1
                if len(ySolNeg) > 0 and i not in ySolNeg:
                    print("discrepancy with cplex ... ")
                    print("y({0}) = 1".format(i))
                    print("point ", inp.Abin[i])
                    exit(156)

        #print ("Nr. uncovered from cplex vs manual check", len(ySolNeg), nUncovered)
        if (len(ySolNeg) != nUncovered):
            print("check here h1")
            exit(134)
                    
        ySol = [] # Rem: y_i = 0 ==> observation "i" is not covered
        for i in range(nMain):
            if solution.get_values(y_var[i]) < cpx.parameters.mip.tolerances.integrality.get():
                ySol.append(mainClass[i])

        mainClass = [x for x in mainClass if x not in ySol]

        return mainClass, pattern, sign, nCovered

    
    
def patternGenerationMIP(inp, alphaI, mainClass, otherClass, timeLimit=60, solLimit=9999,
                         withPrinting=True, display=4):
    '''
    Use (a modified version of) Bonates formulation to generate maximum-alpha-
    patterns.

    - mainClass  : the set of points of the main class, i.e., the class point "alpha"
                   belongs to
    - otherClass : the set of points of the opposite class.

    The goal is to find a maximum-alpha-pattern, i.e., a patter that covers the max
    number of points of the main class, while keeping the number of points of the other
    class covered by the pattern below a given threshold value.

    A new version here implements a relaxation of Bonates formulation. More
    precisely, by relaxing variables z_b below, we no longer aim at maximize the
    number of covered point (of the same class). What we are now doing here is
    maximizing the percentage of literals of an observation covered by a pattern.
    (We could call it PARTIAL COVERAGE as opposed to the FULL COVERAGE):
    
    - partial coverage: z_b can now take fractional values, e.g., z_b = 0.9 would
    imply that the point is covered up to a 90%. In other words, 90% of the literals
    that define the pattern are also present in this observation. This might make
    sense in the context of robust results, binarization, cut points, measurement
    errors, etc.

    - full coverage: when a point is fully covered by a pattern, w_b = 1 and, thus,
    we count this point as covered (and we take it away from the set of points
    for the next run of the pattern generation phase).
    
    '''

    nMain    = len(mainClass)
    nOther   = len(otherClass)
    nSupport = len(inp.support)

    cpx      = cplex.Cplex()
    
    # define obj function direction
    cpx.objective.set_sense(cpx.objective.sense.maximize)

    # define variables
    # variables z_b
    cpx.variables.add(obj   = [1]*nMain,
                      lb    = [0]*nMain,
                      ub    = [1]*nMain,
                      types = ["C"]*nMain)
    # variables s_i
    cpx.variables.add(obj   = [0]*nOther,
                      lb    = [0]*nOther,
                      ub    = [1]*nOther,
                      types = ["C"]*nOther)
    
    # variables y_i
    cpx.variables.add(obj   = [0]*nSupport,
                      lb    = [0]*nSupport,
                      ub    = [1]*nSupport,
                      types = ["B"]*nSupport)

    # create indexes (progressive values. Each var needs to have a unique identifier)
    z_var = range(nMain)
    s_var = range(nMain, nMain+nOther)
    y_var = range(nMain+nOther, nMain+nOther+nSupport)
    
    # count number of violations (fuzziness)
    progr = 0
    for i in otherClass:
        
        index = [y_var[k] for k in range(nSupport) if (inp.Abin[alphaI][inp.support[k]] != inp.Abin[i][inp.support[k]])]
        value = [1.0 for k in range(nSupport)      if (inp.Abin[alphaI][inp.support[k]] != inp.Abin[i][inp.support[k]])]
        index.append(s_var[progr])
        
        value.append(1.0)
        progr = progr + 1
        
        fuzziness_constr = cplex.SparsePair(ind=index, val=value)
        cpx.linear_constraints.add(lin_expr = [fuzziness_constr],
                                   senses   = ["G"],
                                   rhs      = [1.0])

        
    # max tollerance (for fuzzy patterns)
    index = [s_var[i] for i in range(nOther)]
    value = [1.0 for i in range(nOther)]
    max_fuzziness = cplex.SparsePair(ind=index, val=value)
    cpx.linear_constraints.add(lin_expr = [max_fuzziness],
                               senses   = ["L"],
                               rhs      = [PHI*nOther])

    count = 0
    for i in mainClass:

        # logical constraint 1 : set z_var to 0 if not covered
        wb = sum([abs(inp.Abin[alphaI][k]-inp.Abin[i][k]) for k in inp.support])

        index = [y_var[k] for k in range(nSupport) if inp.Abin[alphaI][inp.support[k]] != inp.Abin[i][inp.support[k]]]
        value = [1.0      for k in range(nSupport) if inp.Abin[alphaI][inp.support[k]] != inp.Abin[i][inp.support[k]]]
        index.append(z_var[count])
        value.append(wb)
        count += 1
        logical1 = cplex.SparsePair(ind=index, val=value)
        cpx.linear_constraints.add(lin_expr = [logical1],
                                   senses   = ["L"],
                                   rhs      = [wb])

        # logical constraint 2 : set z_var to 1 if covered
        nEls = len(value)
        value[nEls-1] = 1.0
        logical2 = cplex.SparsePair(ind=index, val=value)
        cpx.linear_constraints.add(lin_expr = [logical2],
                                   senses   = ["G"],
                                   rhs      = [1.0])
        

    try:
        cpx.parameters.mip.interval.set(100) # how often to print info
        cpx.parameters.timelimit.set(timeLimit)
        cpx.parameters.mip.limits.solutions.set(solLimit)
        cpx.parameters.mip.display.set(display)
        
        cpx.solve()
        
    except CplexSolverError as e:
        print("Exception raised during solve: " + e)
    else:
        # get solution
        solution = cpx.solution
        if withPrinting:
            nCovered = solution.get_objective_value()

            '''
            print ("*** *** *** ub[{0:4d}] = {1:10.2f} with solution status = {2:20s}".\
                   format(0, nCovered, solution.status[solution.get_status()]))
            '''
            # get pattern
            ySol = []
            for i in range(nSupport):
                if solution.get_values(y_var[i]) > 1.0 - cpx.parameters.mip.tolerances.integrality.get(): 
                    ySol.append(i)

            pattern     = [inp.support[k] for k in ySol]
            signPattern = [inp.Abin[alphaI][inp.support[k]] for k in ySol]
            
            # remove covered points of main class
            zSol = []    
            for i in range(nMain):
                if solution.get_values(z_var[i]) > 1.0 -cpx.parameters.mip.tolerances.integrality.get():
                    zSol.append(mainClass[i])

            mainClass = [x for x in mainClass if x not in zSol]

            return mainClass, pattern, signPattern, nCovered


def patternGenerationPhase(inp, mainClass, otherClass, patternSet, signSet, coverageSet, idxCols,
                           depth=sys.maxint):
    '''
    Pattern Generation Phase.
    '''
    nCycles    = 0
    totInClass = len(mainClass)
    index      = mainClass[:] # used to compute actual coverage
    
    while (len(mainClass) > 0 and nCycles < depth):
        alphaI = random.choice(mainClass) # randomly select alpha point
        mainClass.remove(alphaI) # remove  alpha from the set of points

        # generate max-alpha-pattern
        mainClass, patternRel, signPattern, nCovered = patternGenerationMIP(inp, alphaI, mainClass,
                                                                             otherClass, display=0)  

        # compute real coverage before pattern transformation
        totCovered = computeCoverage(inp, index, patternRel, signPattern)
        
        # FOREST: convert pattern to absolute attribute value
        pattern = [idxCols[k] for k in patternRel]

        
        #if (pattern not in patternSet):   # LOOK AT THIS
        patternSet.append(pattern)
        signSet.append(signPattern)
        coverageSet.append(totCovered/totInClass)
            
        nCycles += 1 # depth of the forest
        

def computeCoverage(dataset, index, pattern, sign):
    '''
    Compute coverage of a pattern (w.r.t. a given dataset)
    '''
    return np.sum( [ [dataset.Abin[i][j] for j in pattern]==sign for i in index])
    
        
def computeStats(patternsSet):
    '''
    Compute basic statistics for a given pattern set.
    '''
    totPatterns = len(patternsSet)
    return totPatterns, np.mean([sum(1 for k in patternsSet[i] if k) for i in range(totPatterns)])
    
    
def getPatternsStats(inp, stats, withPrinting=False):
    '''
    Compute statistics about the structure and quality of the patterns.
    '''
    # positive patterns
    tot, avg = computeStats(inp.posP)
    stats.nPos.append(tot)
    stats.avgLengthPos.append(avg) 

    if withPrinting:
        print("** ** Positive Patterns ** **")
        print("* Nr. Patterns  : {0:5d}".format(tot))
        print("* Mean Length   : {0:5.2f}".format(avg))

    tot, avg = computeStats(inp.negP)
    stats.nNeg.append(tot)
    stats.avgLengthNeg.append(avg)
        
    if withPrinting:    
        print("** ** Negative Patterns ** **")
        print("* Nr. Patterns  : {0:5d}".format(tot))
        print("* Mean Length   : {0:5.2f}".format(avg))


def dataClassification(patternsForest, test, countCycle):
    '''
    '''

    positives      = 0
    negatives      = 0
    falsePositives = 0
    falseNegatives = 0
    unclassifiedP  = 0
    unclassifiedN  = 0
    
    nMisclassified = 0
    nUnclassified  = 0
    cardPos = patternsForest.getNPos()
    cardNeg = patternsForest.getNNeg()
    
    
    for i in range(test.nObs):

        # alternative 1: weighted coverage
        totPos = 0
        for row in range(cardPos):
            if [test.Abin[i][k] for k in patternsForest.posP[row]] == patternsForest.signP[row]:
                totPos = totPos + patternsForest.coverageP[row]

        totNeg = 0
        for row in range(cardNeg):
            if [test.Abin[i][k] for k in patternsForest.negP[row]] == patternsForest.signN[row]:
                totNeg = totNeg + patternsForest.coverageN[row]

        '''
        # alternative 2: coverage
        totPos = np.sum([ [test.Abin[i][k] for k in patternsForest.posP[row]] == patternsForest.signP[row] for row in range(cardPos)])
        totNeg = np.sum([ [test.Abin[i][k] for k in patternsForest.negP[row]] == patternsForest.signN[row] for row in range(cardNeg)])
        '''
        
        #print("Covered by {0} positive patterns and {1} negative patterns".format(totPos, totNeg))
        score = totPos/cardPos - totNeg/cardNeg
        #print("score is ", score)
        if abs(score) < EPSI:
            nUnclassified = nUnclassified + 1
        elif (score > EPSI and test.y[i] == neg) or (score < -EPSI and test.y[i] == pos):
            nMisclassified = nMisclassified + 1

        if test.y[i] == pos:
            if abs(score) < EPSI:
                unclassifiedP = unclassifiedP + 1
            elif score > EPSI:
                positives = positives + 1
            elif score < -EPSI:
                falseNegatives = falseNegatives + 1
            else:
                print("problem 1 with CLASSIFICATION")
                exit(1111)
        else:
            if abs(score) < EPSI:
                unclassifiedN = unclassifiedN + 1
            elif score < -EPSI:
                negatives = negatives + 1
            elif score > EPSI:
                falsePositives = falsePositives + 1
            else:
                print("problem 2 with CLASSIFICATION")
                exit(1112)

                
    a = positives/test.nPos
    b = falseNegatives/test.nPos
    c = unclassifiedP/test.nPos
    d = falsePositives/test.nNeg
    e = negatives/test.nNeg
    f = unclassifiedN/test.nNeg
    
    accuracy = 0.5*(a + e + 0.5*(c+f))
    
    print("*** Summary [{0:^3d}] ***".format(countCycle))
    print("Tot nr of Observations  : {0:5d} ({1:3d}, {2:3d})".format(test.nObs, test.nPos, test.nNeg))
    print("Tot nr of Misclassified : {0:5d} ({1:3d}, {2:3d})".format(nMisclassified, falseNegatives, falsePositives))
    print("Tot nr of Unclassified  : {0:5d} ({1:3d}, {2:3d})".format(nUnclassified, unclassifiedP, unclassifiedN))
    print("% of Accuracy           : {0:5.3f}".format((test.nObs - nMisclassified - nUnclassified)/test.nObs))
    print("% of Accuracy           : {0:5.3f}".format(accuracy))

    return accuracy
              

def printSummary(inputfile, train, test, typeOfSet):
    if typeOfSet=="OVERALL":
        print("\n\b============================================")
        print("          marco caserta (c) 2016 ")
        print("============================================")
        print("      ** 10-fold Cross Validation **")
    print("============================================")
    print("Namefile         : {0:>25s}".format(inputfile))
    print("{0} SET :: ".format(typeOfSet))
    print("Nr. Observations : {0:25d}".format(train.nObs))
    print("    Nr. Pos      : {0:25d}".format(train.nPos))
    print("    Nr. Neg      : {0:25d}".format(train.nNeg))
    print("Nr. Attributes   : {0:25d}".format(train.nAttr))

    if typeOfSet=="OVERALL":
        print("============= ALGO PARAMS ==================")
        print("Nr. Folds        : {0:25d}".format(nFolds))
        print("Fuzziness        : {0:25.2f}".format(PHI))
        print("Forest Cycles    : {0:25.2f}".format(nForestCycle))
        print("Forest Depth     : {0:25.2f}".format(forestDepth))
        print("============================================")
        
    if (test != []):
        print("TESTING SET :: ")
        print("Nr. Observations : {0:25d}".format(test.nObs))
        print("    Nr. Pos      : {0:25d}".format(test.nPos))
        print("    Nr. Neg      : {0:25d}".format(test.nNeg))
        print("Nr. Attributes   : {0:25d}".format(test.nAttr))
    print("============================================")

def printSummaryTable(stats):
    print("\n\n{0:^9s} {1:^5s} {2:^8s} {3:^5s} {4:^8s}".format("%", "nPos", "avg", "nNeg", "avg"))
    for i in range(nFolds):
        print("{0:^9.3f} {1:^5d} {2:^8.2f} {3:^5d} {4:^8.2f}".format(stats.accuracy[i], stats.nPos[i],
                                                                stats.avgLengthPos[i], stats.nNeg[i],
                                                                stats.avgLengthNeg[i]))
        
    print("\n{0:^9.3f} {1:^5.0f} {2:^8.2f} {3:^5.0f} {4:^8.2f}".format(np.mean(stats.accuracy),
                                                                        np.mean(stats.nPos),
                                                                        np.mean(stats.avgLengthPos),
                                                                        np.mean(stats.nNeg),
                                                                        np.mean(stats.avgLengthNeg)))


    # print to file to use brkGA
    with open("solution.txt", "w") as outfile:
        outfile.write("{0:^9.3f} {1:^5.0f} {2:^8.2f} {3:^5.0f} {4:^8.2f} \
                      {5:^9.0f} {6:^9.0f} {7:^9.2f}\n"\
                      .format(np.mean(stats.accuracy),
                              np.mean(stats.nPos),
                              np.mean(stats.avgLengthPos),
                              np.mean(stats.nNeg),
                              np.mean(stats.avgLengthNeg),
                              forestDepth,
                              nForestCycle,
                              PHI))
        
    
def main(argv):
    '''
    Entry point.
    '''
    dataset        = Instance()
    patternsForest = Patterns()
    stats          = Stats()

    #random.seed(27)
    random.seed(datetime.now())
    
    parseCommandLine(argv)
    dataset.readDataSet(inputfile)
    
    printSummary(inputfile, dataset, [], "OVERALL")
    means, stdevs = computeStatistics(dataset)
    
    #kf = KFold(dataset.nObs, n_folds=nFolds, shuffle=True, random_state=27)
    kf = KFold(dataset.nObs, n_folds=nFolds, shuffle=True)
    print(kf)
    
    cFoldCycle = 0
    for train_index, test_index in kf:
        cFoldCycle = cFoldCycle + 1
        
        # initialize (and/or reset) pattern structure
        patternsForest.reset()

        if withPrinting:
            print("\n============================================")
            print("[000] Creation of Training/Testing Sets ")
            print("============================================")
            
        # split into training/testing for this fold
        train = splitData(dataset, train_index)
        test  = splitData(dataset, test_index)
        if withPrinting:
            printSummary(inputfile, train, test, "TRAINING")

        if withPrinting:
            print("\n\n=========================================================")
            print("[001-2] Data Binarization Phase")
            print("=========================================================")

        # ensure feasibility of the binarization scheme for the training set
        pTau       = 0.1
        isFeasible = -1    
        while (isFeasible == -1):
            cuts, train.Abin, train.nBinAttr = dataBinarizationPhase(train, means, stdevs, pTau)
            #isFeasible = isFeasibleProblem(train)
            isFeasible = 1
            if isFeasible == -1:
                pTau = pTau * 0.75                

        train.support = range(train.nBinAttr)  # this deactivate the SCP
                
        if withPrinting:
            print("[001-2] End of Phase")

        colsForest = []

        # LAD-FOREST CYCLE STARTS HERE
        for forestCycle in range(nForestCycle):

            if withPrinting:
                print("\n\n=========================================================")
                print("[003-{0:2d}] Bootstrap Sample and FOREST Selection Phase".format(forestCycle))
                print("=========================================================")
                input("aka")
                
            # create bootstrap sample
            idxRows    = np.random.choice(range(train.nObs), size=train.nObs, replace=True)
            
            # select a subset of attributes
            nSelect    = int(np.ceil(1.0*np.sqrt(train.nBinAttr)))
            isFeasible = -1
            while isFeasible == -1:

                #print("We will select {0} attributes out of a total of {1}".format(nSelect, nBinAttr))
                idxCols    = np.sort(np.random.choice(range(train.nBinAttr), size=nSelect, replace=False))
                
                # create dataset for the forest run
                trainForest = createSampleForest(train, idxRows, idxCols)

                # check whether the selected set of attributes leads to a feasible problem
                isFeasible = isFeasibleProblem(trainForest)
                if  isFeasible == -1:
                    nSelect = int(np.ceil(nSelect * 1.3))
                else:
                    trainForest.support = range(trainForest.nBinAttr)
                    colsForest.append(idxCols) # for stat analysis
                    '''
                    # create extra structure for MIP2
                    nCols = trainForest.nBinAttr
                    for i in range(trainForest.nObs):
                        aux = np.concatenate([trainForest.Abin[i], np.ones(nCols, dtype=np.int)-trainForest.Abin[i]])                        
                        trainForest.A2.append(aux)
                    '''     


            if withPrinting:
                print("\n\n=====================================")
                print("[004-{0:2d}] Patter Generation Phase via MIP".format(forestCycle))
                print("=====================================")
            
            posI  = [k for k in range(trainForest.nObs) if trainForest.y[k] == pos]
            negI  = [k for k in range(trainForest.nObs) if trainForest.y[k] == neg]
            patternGenerationPhase(trainForest, posI[:], negI, patternsForest.posP, patternsForest.signP,
                                   patternsForest.coverageP, idxCols, depth=forestDepth) # POSITIVE

            patternGenerationPhase(trainForest, negI, posI, patternsForest.negP, patternsForest.signN,
                                   patternsForest.coverageN, idxCols, depth=forestDepth) # NEGATIVE
            
            if withPrinting:
                print("[004] End of Phase")
                print("Current nr of patterns :: {0}, {1}".format(patternsForest.getNPos(), patternsForest.getNNeg()))

        getPatternsStats(patternsForest, stats, withPrinting=True)
        if withPrinting:
            print("\n\n=========================================")
            print("[005] LAD Theory and Classification Phase")
            print("=========================================")
        test.Abin = createBinaryMatrix(test, cuts)
        accuracy = dataClassification(patternsForest, test, cFoldCycle)
        stats.accuracy.append(accuracy)
        '''
        print("Length is ", len(colsForest), " Cols of Forest: ", colsForest)
        df = pandas.DataFrame(colsForest)
        df = df.transpose()
        S = df.corr()
        print(S)
        eigValues = np.linalg.eigvalsh(S)
        print(eigValues, " with sum = ", sum(eigValues) )
        print("det(S) = {0:20.10f}".format(np.linalg.det(S)))
        editD = 0
        for i in range(len(colsForest)-1):
            for j in range(i+1, len(colsForest)):
                editD += editdistance.eval(colsForest[i], colsForest[j])
                print("ed({0},{1}) = {2})".format(i, j, editdistance.eval(colsForest[i], colsForest[j])))
        print("EDIT DISTANCE ", editD) 
        input("aka")
        '''

    #===========================================================================
    # END OF K-FOLD CROSS-VALIDATION CYCLE

    printSummaryTable(stats) # print summary of results for the k-fold process
                                                                            
            

        
if __name__ == '__main__':
    main(sys.argv[1:])
