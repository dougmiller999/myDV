import sys
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

################################################
class Curve():
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colorIndex = 0

    def __init__(self, name=None, x = None, y = None, plot = False,
                 label = None, xlabel = None, fileName = None):
        self.name = name
        self.x = x
        self.y = y
        self.plot = plot
        self.style = 'o-'
        self.color = Curve.colors[Curve.colorIndex]
        Curve.colorIndex = (Curve.colorIndex  + 1) % len(Curve.colors)
        self.label = label
        self.xlabel = xlabel
        self.fileName = fileName

################################################
class Plot():
    def __init__(self, title='', xlabel='', ylabel='', annotations='', plotList=None):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.annotations = annotations
        self.plotList = []
        
################################################
def readADataEntry(lines, kLine):
    # little bit of error checking about the header line
    headline = lines[kLine].strip()
    if headline[0] != '#':
        print(f'{headline=}')
        raise RuntimeError('first line does not start with a "#", cannot read the header')
        
    # figure out the curves
    xlabelName = headline.split()[1]
    curveNames = headline.split()[2:]
    numCurves = len(curveNames)
    # print(f'{curveNames=}')
    if numCurves < 1:
        raise RuntimeError('it looks like there are no curves in this file?')
    time = []
    curves = [ [] for _ in range(numCurves) ]
    kLine += 1
    data = lines[kLine]
    while data[0] != '#' and kLine < len(lines)-1:
        # print(f'{kLine=}, {data=}')
        dataValues = data.split()
        time.append(float(dataValues[0]))
        kc = 1
        for c in curves:
            c.append(float(dataValues[kc]))
            kc += 1
        kLine += 1
        data = lines[kLine]

    # print(f'{time=}')
    # print(f'{curves=}')

    # sanity check, all the arrays are the same length?
    assert(len(time) == len(curves[0]))
    assert(numCurves*len(time) == sum([len(c) for c in curves]))
    # print(f'each curve has %d points' % len(time))

    # are we at the end of the file?  If we are, record the last line,
    # which we haven't processed yet.  And don't tell me this is ugly
    # code, I know it.  I haven't figured out a good way to loop
    # cleanly over the data and not need something weird to handle the
    # end.
    if lines[kLine][0] == '#':
        thereAreHeaderLinesLeft = True
    else:
        thereAreHeaderLinesLeft = False
        print('running there are no header lines left')
        # print(f'{kLine=}, {data=}')
        dataValues = data.split()
        time.append(float(dataValues[0]))
        kc = 1
        for c in curves:
            c.append(float(dataValues[kc]))
            kc += 1

    # Ok, convert to numpy arrays and return Curve objects
    time = np.array(time)
    curves = np.array(curves)
    if len(curveNames) != len(curves):
        print(f'{len(curveNames)=} and {len(curves)=}')
        raise RuntimeError('len(curveNames) != len(curves)')
    moreCurves = [Curve(name=curveNames[i], x=time, y=c,
                        label = curveNames[i],
                        xlabel = xlabelName) for i,c in enumerate(curves)]

    return moreCurves, kLine, thereAreHeaderLinesLeft

################################################
def doPlot():

    plt.cla()
    plt.ion()
    plt.show()
    
    for c in p.plotList:
        plt.plot(c.x, c.y, 'o-', color=c.color)

    plt.ylabel(p.ylabel)
        
    if len(p.plotList) == 1:
        plt.ylabel(p.plotList[0].name)

    if p.xlabel == '':
        tryXlabel = p.plotList[0].xlabel
        canDoXLabel = True
        for c in p.plotList:
            if c.xlabel != tryXlabel:
                canDoXLabel = False
        if canDoXLabel == True:
            plt.xlabel(tryXlabel)
    else:
        plt.xlabel(p.xlabel)
    
    plt.legend([c.label for c in p.plotList])
    plt.title(p.title)
    plt.grid(visible=True)
    plt.draw()

################################################
    
def getCurves():
    '''function that reads in the ASCII data from space separated columns
    data file and plots it using matplotlib.  Header line has a # in
    front.

    '''
    
    # print(f'{sys.argv=}')
    curves = []
    
    if len(sys.argv) < 2:
        raise RuntimeError('no data file given')

    fileIndex = [[0, f] for f in sys.argv[1:]]
    print(f'{fileIndex=}')
        
    for j, fileName in enumerate(sys.argv[1:]):
        
        try:
            f = open(fileName, 'r')
        except FileNotFoundError:
            print('cannot open data file "%s"' % fileName)
            sys.exit(1)

        # read the file in
        print('reading data file "%s"' % fileName) 
        lines = f.readlines()
        print(f'file has %d lines' % len(lines))

        # skip to first data header line
        kLine = 0
        thereAreHeaderLinesLeft = False
        while kLine < len(lines):
            if lines[kLine][0] == '#':
                thereAreHeaderLinesLeft = True
                break
            else:
                kLine += 1

        # error checking the data file
        if thereAreHeaderLinesLeft == False:
            print('*** Fatal error: there are no header lines in this file!!! ***') 
            raise RuntimeError('*** Fatal error: there are no header lines in this file!!! ***') 

        # looks like we are ok, let's read some data
        while thereAreHeaderLinesLeft == True:
            # read the data for this header
            moreCurves, kLine, thereAreHeaderLinesLeft = readADataEntry(lines, kLine)
            # associate these curves with the current file
            for c in moreCurves:
                c.fileName = fileName
            curves += moreCurves
            # skip any blank lines until we reach the next header 

        print('we have %d total curves now' % len(curves))

        if j < len(sys.argv[1:])-1:
            fileIndex[j+1][0] = len(curves)
            
        print(f'{fileIndex=}')
        

        # doPlot(curves)

        # ylabel = input('enter ylabel: ')
        # plt.ylabel(ylabel)
        # #########
        # doPlot(curves)
        # #########
    print(f'{fileIndex=}')
    
    return curves, fileIndex
 
################################################

def do_foo(x):
    print('x = ',x)
#-----------------------------------------------

def do_label(line):
    assert(len(line.split()) > 1)
    curveIndex = int(line.split()[0])
    c = p.plotList[curveIndex]
    c.label = ' '.join(line.split()[1:])
#-----------------------------------------------

def do_q():
    print('in quit function', flush=True)
    sys.exit(0)
#-----------------------------------------------

def do_p():
    print('in p', flush=True)
    doPlot()
# def do_p(line = None):
#     print('in p', flush=True)
#     myv.plotItAll()
#-----------------------------------------------

def do_menu(line=None):
    print(f'{curves=}')
    for i,c in enumerate(curves):
        print(i, c.name) 
#-----------------------------------------------

def do_xlabel(line=None):
    print(f'{line=}')
    p.xlabel = line
#-----------------------------------------------

def do_ylabel(line=None):
    print(f'{line=}')
    p.ylabel = line
#-----------------------------------------------

def do_title(line=None):
    print(f'{line=}')
    p.title = line
#-----------------------------------------------

def do_cur(line=None):
    print(f'{line=}')
    for w in line.split():
        print(f'{w=}')
        p.plotList.append(curves[int(w)])
        curves[int(w)].plot = True
#-----------------------------------------------
# alias for cur
def do_c(line=None):
    do_cur(line)
#-----------------------------------------------

def do_mcur(line=None):
    print(f'mucr {line=}')
    for w in line.split():
        print(f'mcur {w=}')
        for offset, fileName in fileIndex:
            print(f'mcur {offset=}')
            p.plotList.append(curves[offset+int(w)])
            curves[offset+int(w)].plot = True
#-----------------------------------------------

def do_ls(line=None):
    print(f'{line=}')
    for i,c in enumerate(p.plotList):
        print( i, c.name, c.fileName)
#-----------------------------------------------
        
def do_del(line=None):
    print(f'{line=}')
    nums = [int(w) for w in line.split()]
    nums.reverse()
    for n in nums:
        print(f'{n=}')
        p.plotList.pop(n)
################################################

curves, fileIndex = getCurves()
p = Plot()

def commandLoop():

    s = None
    while True:
        s = input('myv: ')

        st = 'do_%s' % (s.split()[0])
        if len(s.split()) > 1:
            st += '("%s")' % (' '.join(s.split()[1:]))
        else:
            st += '()'

        eval(st)  # leave unprotected so we can see errors while developing
        try:
            pass
            # eval(st)
        except NameError:
            print('error, no such command, "%s"' % st, flush=True)

    print('out of the while loop', flush = True)
    
