import sys
import numpy as np
import copy
import re

import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

################################################
class Curve():
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colorIndex = 0

    def __init__(self, name=None, x = None, y = None, plot = False,
                 label = None, xlabel = None, fileName = None,
                 identifier = None):
        self.name = name
        self.x = x
        self.y = y
        self.plot = plot
        self.style = 'o-'  # line style the curve will be plotted with
        self.color = Curve.colors[Curve.colorIndex]
        Curve.colorIndex = (Curve.colorIndex  + 1) % len(Curve.colors)
        self.label = label  # this get printed in plot legend
        self.identifier = identifier # this is the 'A,B,etc' label by which the user will refer to the curve
        self.xlabel = xlabel  # what goes at the bottom of the plot
        self.fileName = fileName  # file the curve came from

################################################
class Plot():
    def __init__(self, title='', xlabel='', ylabel='', annotations='', plotList=None):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.yscale = 'linear'
        self.xscale = 'linear'
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.annotations = annotations
        self.plotList = []
        # initialize curve identifiers as [a-z]+[A-Z]
        self.curveIdentifiers = [chr(a) for a in range(97,123)] + [chr(a) for a in range(65,91)]
        self.currentCurveIdentifierIndex = 0 # first one is 'a'
        self.styleDict = { 'o-' : 'o-',
                           'solid' : '-',
                           'dashed' : '--',
                           'dash-dot' : '-.',
                           'dotted' : ':',
                           'o' :  'o',
                           '^' : '^',
                           'v' : 'v',
                           'x' : 'x',
                           'diamond' : 'D'}
        
    def getNextIdentifier(self):
        index = self.currentCurveIdentifierIndex
        ident = self.curveIdentifiers[index]
        self.currentCurveIdentifierIndex += 1
        return ident
        
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
def expandColonSyntax(s):
    '''take instances of 'a:e' and return 'a b c d e' '''
    found = re.findall(r"\b\w:\w\b", s)
    if len(found) > 0:
        outList = []
        for colonpair in found:
            for c in p.plotList:
                if c.plot==True:
                    print('colonpair = ', colonpair)
                    expansion = exp(colonpair)
                    print('expansion = ', expansion)
                    s = re.sub(colonpair, expansion, s)
                    print('expanded s = ', s)
    return s

#-----------------------------------------------
def exp(colonpair):
    outList = []
    pairMin = min(ord(colonpair[0]),ord(colonpair[2]))
    pairMax = max(ord(colonpair[0]),ord(colonpair[2]))
    for cid in [c.identifier for c in p.plotList]:
        if ord(cid) in range(pairMin, pairMax+1):
            outList.append(cid)
        
    return ' '.join(outList)

# def exp(colonpair):
#     print('colonpair = ', colonpair)
#     outList = []
#     pairMin = min(ord(colonpair[0]),ord(colonpair[2]))
#     pairMax = max(ord(colonpair[0]),ord(colonpair[2]))
#     for cid in pl:
#         print('cid = ', cid)
#         print('range = ', range(ord(colonpair[0]), ord(colonpair[2])+1))
#         if ord(cid) in range(pairMin, pairMax+1):
#             outList.append(cid)
#             print('appended ', cid)
#     print('returning ', ' '.join(outList))
#     return ' '.join(outList)
################################################
def doPlot():
    print('in doPlot')

    plt.cla()
    plt.ion()
    plt.show()

    legendList = []
    for c in p.plotList:
        if c.plot == True: 
            if c.style in p.styleDict.keys():
                style = p.styleDict[c.style]
            else:
                style = c.style
            plt.plot(c.x, c.y, style, color=c.color)
            # and only the plotted curves are in legend
            legendList.append(c.identifier + ' - ' + c.label)

    # plot the legend
    plt.legend(legendList)

    
    # handle axis labels
    # print('doPlot, ylabel = ', p.ylabel)
    if len(p.plotList) == 1 and p.ylabel == '':
        plt.ylabel(p.plotList[0].name)
    else:
        plt.ylabel(p.ylabel)

    if p.xlabel == '':
        if len(p.plotList) > 0:
            tryXlabel = p.plotList[0].xlabel
            canDoXLabel = True
            for c in p.plotList:
                if c.xlabel != tryXlabel:
                    canDoXLabel = False
            if canDoXLabel == True:
                plt.xlabel(tryXlabel)
    else:
        plt.xlabel(p.xlabel)

    # log scales or not
    plt.yscale(p.yscale)
    plt.xscale(p.xscale)
    # limits
    plt.xlim(p.xmin, p.xmax)
    plt.ylim(p.ymin, p.ymax)
    # title, turn the grid on
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
            if lines[kLine].strip()[0] == '#':
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
def do_hide(line=None):
    if line is None: return
    print('hide ', line)
    line_args = line.strip().split()
    for cid in  line_args:
        c = getCurveFromIdentifier(cid)
        print('hide ', c.name)
        c.plot = False
    doPlot()
#-----------------------------------------------
def do_show(line=None):
    if line is None: return
    print('show ', line)
    line_args = line.strip().split()
    for cid in  line_args:
        c = getCurveFromIdentifier(cid)
        print('show ', c.name)
        c.plot = True
    doPlot()
#-----------------------------------------------
def do_xls(line=None):
    if line is not None:
        line = line.strip()
    if not line == 'on' and not line == 'off':
        print('xls requires "on" or "off"')
        return
    else:
        if line == 'on':
            p.xscale = 'log'
        else:
            p.xscale = 'linear'
    doPlot()
#-----------------------------------------------
def do_yls(line=None):
    if line is not None:
        line = line.strip()
    if not line == 'on' and not line == 'off':
        print('yls requires "on" or "off"')
        return
    else:
        if line.strip() == 'on':
            p.yscale = 'log'
        else:
            p.yscale = 'linear'
    doPlot()
#-----------------------------------------------

def do_label(line):
    assert(len(line.split()) > 1)
    # curveIndex = int(line.split()[0])
    curveIdentifier = line.split()[0]
    for i,c in enumerate(p.plotList):
        if curveIdentifier == c.identifier:
            break
    c.label = ' '.join(line.split()[1:])
    doPlot()
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
    # print(f'menu {curves=}')
    for i,c in enumerate(curves):
        print(i, c.name) 
#-----------------------------------------------

def do_xlabel(line=None):
    # print(f'xlabel {line=}')
    p.xlabel = line
    doPlot()
#-----------------------------------------------

def do_ylabel(line=None):
    # print(f'ylabel {line=}')
    p.ylabel = line
    doPlot()
#-----------------------------------------------

def do_title(line=None):
    print(f'title {line=}')
    p.title = line
    doPlot()
#-----------------------------------------------
def getCurveFromIdentifier(curveIdentifier):
    for c in p.plotList:
        if curveIdentifier == c.identifier:
            break
    return c 

#-----------------------------------------------

def addCurveToPlot(c):
    p.plotList.append(c)
    c.plot = True
    # assign identifier letter
    cid = p.curveIdentifiers[p.currentCurveIdentifierIndex]
    p.currentCurveIdentifierIndex += 1  # get ready for the next curve
    if p.currentCurveIdentifierIndex > 51:
        raise RuntimeError('tried to plot too many curves, out of identifiers')
    c.identifier = cid
    
#-----------------------------------------------

def do_cur(line=None):
    # print(f'{line=}')
    for w in line.split():
        # print(f'{w=}')
        c = copy.deepcopy(curves[int(w)])
        addCurveToPlot(c)
    doPlot()
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
            c = copy.deepcopy(curves[offset+int(w)])
            addCurveToPlot(c)
    doPlot()
#-----------------------------------------------

def do_lst(line=None):
    # print(f'{line=}')
    for c in p.plotList:
        print( c.identifier, c.name, c.fileName)
#-----------------------------------------------
        
def do_del(line=None):
    # print(f'{line=}')
    cids = line.split()
    for cid in cids:
        c = getCurveFromIdentifier(cid)
        p.plotList.remove(c)
    doPlot()
#-----------------------------------------------
# alias for del
def do_d(line=None):
    do_del(line)
#-----------------------------------------------
def do_era(line=None):
    # print(f'{line=}')
    p.plotList.clear()
    p.currentCurveIdentifierIndex = 0

    doPlot()
#-----------------------------------------------
def do_color(line=None):
    # print(f'{line=}')
    cids = line.split()[:-1]
    color = line.split()[-1]
    for cid in cids:
        c = getCurveFromIdentifier(cid)
        c.color = color
    doPlot()
#-----------------------------------------------
def do_ls(line=None):
    # print(f'{line=}')
    cids = line.split()[:-1]
    style = line.split()[-1]
    for cid in cids:
        c = getCurveFromIdentifier(cid)
        c.style = style # will get converted at plot time
    doPlot()
#-----------------------------------------------
def doAopB(op,line=None):
    '''op is a char var that is the binary operation being done,
    like '+' for addition'''
    line_args = line.split()
    c1 = getCurveFromIdentifier(line_args[0])
    c2 = getCurveFromIdentifier(line_args[1])
    y = eval('c1.y' + op + 'c2.y')  # actually do the operation here
    cnew = Curve(name=c1.identifier+op+c2.identifier,
                 x = c1.x, y = y, plot = True,
                 label = c1.identifier+op+c2.identifier,
                 xlabel = None, fileName = None)
    addCurveToPlot(cnew) # will add identifier for us

    doPlot()
    
#-----------------------------------------------
def doFunctionOfCurve(func,line=None):
    '''func is a string var that is the operation being done,
    like 'sin' for sine'''
    line_args = line.split()
    c = getCurveFromIdentifier(line_args[0]) # argument
    y = eval('np.'+func + '(c.y)')  # actually do the operation here
    cnew = Curve(name=func+'(c.identifier)',
                 x = c.x, y = y, plot = True,
                 label = func+'('+c.identifier+')',
                 xlabel = None, fileName = None)
    addCurveToPlot(cnew) # will add new curve identifier for us

    doPlot()
    
#-----------------------------------------------
def do_log(line=None):
    # print(f'{line=}')
    doFunctionOfCurve('log', line)
#-----------------------------------------------
def do_log10(line=None):
    # print(f'{line=}')
    doFunctionOfCurve('log10', line)
#-----------------------------------------------
def do_exp(line=None):
    # print(f'{line=}')
    doFunctionOfCurve('exp', line)
#-----------------------------------------------
def do_sin(line=None):
    # print(f'{line=}')
    doFunctionOfCurve('sin', line)
#-----------------------------------------------
def do_cos(line=None):
    # print(f'{line=}')
    doFunctionOfCurve('cos', line)
#-----------------------------------------------
def do_add(line=None):
    # print(f'{line=}')
    doAopB('+', line)
#-----------------------------------------------
def do_sub(line=None):
    print(f'{line=}')
    doAopB('-', line)
#-----------------------------------------------
def do_mul(line=None):
    # print(f'{line=}')
    doAopB('*', line)
#-----------------------------------------------
def do_div(line=None):
    # print(f'{line=}')
    doAopB('/', line)
#-----------------------------------------------
def do_mx(line=None):
    #('') print(f'{line=}')
    line_args = line.split()
    mFactor = float(line_args[-1])
    for cid in line_args[:-1]:
        c = getCurveFromIdentifier(cid)
        c.x = mFactor*c.x
    doPlot()
#-----------------------------------------------
def do_my(line=None):
    # print(f'{line=}')
    line_args = line.split()
    mFactor = float(line_args[-1])
    for cid in line_args[:-1]:
        c = getCurveFromIdentifier(cid)
        c.y = mFactor*c.y
    doPlot()
#-----------------------------------------------
def doAxesMinMax(axis,line=None):
    '''line should have two numbers in it or be the
    string 'de', nothing else counts as good.
    axis must be 'x' or 'y'.
    '''
    # asserts because user input has already been filtered
    assert(line is not None)
    assert(axis=='x' or axis=='y')
    
    line_args = line.split()
    if len(line_args) == 1 and line_args[0] == 'de':
        # find the new limits
        # we have coded like it's the y axis but it could be x
        ymin = 99e99
        ymax = -99e99
        for c in p.plotList:
            values = eval('c.'+axis)  # either axis is fine
            mx = max(values)
            mn = min(values)
            if mx > ymax:
                ymax = mx
            if mn < ymin:
                ymin = mn
        ymax = ymax + 0.1*(ymax-ymin)
        ymin = ymin - 0.1*(ymax-ymin)
    elif len(line_args) == 2:
        ymin = float(line_args[0])
        ymax = float(line_args[1])
    else:
        print('ran: args are "x/ymin x/ymax" or "de"')
        return

    # execute setting of the plot min/max
    exec('p.'+axis+'min = ymin')
    exec('p.'+axis+'max = ymax')
    doPlot()
    
#-----------------------------------------------
def do_ran(line=None):
    '''line should have two numbers in it or be the
    string 'de', nothing else counts as good'''
    if line is None: 
        print('ran: args are "ymin ymax" or "de"')
        return
    doAxesMinMax('y', line)
    
#-----------------------------------------------
def do_dom(line=None):
    '''line should have two numbers in it or be the
    string 'de', nothing else counts as good'''
    if line is None: 
        print('dom: args are "xmin xmax" or "de"')
        return
    doAxesMinMax('x', line)
    
#-----------------------------------------------

def do_der(line=None):
    '''return new curve that is derivative of the curve'''
    line_args = line.split()
    c = getCurveFromIdentifier(line_args[0]) # argument
    y = np.gradient(c.y, c.x)  # actually do the operation here
    cnew = Curve(name='deriv(c.identifier)',
                 x = c.x, y = y, plot = True,
                 label = 'deriv('+c.identifier+')',
                 xlabel = None, fileName = None)
    addCurveToPlot(cnew) # will add new curve identifier for us

    doPlot()
    
#-----------------------------------------------

def do_span(line=None):
    '''return new curve that is y=x over user-settable range'''
    if line is None: 
        print('span: args are "xmin xmax", returns new curve')
        return
    line_args = line.split()
    if len(line_args) != 2: 
        print('span: args are "xmin xmax", returns new curve')
        return
    x0 = float(line_args[0])
    x1 = float(line_args[1])
    x = np.linspace(x0,x1,num=len(curves[0].x), endpoint=True)  # actually do the operation here
    y = np.linspace(x0,x1,num=len(curves[0].x), endpoint=True)  # actually do the operation here
    cnew = Curve(name='span',
                 x = x, y = y, plot = True,
                 label = 'span',
                 xlabel = None, fileName = None)
    addCurveToPlot(cnew) # will add new curve identifier for us

    doPlot()
    
#-----------------------------------------------

################################################

def commandLoop():

    s = None
    while True:
        s = input('myv: ')
        s = expandColonSyntax(s) # instances of 'a:c' -> 'a b c'
        print('fully expanded s=',s)

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
    
################################################

# this is what gets executed when this file is imported
curves, fileIndex = getCurves()
p = Plot()

