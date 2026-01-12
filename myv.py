import sys, os
import numpy as np
import copy
import re
import readline
from numpy import sin, cos, log, exp

import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

################################################
class Curve():
    plt.gca().set_prop_cycle(None) # reset color cycle at startup
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
        self.width = 1  # line width the curve will be plotted with
        self.markersize = 6 # default marker size, 6 pts
        self.color = Curve.colors[Curve.colorIndex]
        Curve.colorIndex = (Curve.colorIndex  + 1) % len(Curve.colors)
        self.label = label  # this get printed in plot legend
        self.identifier = identifier # this is the 'A,B,etc' label by which the user will refer to the curve
        self.xlabel = xlabel  # what goes at the bottom of the plot
        self.fileName = fileName  # file the curve came from
        self.fill = False

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
                           'dash' : '--',
                           'dash-dot' : '-.',
                           'dotted' : ':',
                           'o' :  'o',
                           '^' : '^',
                           'v' : 'v',
                           'x' : 'x',
                           'diamond' : 'D'}
        self.key = 'on' # 'on', 'off', 'ul', 'ur', 'll', 'lr'
        
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

    # warning that you can have too many curves for our little system
    # here to plot all of them at once, because we label them A-Za-z
    if (len(moreCurves) > 26*2):
        print('WARNING, you have more than 52 curves, and we can only plot 52 at a time')
        
    return moreCurves, kLine, thereAreHeaderLinesLeft

################################################
def expandColonSyntax(s):
    '''take instances of 'a:e' and return 'a b c d e' '''
    found = re.findall(r"\b\w+:\w+\b", s)
    # print('found = ', found)
    if len(found) > 0:
        outList = []
        for colonpair in found:
            if colonpair[0].isnumeric(): # it's curve numbers from the file
                expansion = expandPairNumbers(colonpair)
                # print('expansion = ', expansion)
                s = re.sub(colonpair, expansion, s)
                # print('expanded s = ', s)
            else:
                for c in p.plotList: # it's letters for the plotlist
                    # print('colonpair = ', colonpair)
                    expansion = expandPairLetters(colonpair)
                    # print('expansion = ', expansion)
                    s = re.sub(colonpair, expansion, s)
                    # print('expanded s = ', s)
    return s

#-----------------------------------------------
def expandPairLetters(colonpair):
    outList = []
    pairMin = min(ord(colonpair[0]),ord(colonpair[2]))
    pairMax = max(ord(colonpair[0]),ord(colonpair[2]))
    for cid in [c.identifier for c in p.plotList]:
        if ord(cid) in range(pairMin, pairMax+1):
            outList.append(cid)
        
    return ' '.join(outList)

#-----------------------------------------------
def expandPairNumbers(colonpair):
    outList = []
    cp_tuple = colonpair.split(':')
    pairMin = min(int(cp_tuple[0]),int(cp_tuple[1]))
    pairMax = max(int(cp_tuple[0]),int(cp_tuple[1]))
    for i in range(pairMin, pairMax+1):
        outList.append(str(i)+' ')
        
    return ' '.join(outList)

################################################
def doPlot():
    # print('in doPlot')

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
            plt.plot(c.x, c.y, style, color=c.color, linewidth=c.width,
                     ms=c.markersize)
            # and only the plotted curves are in legend
            legendList.append(c.identifier + ' - ' + c.label)
            # do shading under the plot if the curve says to
            if c.fill == True:
                plt.fill_between(x=c.x, y1=c.y, alpha=0.2)

    # plot the legend
    keyDict = {'ur': 'upper right',
               'cr': 'center right',
               'lr': 'lower right',
               'ul': 'upper left',
               'cl': 'center left',
               'll': 'lower left',
               'uc': 'upper center',
               'cc': 'center',
               'lc': 'lower center',
               'off': 'off',
               'on': 'on',
    }
    if p.key != 'off' and p.key != 'on':
        keyLocation = keyDict[p.key]
        plt.legend(legendList, loc=keyLocation)
    elif p.key == 'on':  # plot wherever matplotlib thinks best
        plt.legend(legendList)
    elif p.key == 'off':
        pass # don't plot the legend
        
        

    
    # handle axis labels
    # # print('doPlot, ylabel = ', p.ylabel)
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
    # print(f'{fileIndex=}')
        
    for j, fileName in enumerate(sys.argv[1:]):
        
        try:
            f = open(fileName, 'r')
        except FileNotFoundError:
            print('cannot open data file "%s"' % fileName)
            sys.exit(1)

        # read the file in
        print('reading data file "%s"' % fileName) 
        lines = f.readlines()
        # print(f'file has %d lines' % len(lines))

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
            
        # print(f'{fileIndex=}')
        

        # doPlot(curves)

        # ylabel = input('enter ylabel: ')
        # plt.ylabel(ylabel)
        # #########
        # doPlot(curves)
        # #########
    # print(f'{fileIndex=}')
    
    return curves, fileIndex
 
################################################

def do_foo(x=0):
    print('x = ',x)
    plt.rc('font', size=22)

#-----------------------------------------------
def do_add(line=None):
    # print(f'{line=}')
    doAopB('+', line)

#-----------------------------------------------
def do_cur(line=None):
    # print(f'{line=}')
    if line is None: 
        print('cur: add curves to plot. e.g, "cur a c "')
        return
    line = line.strip()
    # if the 1st arg is 'menu', then process the menu command
    # and feed its result into 'do_cur'
    if line.split()[0] == 'menu':
        line = do_menu(line.split()[1])
    else:
        pass #otherwish just use the line of args as normal
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
def do_der(line=None):
    '''return new curve that is derivative of the curve'''
    line_args = line.split()
    for cID in line_args:
        c = getCurveFromIdentifier(cID) # argument
        y = np.gradient(c.y, c.x)  # actually do the operation here
        cnew = Curve(name='deriv('+c.identifier+')',
                     x = c.x, y = y, plot = True,
                     label = 'deriv('+c.identifier+')',
                     xlabel = None, fileName = c.fileName)
        addCurveToPlot(cnew) # will add new curve identifier for us

    doPlot()
    
#-----------------------------------------------
def do_div(line=None):
    # print(f'{line=}')
    if line is None: 
        print('div: takes 2 curve args, divides them.  Example - "div a b "')
        return
    doAopB('/', line)
#-----------------------------------------------
def do_dx(line=None):
    #('') print(f'{line=}')
    if line is None: 
        print('dx: shifts x of curves.  Example - "dx a b c 23.3"')
        return
    line_args = line.split()
    dFactor = float(line_args[-1])
    for cid in line_args[:-1]:
        c = getCurveFromIdentifier(cid)
        c.x = dFactor+c.x
    doPlot()
#-----------------------------------------------
def do_dy(line=None):
    # print(f'{line=}')
    if line is None: 
        print('dy: shifts y of curves.  Example - "dy a b c 32.2"')
        return
    line_args = line.split()
    dFactor = float(line_args[-1])
    for cid in line_args[:-1]:
        c = getCurveFromIdentifier(cid)
        c.y = dFactor+c.y
    doPlot()
#-----------------------------------------------
def do_fill(line=None):
    if line is None: return
    print('fill ', line)
    line_args = line.strip().split()
    if line_args[-1] == 'on':
        val = True
    elif line_args[-1] == 'off':
        val = False
    else:
        raise RuntimeError('usage: fill a b c on/off')
    for cid in line_args[:-1]:
        c = getCurveFromIdentifier(cid)
        c.fill = val
    doPlot()
#-----------------------------------------------
def do_fontsize(line=None):
    if line is None or len(line.split()) > 1:
        print('fontsize: takes one integer argument')
        return
    plt.rc('font', size=int(line.strip()))
    doPlot()
           
#-----------------------------------------------
def do_fs(line=None):
    do_fontsize(line)
#-----------------------------------------------
def do_hide(line=None):
    if line is None: return
    # print('hide ', line)
    line_args = line.strip().split()
    for cid in  line_args:
        c = getCurveFromIdentifier(cid)
        # print('hide ', c.name)
        c.plot = False
    doPlot()
#-----------------------------------------------
def do_integrate(line=None):
    '''return new curve that is the integral of the curve'''
    if line is None: 
        print('integrate: return the cumulative integral of a curve, no args except curve list')
        return
    line_args = line.split()
    for cID in line_args:
        c = getCurveFromIdentifier(cID) # argument
        # y = np.integrate(c.y, c.x)  # actually do the operation here
        #-------my own cumulative integrate cuz I can't find one in numpy----
        N = len(c.x) - 1
        dx = c.x[1:] - c.x[:-1]
        x = np.zeros(N)
        y = np.zeros(N)
        y[0] = 0.5*(c.y[1] + c.y[0])*dx[0]
        for i in range(1,N):
            ybar = 0.5*(c.y[i+1] + c.y[i])
            y[i] = y[i-1] + ybar*dx[i]
            x[i] = c.x[i] + 0.5*dx[i]
        y /= c.x[-1]
        #-------end cumulative integrate------------------------------------- 
        
        cnew = Curve(name='integral('+c.identifier+')',
                     x = x, y = y, plot = True,
                     label = 'integral('+c.identifier+')',
                     xlabel = None, fileName = c.fileName)
        
        addCurveToPlot(cnew) # will add new curve identifier for us

    doPlot()
    
#-----------------------------------------------
def do_key(line=None):
    if line is None: return
    # print('key ', line)
    p.key = line.strip() # arg can only be one word
    doPlot()
#-----------------------------------------------

def do_label(line=None):
    if line is None: 
        print('label: change legend label of a curve. e.g, "label a fooness "')
        return
    assert(len(line.split()) > 1)
    # curveIndex = int(line.split()[0])
    curveIdentifier = line.split()[0]
    for i,c in enumerate(p.plotList):
        if curveIdentifier == c.identifier:
            break
    c.label = ' '.join(line.split()[1:])
    doPlot()
#-----------------------------------------------

def do_labelfilenames(line=None):
    if line is None: 
        print('label: add filename to legend label of a curve.')
        return
    line_args = line.strip().split()
    for cID in line_args:
        c = getCurveFromIdentifier(cID) # argument
        c.label = c.label + ': '+c.fileName
    doPlot()
#-----------------------------------------------
def do_ls(line=None):
    # print(f'{line=}')
    if line is None: 
        print('ls: change linestyle of curves. e.g, "ls a c dashed"')
        return
    cids = line.split()[:-1]
    style = line.split()[-1]
    for cid in cids:
        c = getCurveFromIdentifier(cid)
        c.style = style # converted at plot time to legit matplotlib style
    doPlot()
#-----------------------------------------------
def do_lw(line=None):
    # print(f'{line=}')
    if line is None: 
        print('ls: change linewidth of curves. e.g, "lw a:c 3"')
        return
    cids = line.split()[:-1]
    width = line.split()[-1]
    for cid in cids:
        c = getCurveFromIdentifier(cid)
        c.width = width
    doPlot()
#-----------------------------------------------
def do_makecolor(line=None):
    if line is None: 
        print('makecolor: make color of first curve the same as second curve,. e.g,\
        "makecolor b f"')
        return
    print('line = ',line)
    line_args = line.split()
    # we must have an even number of target-source pairs
    N = len(line_args)
    if N %2 != 0 or N == 0:
        print('makecolor: must have an even number of arg so targets match sources')
        return
    for targetID, sourceID in zip(line_args[0:N//2],line_args[N//2:]):
        targetC = getCurveFromIdentifier(targetID)
        sourceC = getCurveFromIdentifier(sourceID)
        targetC.color = sourceC.color
    doPlot()
    
#-----------------------------------------------
def do_markersize(line=None):
    if line is None: 
        print('markersize: marker size is set for curves, e.g., "ms a c 20", 6 is default')
        return
    print('line = ',line)
    cids = line.split()[:-1]
    markersize = line.split()[-1]
    for cid in cids:
        c = getCurveFromIdentifier(cid)
        c.markersize = int(markersize)
    doPlot()
    
#-----------------------------------------------
def do_ms(line=None):
    do_markersize(line)
    
#-----------------------------------------------
def do_menu(line=None):
    outList = []
    print(f'menu: {line=}')
    if line is None: line = '.' # match everything, no filter
    pattern = re.compile(line)
    # filter on the pattern, if there is one
    for i,c in enumerate(curves):
        if not re.search(pattern, c.name) is None:
            print(i, c.name)
            outList.append(str(i)) # sometimes we need the return
    outList = ' '.join(outList)
    return outList
    
#-----------------------------------------------
def do_movefront(line=None):
    '''move the named curve so it is plotted last'''
    if line is None: return
    # print('movefront ', line)
    line_args = line.strip().split()
    for cid in  line_args:
        c = getCurveFromIdentifier(cid)
        p.plotList.remove(c)
        p.plotList.append(c)
    doPlot()
#-----------------------------------------------
def do_mf(line=None): # alias for movefront
    do_movefront(line)
#-----------------------------------------------

def do_q():
    # print('in quit function', flush=True)
    try:
        readline.write_history_file(os.getenv('HOME') + '/.myvhistory')
    except:
        traceback.print_exc(file=sys.stdout)
    finally:
        sys.exit(0)
        
#-----------------------------------------------

def do_p():
    # print('in p', flush=True)
    doPlot()
# def do_p(line = None):
#     print('in p', flush=True)
#     myv.plotItAll()
#-----------------------------------------------
def do_show(line=None):
    if line is None: return
    line_args = line.strip().split()
    for cid in  line_args:
        c = getCurveFromIdentifier(cid)
        c.plot = True
    doPlot()
#-----------------------------------------------
def do_title(line=None):
    print(f'title {line=}')
    p.title = line
    doPlot()

#-----------------------------------------------
def do_xlabel(line=None):
    # print(f'xlabel {line=}')
    p.xlabel = line
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

def do_ylabel(line=None):
    # print(f'ylabel {line=}')
    p.ylabel = line
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

def do_mcur(line=None):
    # print(f'mucr {line=}')
    if line is None: 
        print('mcur: plot multiple curves. e.g, "mcur 4" plots 4th curve of every file.')
        return
    for w in line.split():
        # print(f'mcur {w=}')
        for offset, fileName in fileIndex:
            # print(f'mcur {offset=}')
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
    if line is None: 
        print('del: delete curves. e.g, "del a c "')
        return
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
    if line is None: 
        print('color: change color of curves. e.g, "color a c red"')
        return
    cids = line.split()[:-1]
    color = line.split()[-1]
    for cid in cids:
        c = getCurveFromIdentifier(cid)
        c.color = color
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
def do_newcurve(line=None):
    '''optional first arg is a curve to copy from,
    second arg is the math expression, where x is the
    independent variable.  "nc a sin(x)" or "nc exp(x)"'''
    
    print('newcurve: line = ', line)
    line_args = line.split()
    if len(line_args[0]) == 1 and line_args[0].isalpha():
        c = getCurveFromIdentifier(line_args[0]) # get user's curve
        mathExpression = ' '.join(line_args[1:])
    else:
        c = p.plotList[0] # just grab the first curve, it'll be fine
        mathExpression = ' '.join(line_args)
    x = c.x
    print(f'newcurve: {mathExpression=}')
    y = eval(mathExpression)  # actually do the operation here
    cnew = Curve(name=mathExpression,
                 x = x, y = y, plot = True,
                 label = mathExpression,
                 xlabel = None, fileName = None)
    addCurveToPlot(cnew) # will add new curve identifier for us

    doPlot()
    
#-----------------------------------------------
def do_nc(line=None):  #alias for newcurve()
    do_newcurve(line)
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
def do_sub(line=None):
    print(f'{line=}')
    doAopB('-', line)
#-----------------------------------------------
def do_mul(line=None):
    # print(f'{line=}')
    doAopB('*', line)
#-----------------------------------------------
def do_mdiv(line=None):
    #('') print(f'{line=}')
    if line is None: 
        print('mdiv: divides curves by x.  Example - "mdiv a b c 1e-9"')
        return
    line_args = line.split()
    mFactor = float(line_args[-1])
    if mFactor == 0.0:
        print('mdiv: divides curves by x.  Example - "mdiv a b c 1e-9".  You have supplied x=zero, cannot divide by that.')
    for cid in line_args[:-1]:
        c = getCurveFromIdentifier(cid)
        c.x = c.x/mFactor
    doPlot()
#-----------------------------------------------
def do_mx(line=None):
    #('') print(f'{line=}')
    if line is None: 
        print('mx: multiplies x of curves.  Example - "mx a b c 1e9"')
        return
    line_args = line.split()
    mFactor = float(line_args[-1])
    for cid in line_args[:-1]:
        c = getCurveFromIdentifier(cid)
        c.x = mFactor*c.x
    doPlot()
#-----------------------------------------------
def do_my(line=None):
    # print(f'{line=}')
    if line is None: 
        print('my: multiplies y of curves.  Example - "my a b c 1e9"')
        return
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

def do_xmin(line=None):
    '''return new curve that is the old curve from arg to end'''
    line_args = line.split()
    for cID in line_args[:-1]: # last arg is the mix_x value
        c = getCurveFromIdentifier(cID) # argument
        min_x = float(line_args[-1])
        for k,x in enumerate(c.x):
            if x > min_x:
                break
        new_x = c.x[k:]
        new_y = c.y[k:]
        cnew = Curve(name='xmin('+c.identifier+')',
                     x = new_x, y = new_y, plot = True,
                     label = 'xmin('+c.identifier+')',
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

    debug = False
    
    try:
        readline.read_history_file(os.getenv('HOME') + '/.myvhistory')
    except:
        f = open(os.getenv('HOME') + '/.myvhistory', 'w')
        f.close()
        
    s = None
    while True:
        s = input('myv: ')
        s = expandColonSyntax(s) # instances of 'a:c' -> 'a b c'
        print('fully expanded s=',s)

        # toggle debug if needed
        if s == 'debug' and debug == False:
            debug = True
            continue
        elif s == 'debug' and debug == True:
            debug = False
            continue
            
        if s == '': continue
        
        st = 'do_%s' % (s.split()[0])
        if len(s.split()) > 1:
            st += '("%s")' % (' '.join(s.split()[1:]))
        else:
            st += '()'

        if debug:
            eval(st)  # leave unprotected so we can see errors while developing
        else:
            try:
                eval(st)
            except NameError:
                print('error, no such command, "%s"' % st, flush=True)

    print('out of the while loop', flush = True)
    
################################################

# this is what gets executed when this file is imported
curves, fileIndex = getCurves()
p = Plot()

manager = plt.get_current_fig_manager()
manager.window.setGeometry(1500,100,800,800)

### window positioning taken from this example on
### stackoverflow discussion:
# import matplotlib
# import matplotlib.pyplot as plt

# def move_figure(f, x, y):
#     """Move figure's upper left corner to pixel (x, y)"""
#     backend = matplotlib.get_backend()
#     if backend == 'TkAgg':
#         f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
#     elif backend == 'WXAgg':
#         f.canvas.manager.window.SetPosition((x, y))
#     else:
#         # This works for QT and GTK
#         # You can also use window.setGeometry
#         f.canvas.manager.window.move(x, y)

# f, ax = plt.subplots()
# move_figure(f, 500, 500)
# plt.show()
