from PIL import Image, ImageDraw
from pug.nlp import forest
from pug.db.explore import count_unique

from matplotlib import pyplot as plt
import pandas.np as np
import bisect

############################################################################
## tree/network/graph plotting routines
## for use with decision trees generated in forest.py


def draw_tree(tree, path='tree.jpg'):
    w = forest.get_width(tree) * 100
    h = forest.get_depth(tree) * 100 + 120

    img = Image.new('RGB',(w, h),(255,255,255))
    draw = ImageDraw.Draw(img)

    draw_node(draw, tree, w / 2,20)
    img.save(path,'JPEG')


def draw_node(draw, tree, x, y):
    if tree.results == None:
        # Get the width of each branch
        w1 = forest.get_width(tree.fb) * 100
        w2 = forest.get_width(tree.tb) * 100

        # Determine the total space required by this node
        left = x - (w1 + w2) / 2
        right = x + (w1 + w2) / 2

        # Draw the condition string
        draw.text((x - 20, y - 10), str(tree.col) + ':' + str(tree.value),(0,0,0))

        # Draw links to the branches
        draw.line((x, y, left + w1 / 2, y + 100), fill = (255,0,0))
        draw.line((x, y, right - w2 / 2, y + 100), fill = (255,0,0))

        # Draw the branch nodes
        draw_node(draw, tree.fb, left + w1 / 2, y + 100)
        draw_node(draw, tree.tb, right - w2 / 2, y + 100)
    else:
        txt = ' \n'.join(['%s:%d'%v for v in tree.results.items()])
        draw.text((x - 20, y), txt,(0,0,0))


def classify(observation, tree):
    if tree.results != None:
        return tree.results
    else:
        v = observation[tree.col]
        branch = None
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value: branch = tree.tb
            else: branch = tree.fb
        else:
            if v == tree.value: branch = tree.tb
            else: branch = tree.fb
        return classify(observation, branch)

def prune(tree, mingain):
    # If the branches aren't leaves, then prune them
    if tree.tb.results == None:
        prune(tree.tb, mingain)
    if tree.fb.results == None:
        prune(tree.fb, mingain)
        
    # If both the subbranches are now leaves, see if they
    # should merged
    if tree.tb.results != None and tree.fb.results != None:
        # Build a combined dataset
        tb, fb = [],[]
        for v, c in tree.tb.results.items():
            tb += [[v]] * c
        for v, c in tree.fb.results.items():
            fb += [[v]] * c
        
        # Test the reduction in entropy
        delta = forest.entropy(tb + fb) - (forest.entropy(tb) + forest.entropy(fb) / 2)

        if delta < mingain:
            # Merge the branches
            tree.tb, tree.fb = None, None
            tree.results = count_unique(tb + fb)


def mdclassify(observation, tree):
    if tree.results != None:
        return tree.results
    else:
        v = observation[tree.col]
        if v == None:
            tr, fr = mdclassify(observation, tree.tb), mdclassify(observation, tree.fb)
            tcount = sum(tr.values())
            fcount = sum(fr.values())
            tw = float(tcount) / (tcount + fcount)
            fw = float(fcount) / (tcount + fcount)
            result = {}
            for k, v in tr.items():
                result[k] = v*tw
            for k, v in fr.items():
                result[k] = v*fw            
            return result
        else:
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.value:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return mdclassify(observation, branch)



def get(obj, key, default=None):
    try:
        # integer key and sequence (list) object
        return obj[key]
    except:
        if isinstance(obj, dict):
            try:
                return obj.get(key, default)
            except:
                return obj.get(tuple(obj)[key], default) 
    return obj.getattr(key, default)


#####################################################################################
######## Based on the statistics plotting wrapper from Udacity ST-101
######## https://www.udacity.com/wiki/plotting_graphs_with_python





def scatterplot(x, y):
    plt.ion()
    plt.plot(x, y, 'b.')
    plt.xlim(min(x) - 1, max(x) + 1)
    plt.ylim(min(y) - 1, max(y) + 1)
    plt.draw()


def barplot(labels, data):
    pos = np.arange(len(data))
    plt.ion()
    plt.xticks(pos + 0.4, labels)
    plt.bar(pos, data)
    plt.grid('on')
    #plt.draw()


def histplot(data, bins=None, nbins=5):
    if not bins:
        minx, maxx = min(data), max(data)
        space = (maxx - minx) / float(nbins)
        bins = np.arange(minx, maxx, space)
    binned = [bisect.bisect(bins, x) for x in data]
    l = ['%.1g' % x for x in list(bins) + [maxx]] if space < 1 or space > 1000 else [str(int(x)) for x in list(bins) + [maxx]]
    print l
    if len(str(l[1]) + '-' + l[2]) > 10:
        displab = l[:-1]
    else:
        displab = [x + '-\n ' + y for x, y in zip(l[:-1], l[1:])]
    barplot(displab, [binned.count(x + 1) for x in range(len(bins))])


def barchart(x, y, numbins=None):
    if numbins is None:
        numbins = int(len(x) ** 0.75) + 1
    datarange = max(x) - min(x)
    bin_width = float(datarange) / numbins
    pos = min(x)
    bins = [0 for i in range(numbins + 1)]

    for i in range(numbins):
        bins[i] = pos
        pos += bin_width
    bins[numbins] = max(x) + 1
    binsum = [0 for i in range(numbins)]
    bincount = [0 for i in range(numbins)]
    binaverage = [0 for i in range(numbins)]

    for i in range(numbins):
        for j in range(len(x)):
            if x[j] >= bins[i] and x[j] < bins[i + 1]:
                bincount[i] += 1
                binsum[i] += y[j]

    for i in range(numbins):
        binaverage[i] = float(binsum[i]) / bincount[i]
    barplot(range(numbins), binaverage)
    return x, y


def piechart(labels, data):
    plt.ion()
    fig = plt.figure(figsize=(7, 7))
    plt.pie(data, labels=labels, autopct='%1.2f%%')
    plt.draw()
    return fig


def regression_and_plot(x, y=None):
    """
    Fit a line to the x, y data supplied and plot it along with teh raw samples

    >>> age=[25, 26, 33, 29, 27, 21, 26, 35, 21, 37, 21, 38, 18, 19, 36, 30, 29, 24, 24, 36, 36, 27, 33, 23, 21, 26, 27, 27, 24, 26, 25, 24, 22, 25, 40, 39, 19, 31, 33, 30, 33, 27, 40, 32, 31, 35, 26, 34, 27, 34, 33, 20, 19, 40, 39, 39, 37, 18, 35, 20, 28, 31, 30, 29, 31, 18, 40, 20, 32, 20, 34, 34, 25, 29, 40, 40, 39, 36, 39, 34, 34, 35, 39, 38, 33, 32, 21, 29, 36, 33, 30, 39, 21, 19, 38, 30, 40, 36, 34, 28, 37, 29, 39, 25, 36, 33, 37, 19, 28, 26, 18, 22, 40, 20, 40, 20, 39, 29, 26, 26, 22, 37, 34, 29, 24, 23, 21, 19, 29, 30, 23, 40, 30, 30, 19, 39, 39, 25, 36, 38, 24, 32, 34, 33, 36, 30, 35, 26, 28, 23, 25, 23, 40, 20, 26, 26, 22, 23, 18, 36, 34, 36, 35, 40, 39, 39, 33, 22, 37, 20, 37, 35, 20, 23, 37, 32, 25, 35, 35, 22, 21, 31, 40, 26, 24, 29, 37, 19, 33, 31, 29, 27, 21, 19, 39, 34, 34, 40, 26, 39, 35, 31, 35, 24, 19, 27, 27, 20, 28, 30, 23, 21, 20, 26, 31, 24, 25, 25, 22, 32, 28, 36, 21, 38, 18, 25, 21, 33, 40, 19, 38, 33, 37, 32, 31, 31, 38, 19, 37, 37, 32, 36, 34, 35, 35, 35, 37, 35, 39, 34, 24, 25, 18, 40, 33, 32, 23, 25, 19, 39, 38, 36, 32, 27, 22, 40, 28, 29, 25, 36, 26, 28, 32, 34, 34, 21, 21, 32, 19, 35, 30, 35, 26, 31, 38, 34, 33, 35, 37, 38, 36, 40, 22, 30, 28, 28, 29, 36, 24, 28, 28, 28, 26, 21, 35, 22, 32, 28, 19, 33, 18, 22, 36, 26, 19, 26, 30, 27, 28, 24, 36, 37, 20, 32, 38, 39, 38, 30, 32, 30, 26, 23, 19, 29, 33, 34, 23, 30, 32, 40, 36, 29, 39, 34, 34, 22, 22, 22, 36, 38, 38, 30, 26, 40, 34, 21, 34, 38, 32, 35, 35, 26, 28, 20, 40, 23, 24, 26, 24, 39, 21, 33, 31, 39, 39, 20, 22, 18, 23, 36, 32, 37, 36, 26, 30, 30, 30, 21, 22, 40, 38, 22, 27, 23, 21, 22, 20, 30, 31, 40, 19, 32, 24, 21, 27, 32, 30, 34, 18, 25, 22, 40, 23, 19, 24, 24, 25, 40, 27, 29, 22, 39, 38, 34, 39, 30, 31, 33, 34, 25, 20, 20, 20, 20, 24, 19, 21, 31, 31, 29, 38, 39, 33, 40, 24, 38, 37, 18, 24, 38, 38, 22, 40, 21, 36, 30, 21, 30, 35, 20, 25, 25, 29, 30, 20, 29, 29, 31, 20, 26, 26, 38, 37, 39, 31, 35, 36, 30, 38, 36, 23, 39, 39, 20, 30, 34, 21, 23, 21, 33, 30, 33, 32, 36, 18, 31, 32, 25, 23, 23, 21, 34, 18, 40, 21, 29, 29, 21, 38, 35, 38, 32, 38, 27, 23, 33, 29, 19, 20, 35, 29, 27, 28, 20, 40, 35, 40, 40, 20, 36, 38, 28, 30, 30, 36, 29, 27, 25, 33, 19, 27, 28, 34, 36, 27, 40, 38, 37, 31, 33, 38, 36, 25, 23, 22, 23, 34, 26, 24, 28, 32, 22, 18, 29, 19, 21, 27, 28, 35, 30, 40, 28, 37, 34, 24, 40, 33, 29, 30, 36, 25, 26, 26, 28, 34, 39, 34, 26, 24, 33, 38, 37, 36, 34, 37, 33, 25, 27, 30, 26, 21, 40, 26, 25, 25, 40, 28, 35, 36, 39, 33, 36, 40, 32, 36, 26, 24, 36, 27, 28, 26, 37, 36, 37, 36, 20, 34, 30, 32, 40, 20, 31, 23, 27, 19, 24, 23, 24, 25, 36, 26, 33, 30, 27, 26, 28, 28, 21, 31, 24, 27, 24, 29, 29, 28, 22, 20, 23, 35, 30, 37, 31, 31, 21, 32, 29, 27, 27, 30, 39, 34, 23, 35, 39, 27, 40, 28, 36, 35, 38, 21, 18, 21, 38, 37, 24, 21, 25, 35, 27, 35, 24, 36, 32, 20]
    >>> wage=[17000, 13000, 28000, 45000, 28000, 1200, 15500, 26400, 14000, 35000, 16400, 50000, 2600, 9000, 27000, 150000, 32000, 22000, 65000, 56000, 6500, 30000, 70000, 9000, 6000, 34000, 40000, 30000, 6400, 87000, 20000, 45000, 4800, 34000, 75000, 26000, 4000, 50000, 63000, 14700, 45000, 42000, 10000, 40000, 70000, 14000, 54000, 14000, 23000, 24400, 27900, 4700, 8000, 19000, 17300, 45000, 3900, 2900, 138000, 2100, 60000, 55000, 45000, 40000, 45700, 90000, 40000, 13000, 30000, 2000, 75000, 60000, 70000, 41000, 42000, 31000, 39000, 104000, 52000, 20000, 59000, 66000, 63000, 32000, 11000, 16000, 6400, 17000, 47700, 5000, 25000, 35000, 20000, 14000, 29000, 267000, 31000, 27000, 64000, 39600, 267000, 7100, 33000, 31500, 40000, 23000, 3000, 14000, 44000, 15100, 2600, 6200, 50000, 3000, 25000, 2000, 38000, 22000, 20000, 2500, 1500, 42000, 30000, 27000, 7000, 11900, 27000, 24000, 4300, 30200, 2500, 30000, 70000, 38700, 8000, 36000, 66000, 24000, 95000, 39000, 20000, 23000, 56000, 25200, 62000, 12000, 13000, 35000, 35000, 14000, 24000, 12000, 14000, 31000, 40000, 22900, 12000, 14000, 1600, 12000, 80000, 90000, 126000, 1600, 100000, 8000, 71000, 40000, 42000, 40000, 120000, 35000, 1200, 4000, 32000, 8000, 14500, 65000, 15000, 3000, 2000, 23900, 1000, 22000, 18200, 8000, 30000, 23000, 30000, 27000, 70000, 40000, 18000, 3100, 57000, 25000, 32000, 10000, 4000, 49000, 93000, 35000, 49000, 40000, 5500, 30000, 25000, 5700, 6000, 30000, 42900, 8000, 5300, 90000, 85000, 15000, 17000, 5600, 11500, 52000, 1000, 42000, 2100, 50000, 1500, 40000, 28000, 5300, 149000, 3200, 12000, 83000, 45000, 31200, 25000, 72000, 70000, 7000, 23000, 40000, 40000, 28000, 10000, 48000, 20000, 60000, 19000, 25000, 39000, 68000, 2300, 23900, 5000, 16300, 80000, 45000, 12000, 9000, 1300, 35000, 35000, 47000, 32000, 18000, 20000, 20000, 23400, 48000, 8000, 5200, 33500, 22000, 22000, 52000, 104000, 28000, 13000, 12000, 15000, 53000, 27000, 50000, 13900, 23000, 28100, 23000, 12000, 55000, 83000, 31000, 33200, 45000, 3000, 18000, 11000, 41000, 36000, 33600, 38000, 45000, 53000, 24000, 3000, 37500, 7700, 4800, 29000, 6600, 12400, 20000, 2000, 1100, 55000, 13400, 10000, 6000, 6000, 16000, 19000, 8300, 52000, 58000, 27000, 25000, 80000, 10000, 22000, 18000, 21000, 8000, 15200, 15000, 5000, 50000, 89000, 7000, 65000, 58000, 42000, 55000, 40000, 14000, 36000, 30000, 7900, 6000, 1200, 10000, 54000, 12800, 35000, 34000, 40000, 45000, 9600, 3300, 39000, 22000, 40000, 68000, 24400, 1000, 10800, 8400, 50000, 22000, 20000, 20000, 1300, 9000, 14200, 32000, 65000, 18000, 18000, 3000, 16700, 1500, 1400, 15000, 55000, 42000, 70000, 35000, 21600, 5800, 35000, 5700, 1700, 40000, 40000, 45000, 25000, 13000, 6400, 11000, 4200, 30000, 32000, 120000, 10000, 19000, 12000, 13000, 37000, 40000, 38000, 60000, 3100, 16000, 18000, 130000, 5000, 5000, 35000, 1000, 14300, 100000, 20000, 33000, 8000, 9400, 87000, 2500, 12000, 12000, 33000, 16500, 25500, 7200, 2300, 3100, 2100, 3200, 45000, 40000, 3800, 30000, 12000, 62000, 45000, 46000, 50000, 40000, 13000, 50000, 23000, 4000, 40000, 25000, 16000, 3000, 80000, 27000, 68000, 3500, 1300, 10000, 46000, 5800, 24000, 12500, 50000, 48000, 29000, 19000, 26000, 30000, 10000, 10000, 20000, 43000, 105000, 55000, 5000, 65000, 68000, 38000, 47000, 48700, 6100, 55000, 30000, 5000, 3500, 23400, 11400, 7000, 1300, 80000, 65000, 45000, 19000, 3000, 17100, 22900, 31200, 35000, 3000, 5000, 1000, 36000, 4800, 60000, 9800, 30000, 85000, 18000, 24000, 60000, 30000, 2000, 39000, 12000, 10500, 60000, 36000, 10500, 3600, 1200, 28600, 48000, 20800, 5400, 9600, 30000, 30000, 20000, 6700, 30000, 3200, 42000, 37000, 5000, 18000, 20000, 14000, 12000, 18000, 3000, 13500, 35000, 38000, 30000, 36000, 66000, 45000, 32000, 46000, 80000, 27000, 4000, 21000, 7600, 16000, 10300, 27000, 19000, 14000, 19000, 3100, 20000, 2700, 27000, 7000, 13600, 75000, 35000, 36000, 25000, 6000, 36000, 50000, 46000, 3000, 37000, 40000, 30000, 48800, 19700, 16000, 14000, 12000, 25000, 25000, 28600, 17000, 31200, 57000, 23000, 23500, 46000, 18700, 26700, 9900, 16000, 3000, 52000, 51000, 14000, 14400, 27000, 26000, 60000, 25000, 6000, 20000, 3000, 69000, 24800, 12000, 3100, 18000, 20000, 267000, 28000, 9800, 18200, 80000, 6800, 21100, 20000, 68000, 20000, 45000, 8000, 40000, 31900, 28000, 24000, 2000, 32000, 11000, 20000, 5900, 16100, 23900, 40000, 37500, 11000, 55000, 37500, 60000, 23000, 9500, 34500, 4000, 9000, 11200, 35200, 30000, 18000, 21800, 19700, 16700, 12500, 11300, 4000, 39000, 32000, 14000, 65000, 50000, 2000, 30400, 22000, 1600, 56000, 40000, 85000, 9000, 10000, 19000, 5300, 5200, 43000, 60000, 50000, 38000, 267000, 15600, 1800, 17000, 45000, 31000, 5000, 8000, 43000, 103000, 45000, 8800, 26000, 47000, 40000, 8000]
    >>> # udacity class data shows that people earn on average $1.8K more for each year of age and start with a $21K deficit
    >>> regressionplot(age, wage)   # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    (1768.275..., -21991.9...)
    >>> # Gainseville, FL census data shows 14 more new homes are built each year, starting with 517 completed in 1991
    >>> regression_and_plot([483, 576, 529, 551, 529, 551, 663, 639, 704, 675, 601, 621, 630, 778, 831, 610])  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    (14.213..., 516.588...)
    """
    if y is None:
        y = x
        x = range(len(x))
    if not isinstance(x[0], (float, int, np.float64, np.float32)):
        x = [row[0] for row in x]
    A = np.vstack([np.array(x), np.ones(len(x))]).T
    fit = np.linalg.lstsq(A, y)
    poly = regressionplot(x, y, fit)
    return poly


def regressionplot(x, y, fit=None):
    """
    Plot a 2-D linear regression (y = slope * x + offset) overlayed over the raw data samples
    """
    if fit is None:
        fit = [(1, 0), None, None, None]
    if not isinstance(x[0], (float, int, np.float64, np.float32)):
        x = [row[0] for row in x]
    poly = fit[0][0], fit[0][-1]
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x, poly[0] * np.array(x) + poly[-1], 'r-', x, y, 'o', markersize=5)
    plt.legend(['%+.2g * x + %.2g' % poly, 'Samples'])
    ax.grid(True)
    plt.draw()
    return poly
