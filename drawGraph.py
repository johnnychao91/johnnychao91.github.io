from PyQt5.QtWidgets import QLineEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import LassoSelector, SpanSelector
from matplotlib.path import Path as mplPath
import pandas as pd
import numpy as np

def func_PrcData4Plot(dt: list, reshapeYX: tuple , mz: float, tlrnc: float):
    ### create pandas.Dataframe from datum ###
    # special case: mz == -1 or tlrnc == -1: merge without tolerance and mz
    if mz == -1 or mz == -2 or tlrnc == -1 or tlrnc == -2:
        mask = None
        lstTmp = dt
    # mask is a boolen array. 
    # The mask will be True when the values in dt[0] that are greater than (mz - tlrnc)
    #  and less than (mz + tlrnc), and False for the rest.
    else:
        mask = (dt[0] > (mz-tlrnc)) & (dt[0] < (mz+tlrnc))
        lstTmp = []
        for i in dt:
            lstTmp.append(i[mask])
    ### end region ###
    
    rslt = pd.DataFrame([i.sum() for i in lstTmp])
    ### 文 ###
    # print("rslt : ")
    # for i in range(rslt.shape[0]):
    #     print(rslt.index[i]," ", rslt[i])
    # 橫向相加
    ###
    # special case: mz == -2 and tlrnc == -2: reassign outliers and take log
    if mz == -2 and tlrnc == -2:
        npary = rslt.loc[1:].values
        ### 文 ###
        # print("npary : ",npary)
        # 除了第0列m/z的總和以外的值
        ###
        sortedNpary = np.sort(npary)
        ### check-文 ###
        # print(sortedNpary)
        # print("size : ",sortedNpary.size)
        # for i in range(len(sortedNpary)):
        #     print(sortedNpary[i],end=" ")
        diff_1=[]
        for i in range(0,len(sortedNpary)-2,1):
            diff_i = sortedNpary[i]-sortedNpary[i+1]
            diff_1.append(diff_i)
            if i == 2427:
                print("差異最多 : ",diff_i)
                print(sortedNpary[i]," - ",sortedNpary[i+1])
        for i in range(len(diff_1)):
            if i == np.argmin(diff_1,axis=0):
                print("差異index : ",i) # 2427
        print("show in func_PrcData4Plot function in drawGraph.py")               
        # print(np.amax(sortedNpary))
        # print(np.amin(sortedNpary))
        ###
        Q1 = np.percentile(sortedNpary, 25)
        Q3 = np.percentile(sortedNpary, 75)
        ### check-文 ###
        # print("Q1 : ",Q1)
        # print("Q3 : ",Q3)
        # 4-3 # Q1 :  3808.0 # Q3 :  5326.25
        ###
        IQR = Q3 - Q1
        upperBound = Q3 + 3 * IQR
        upperOutliers = npary > upperBound
        lowerBound = Q1 - 1.5 * IQR
        lowerOutliers = npary < lowerBound
        npary[upperOutliers] = 0
        npary[upperOutliers] = npary.max()
        npary[lowerOutliers] = 0

        npary = np.log10(npary + 0.0001)
        mask = npary < 0
        npary[mask] = 0

        ### check-文 ###
        # print(np.amax(npary))
        # print(np.amin(npary))
        # print("Q1 : ",np.percentile(npary, 25))
        # print("Q3 : ",np.percentile(npary, 75))
        ###
        # log10(10000)=4
        # log10(9000)=3.95
        # log10(8000)=3.90
        # log10(7000)=3.845
        # log10(6000)=3.778
        # log10(5000)=3.69897

        npary = npary.reshape(reshapeYX)
        return npary
    

    ### preprocess the dataframe for plot the pixelplot ###
    ## accumulating version ##
    # rslt = pd.DataFrame([i.sum() for i in lstTmp])
    npary = rslt.loc[1:].values.reshape(reshapeYX)
    
    ## normalization version ##
    # rslt = pd.DataFrame([i.sum()/i.__len__() for i in lstTmp])
    # min = rslt.loc[1:].min()[0]
    # max = rslt.loc[1:].max()[0]
    # scope = abs(max-min)
    # npary = rslt.loc[1:].apply(
    #     lambda x: x-min).div(scope).values.reshape(reshapeYX)
    ### end region ###
    return npary


class MatplotlibWidget(FigureCanvas):
    def __init__(self, lineEdit:QLineEdit, parent=None, **kwargs):
        self._commsLineEdit = lineEdit
        self._figure = Figure()
        # self._figure.set_facecolor("gray")
        self._axs = self._figure.add_subplot(111)
        # self._axs = self._figure.add_subplot(1,1,1) #1列1行之第x張的位置

        super(MatplotlibWidget, self).__init__(self._figure)
        self.setParent(parent)
        self._mouseXY = None
        self._roi = []
        self._pixelYX = None
        self._barGraphMinMax = None
        self._figure.canvas.mpl_connect('button_press_event', self.onClicked)
        self._lassoSelector = None
        self._lassoSelectorState = False
        self._spanSelector = None
        self._spanSelectorState = False
        self._scatter = None # merge
        
    def getMouseXY(self):
        return self._mouseXY
    def getROI(self):
        return self._roi
    def getBarGraphMinMax(self):
        return self._barGraphMinMax
    def getLassoSelectorState(self):
        return self._lassoSelectorState
    def getSpanSelectorState(self):
        return self._spanSelectorState
    ### merge ###
    def setPixelYX(self, pixelYX):
        self._pixelYX = pixelYX
    ###

    def onClicked(self, event):
        try:
            x = int(event.xdata)
            y = int(event.ydata)
            self._mouseXY = [x,y]
            self._commsLineEdit.setText(f"{(x,y)}")
        except Exception as e:
            print(f"drawGraph.py > MatplotlibWidget class > onClicked method {e}")
            print('Please clicked on the Pixel Plot, not clicked out outside, thanks.') # 文
        # print('x: {} and y: {}'.format(x, y))

    def onLassoSelect(self, vertices): # merge
        try:
            pixelPth = mplPath(vertices)
            selected_pixels = []
            for i in range(self._pixelYX[1]):
                for j in range(self._pixelYX[0]):
                    if pixelPth.contains_point([i, j]):
                        selected_pixels.append((i, j))###新增單個 圈選範圍
            self._roi = set(self._roi + selected_pixels)###多個 圈選範圍 ==>聯集
            self._roi = sorted(self._roi, key=lambda x: (x[0], x[1]))
            ### merge ###
            for _ in selected_pixels:
                self._scatter = self._axs.scatter(_[0],_[1], c='none', marker='s', s=10, edgecolors=['red'])
            self._figure.canvas.draw()
            ###
        except Exception as e:
            print(f"drawGraph.py > MatplotlibWidget class > onLassoSelect method {e}")

    def onSpanSelect(self,xmin ,xmax):
        try: # xmin, xmax are numpy.float64
            self._barGraphMinMax = [xmin, xmax]
            # print(f"xmin: {xmin}, xmax: {xmax}")
        except Exception as e:
            print(f"drawGraph.py > MatplotlibWidget class > onSpanSelect method {e}")
    
    """
    def toggleSpanSelector(self):
        self._spanSelectorState = not self._spanSelectorState
        if self._spanSelectorState:
            self._spanSelector = SpanSelector(
                            self._axs, self.onSpanSelect,
                            "horizontal", useblit=True,
                            props=dict(alpha=0.5, facecolor="tab:blue"),
                            interactive=True, drag_from_anywhere=True
                            )
        else:
            self._spanSelector.set_active(False)
            self._spanSelector.set_visible(False)
        self._figure.canvas.draw()
    """

    def toggleSpanSelector(self):
        self._spanSelectorState = not self._spanSelectorState
        if self._spanSelectorState:
            self._spanSelector = SpanSelector(
                            self._axs, self.onSpanSelect,
                            "horizontal", useblit=True,
                            props=dict(alpha=0.5, facecolor="tab:blue"),
                            interactive=True, drag_from_anywhere=True
                            )
        else:
            self._spanSelector.set_active(False)
            self._spanSelector.set_visible(False)
        self._figure.canvas.draw()
    
    def toggleLassoSelector(self):
        self._lassoSelectorState = not self._lassoSelectorState
        self._roi = []
        if self._lassoSelectorState:
            self._lassoSelector = LassoSelector(ax=self._axs, onselect=self.onLassoSelect)
        else:
            self._lassoSelector.set_active(False)
            self._lassoSelector.set_visible(False)
        self._figure.canvas.draw()
    
    def plotPixelplot(self, pixel_array):
        self._axs.clear()
        # self._figure.clf()
        pcm = self._axs.pcolor(pixel_array, cmap='inferno')
        # self._axs.imshow(pixel_array, cmap='inferno')
        self._axs.set_aspect('equal')
        # self._figure.colorbar(pcm, ax=self._axs)
        self._figure.canvas.draw()
    
    def plotBargraph(self, bargraph_x, bargraph_y):
        self._axs.clear()
        ### merge ###
        self._axs.set_xlabel("m/z", loc="left")
        self._axs.xaxis.set_label_coords(1.0, -0.025)
        self._axs.set_ylabel("intensity")
        self._axs.grid(axis='y')
        ###
        self._axs.stem(bargraph_x, bargraph_y, markerfmt=" ")
        self._figure.canvas.draw()