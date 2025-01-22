import numpy as np
import wx

import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar

class ImagenMatplotlibLibre(wx.Dialog):
    def __init__(self, parent, id=-1, dpi=None, **kwargs):
        wx.Dialog.__init__(self, parent, id=id, style= wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.figure = mpl.figure.Figure(dpi=dpi)
        self.ax=self.figure.gca()
        self.arr=''
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()
        self.Text = wx.StaticText( self, wx.ID_ANY, u"  Available Channels  ", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.Text.Wrap( -1 )
        mouseMoveID = self.canvas.mpl_connect('motion_notify_event',self.onMotion)
        self.Fit()
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        sizer.Add(self.toolbar, 0, wx.LEFT | wx.EXPAND)
        sizer.Add(self.Text,0, wx.LEFT | wx.EXPAND)
        self.SetSizer(sizer)
        self.toolbar.update()
        
    def onMotion(self, evt):
        xdata = evt.xdata
        ydata = evt.ydata
        try:
            x = round(xdata,4)
            y = round(ydata,4)
        except:
            x = ""
            y = ""
        if self.arr=='':    
            self.Text.SetLabelText("%s , %s " % (x,y))
        else:
            self.Text.SetLabelText("%s , %s, %s " % (x,y,self.arr[int(x),int(y)]))    
        
