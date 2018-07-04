# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

###############################################################################
#
# spline = a cubic b spline with equally spaced nodes
#
###############################################################################

class spline:
  
# spline with support in [x_min,x_max] and n >= 8 intervals
   
    def __init__(self, weights, x_min = 0, x_max = 1):
      assert x_max > x_min, 'x_max must be greater than x_min'
      assert type(weights) == np.ndarray, 'weight must be a numpy array'
      assert len(weights) >= 8, 'there must be at least 8 weights'
      dx = x_max - x_min
      self.n = len(weights) - 3
      self.scale = self.n/dx
      self.x_min = x_min
      self.x_max = x_max
      self.weights = weights

    def locate(self, t):
      if t <= self.x_min:
        return [0.0, 0]

      nw = len(self.weights)
      
      if t >= self.x_max:
        return [1.0, nw - 4]

      dt = self.scale * (t - self.x_min)
      
      if dt >= nw: 
        dt = 1.0
        return [1.0, nw - 1]

      bin = int(np.floor(dt))
      dt -= bin
      return [dt,bin]
    
    
    def piece_0_0(self, x):
      return  240 - x * (720 - x * (720 - x * 240))    
      
    def piece_1_0(self, x): 
      return x * (720 - x * (1080 - x * 420))    
    
    def piece_1_1(self,x): 
      return 60 - x * (180 - x * (180 - x * 60))   
    
    def piece_2_0(self,x):
      return x * x * (360 - 220 * x)

    def piece_2_1(self,x):
      return 140 + x * (60 - x * (300 - x * 140))    
    
    def piece_2_2(self,x):
      return 40 - x * (120 - x * (120 - x * 40))    

    def piece_3_0(self,x):
      return 40  * x * x * x    

    def piece_3_1(self,x):    
      return 40 + x * (120 + x * (120 - x * 120))

    def piece_3_2(self,x):    
      return 160 - x * x * (240 - x * 120)    

    def piece_3_3(self,x):    
      return 40 - x * (120 - x * (120 - x * 40))   
    
    def piece_4_1(self,x):
      return  40 * x * x * x

    def piece_4_2(self,x):   
      return 40 + x * (120 + x * (120 - x * 140))    
      
    def piece_4_3(self,x):
      return 140 - x * (60 + x * (300 - x * 220))    
    
    def piece_5_2(self,x):
      return 60 * x * x * x    
    
    def piece_5_3(self,x):
      return 60 + x * (180 + x * (180 - x * 420))    
    
    def piece_6_3(self,x):
      return 240  * x * x * x
      
    def eval(self,x):
      dt, bin = self.locate(x)
      if bin <= 2:
        if bin <= 0:
          a = self.piece_0_0(dt) * self.weights[0] + self.piece_1_0(dt) * self.weights[1]
          return a + self.piece_2_0(dt) * self.weights[2] + self.piece_3_0(dt) * self.weights[3]
#             
        if bin == 1:
          a = self.piece_1_1(dt) * self.weights[1] + self.piece_2_1(dt) * self.weights[2]
          return a + self.piece_3_1(dt) * self.weights[3] + self.piece_3_0(dt) * self.weights[4]
        else: # bin = 2
          a = self.piece_2_2(dt) * self.weights[2] + self.piece_3_2(dt) * self.weights[3]
          return a + self.piece_3_1(dt) * self.weights[4] + self.piece_3_0(dt) * self.weights[5]
          
      # now bin > 2
      nw = len(self.weights)
      if bin >= nw - 6:
        if bin == nw - 6:
          a = self.piece_3_3(dt) * self.weights[bin] + self.piece_3_2(dt) * self.weights[bin + 1]
          return a + self.piece_3_1(dt) * self.weights[bin + 2] + self.piece_4_1(dt) * self.weights[bin + 3]
    
        if bin == nw - 5:
          a = self.piece_3_3(dt) * self.weights[bin] + self.piece_3_2(dt) * self.weights[bin + 1] 
          return a + self.piece_4_2(dt) * self.weights[bin + 2] + self.piece_5_2(dt) * self.weights[bin + 3]
          
        # now bin = nw - 4 
        a = self.piece_3_3(dt) * self.weights[bin] + self.piece_4_3(dt) * self.weights[bin + 1] 
        return a + self.piece_5_3(dt) * self.weights[bin + 2] + self.piece_6_3(dt) * self.weights[bin + 3] 
        
    # finally, the normal case 3 < bin < nw - 6
      a = self.piece_3_3(dt) * self.weights[bin] + self.piece_3_2(dt) * self.weights[bin + 1]
      return a +  self.piece_3_1(dt) * self.weights[bin + 2] + self.piece_3_0(dt) * self.weights[bin + 3]

    def __call__(self, x):
      if type(x) == np.ndarray:
        y = x.copy()
        for i in range(len(x)):
          y[i] = self.eval(x[i])
        return y
      else:
        return self.eval(x)

###############################################################################
#
# d_spline = the derivative of the previous spline
#
###############################################################################

class d_spline:
  
# the derivative of an spline with support in [x_min,x_max] and n >= 8 intervals
   
    def __init__(self, weights, x_min = 0, x_max = 1):
      assert x_max > x_min, 'x_max must be greater than x_min'
      assert type(weights) == np.ndarray, 'weight must be a numpy array'
      assert len(weights) >= 8, 'there must be at least 8 weights'
      dx = x_max - x_min
      self.n = len(weights) - 3
      self.scale = self.n/dx
      self.x_min = x_min
      self.x_max = x_max
      self.weights = weights

    def locate(self, t):
      if t <= self.x_min:
        return [0.0, 0]

      nw = len(self.weights)
      
      if t >= self.x_max:
        return [1.0, nw - 4]

      dt = self.scale * (t - self.x_min)
      
      if dt >= nw: 
        dt = 1.0
        return [1.0, nw - 1]

      bin = int(np.floor(dt))
      dt -= bin
      return [dt,bin]
        
    def piece_0_0(self, x):
      return -720 + x * (1440 - x * 720)
      
    def piece_1_0(self, x): 
      return 720 - x * (2160 - x * 1260)
    
    def piece_1_1(self,x): 
      return (-180 + x * (360 - x * 180))
    
    def piece_2_0(self,x):
      return x * (720 - 660 * x)

    def piece_2_1(self,x):
      return 60 - x * (600 - x * 420)
   
    def piece_2_2(self,x):
      return -120 + x * (240 - x * 120)

    def piece_3_0(self,x):
      return 120 * x * x;

    def piece_3_1(self,x):    
      return 120 + x * (240 - x * 360)

    def piece_3_2(self,x):    
      return x * (-480 + x * 360)

    def piece_3_3(self,x):    
      return -120 + x * (240 - x * 120)
    
    def piece_4_1(self,x):
      return 120 * x * x

    def piece_4_2(self,x):   
      return 120 + x * (240 - x * 420)
      
    def piece_4_3(self,x):
      return  -60 - x * (600 - x * 660)
    
    def piece_5_2(self,x):
      return 180 * x * x
    
    def piece_5_3(self,x):
      return 180 + x * (360 - x * 1260)
    
    def piece_6_3(self,x):
      return 720  * x * x;
      
    def eval(self,x):
      dt, bin = self.locate(x)
      if bin <= 2:
        if bin <= 0:
          a = self.piece_0_0(dt) * self.weights[0] + self.piece_1_0(dt) * self.weights[1]
          return a + self.piece_2_0(dt) * self.weights[2] + self.piece_3_0(dt) * self.weights[3]
#             
        if bin == 1:
          a = self.piece_1_1(dt) * self.weights[1] + self.piece_2_1(dt) * self.weights[2]
          return a + self.piece_3_1(dt) * self.weights[3] + self.piece_3_0(dt) * self.weights[4]
        else: # bin = 2
          a = self.piece_2_2(dt) * self.weights[2] + self.piece_3_2(dt) * self.weights[3]
          return a + self.piece_3_1(dt) * self.weights[4] + self.piece_3_0(dt) * self.weights[5]
          
      # now bin > 2
      nw = len(self.weights)
      if bin >= nw - 6:
        if bin == nw - 6:
          a = self.piece_3_3(dt) * self.weights[bin] + self.piece_3_2(dt) * self.weights[bin + 1]
          return a + self.piece_3_1(dt) * self.weights[bin + 2] + self.piece_4_1(dt) * self.weights[bin + 3]
    
        if bin == nw - 5:
          a = self.piece_3_3(dt) * self.weights[bin] + self.piece_3_2(dt) * self.weights[bin + 1] 
          return a + self.piece_4_2(dt) * self.weights[bin + 2] + self.piece_5_2(dt) * self.weights[bin + 3]
          
        # now bin = nw - 4 
        a = self.piece_3_3(dt) * self.weights[bin] + self.piece_4_3(dt) * self.weights[bin + 1] 
        return a + self.piece_5_3(dt) * self.weights[bin + 2] + self.piece_6_3(dt) * self.weights[bin + 3] 
        
    # finally, the normal case 3 < bin < nw - 6
      a = self.piece_3_3(dt) * self.weights[bin] + self.piece_3_2(dt) * self.weights[bin + 1]
      return a +  self.piece_3_1(dt) * self.weights[bin + 2] + self.piece_3_0(dt) * self.weights[bin + 3]

    def __call__(self, x):
      if type(x) == np.ndarray:
        y = x.copy()
        for i in range(len(x)):
          y[i] = self.eval(x[i])
        return y
      else:
        return self.eval(x)

###############################################################################
#
# d2_spline = the second derivative of the previous spline
#
###############################################################################

class d2_spline:
  
# the derivative of an spline with support in [x_min,x_max] and n >= 8 intervals
   
    def __init__(self, weights, x_min = 0, x_max = 1):
      assert x_max > x_min, 'x_max must be greater than x_min'
      assert type(weights) == np.ndarray, 'weight must be a numpy array'
      assert len(weights) >= 8, 'there must be at least 8 weights'
      dx = x_max - x_min
      self.n = len(weights) - 3
      self.scale = self.n/dx
      self.x_min = x_min
      self.x_max = x_max
      self.weights = weights

    def locate(self, t):
      if t <= self.x_min:
        return [0.0, 0]

      nw = len(self.weights)
      
      if t >= self.x_max:
        return [1.0, nw - 4]

      dt = self.scale * (t - self.x_min)
      
      if dt >= nw: 
        dt = 1.0
        return [1.0, nw - 1]

      bin = int(np.floor(dt))
      dt -= bin
      return [dt,bin]
        
    def piece_0_0(self, x):
      return 1440 - x * 1440
      
    def piece_1_0(self, x): 
      return -2160 + x * 2520
    
    def piece_1_1(self,x): 
      return 360 - x * 360
    
    def piece_2_0(self,x):
      return 720 - x * 1320

    def piece_2_1(self,x):
      return -600 + x * 840
   
    def piece_2_2(self,x):
      return 240 - x * 240

    def piece_3_0(self,x):
      return 240 * x

    def piece_3_1(self,x):
      return 240 - x * 720

    def piece_3_2(self,x):    
      return -480 + x * 720

    def piece_3_3(self,x):
      return 240 - x * 240
    
    def piece_4_1(self,x):
      return 240 * x

    def piece_4_2(self,x):   
      return 240 - x * 840
      
    def piece_4_3(self,x):
      return -600 + x * 1320
      
    def piece_5_2(self,x):
      return 360 * x
    
    def piece_5_3(self,x):
      return 360 - x * 2520
    
    def piece_6_3(self,x):
      return 1440 * x
      
    def eval(self,x):
      dt, bin = self.locate(x)
      if bin <= 2:
        if bin <= 0:
          a = self.piece_0_0(dt) * self.weights[0] + self.piece_1_0(dt) * self.weights[1]
          return a + self.piece_2_0(dt) * self.weights[2] + self.piece_3_0(dt) * self.weights[3]
             
        if bin == 1:
          a = self.piece_1_1(dt) * self.weights[1] + self.piece_2_1(dt) * self.weights[2]
          return a + self.piece_3_1(dt) * self.weights[3] + self.piece_3_0(dt) * self.weights[4]
        else: # bin = 2
          a = self.piece_2_2(dt) * self.weights[2] + self.piece_3_2(dt) * self.weights[3]
          return a + self.piece_3_1(dt) * self.weights[4] + self.piece_3_0(dt) * self.weights[5]
          
      # now bin > 2
      nw = len(self.weights)
      if bin >= nw - 6:
        if bin == nw - 6:
          a = self.piece_3_3(dt) * self.weights[bin] + self.piece_3_2(dt) * self.weights[bin + 1]
          return a + self.piece_3_1(dt) * self.weights[bin + 2] + self.piece_4_1(dt) * self.weights[bin + 3]
    
        if bin == nw - 5:
          a = self.piece_3_3(dt) * self.weights[bin] + self.piece_3_2(dt) * self.weights[bin + 1] 
          return a + self.piece_4_2(dt) * self.weights[bin + 2] + self.piece_5_2(dt) * self.weights[bin + 3]
          
        # now bin = nw - 4 
        a = self.piece_3_3(dt) * self.weights[bin] + self.piece_4_3(dt) * self.weights[bin + 1] 
        return a + self.piece_5_3(dt) * self.weights[bin + 2] + self.piece_6_3(dt) * self.weights[bin + 3] 
        
    # finally, the normal case 3 < bin < nw - 6
      a = self.piece_3_3(dt) * self.weights[bin] + self.piece_3_2(dt) * self.weights[bin + 1]
      return a +  self.piece_3_1(dt) * self.weights[bin + 2] + self.piece_3_0(dt) * self.weights[bin + 3]

    def __call__(self, x):
      if type(x) == np.ndarray:
        y = x.copy()
        for i in range(len(x)):
          y[i] = self.eval(x[i])
        return y
      else:
        return self.eval(x)
        
# use  '%matplotlib qt' for plotting on external window

#plotting a 3d curve

def curve(t, spx, spy, spz):
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.plot(spx(t), spy(t), spz(t))
  ax.legend()
  plt.show()




