#!/usr/bin/python
# vi: set fileencoding=utf-8 :
import numpy
import random
'''
particle filterは実装できませんでした。このプログラムはネットにあるparticle filterの一例を改造している途中のものです。
Global Economic Monitor (GEM) Commoditiesのデータを使えるにしましたが、元のデータの種類と違うので、おそらくparticle filterの機能はない。
runfileで実行できます
'''
def resample(weights):
  n = len(weights)
  indices = []
  C = [0.] + [sum(weights[:i+1]) for i in range(n)]
  u0, j = random(), 0
  for u in [(u0+i)/n for i in range(n)]:
    while u > C[j]:
      j+=1
    indices.append(j-1)
  return indices


def particlefilter(sequence, pos, stepsize, n):
  seq = iter(sequence)
  x = numpy.ones((n, 2), int) * pos
  print pos                   
  f0 = seq.next()[tuple(pos)] * numpy.ones(n)   
  yield pos, x, numpy.ones(n)/n
  print pos                       
  for im in seq:
    x += numpy.random.uniform(-stepsize, stepsize, x.shape)
    x  = x.clip(numpy.zeros(1), numpy.array(im.shape)-1).astype(int)
    f  = im[tuple(x.T)]
    w  = 1./(1. + (f0-f)**2)
    w /= sum(w)                         
    yield sum(x.T*w, axis=1), x, w              
    if 1./sum(w**2) < n/2.:                     
      x  = x[resample(w),:]                     
if __name__ == "__main__":
  from pylab import *
  from itertools import izip
  import time
  ion()
  seq = [ im for im in zeros((20,240,320), int)]     
  x0 = array([100])
  input=open("GEM.csv",'r')
  dataArray=input.readlines()
  for i in range(len(dataArray)):
      dataArray[i]=dataArray[i].split()
      dataArray[i]=dataArray[i][0].split(",")
      for j in range(20):
          dataArray[i][j]=[float(dataArray[i][j+5])]
      del dataArray[i][20:]
  xs = vstack(dataArray).T
  print xs
  for t, x in enumerate(xs):
    xslice = slice(x[0]-8, x[0]+8)
    seq[t][xslice] = 255

  for im, p in izip(seq, particlefilter(seq, x0, 8, 320)): 
    pos, xs, ws = p
    position_overlay = zeros_like(im)
    position_overlay[tuple(pos)] = 1
    particle_overlay = zeros_like(im)
    particle_overlay[tuple(xs.T)] = 1
    hold(True)
    draw()
    time.sleep(0.3)
    clf()                                           # Causes flickering, but without the spy plots aren't overwritten
    imshow(im,cmap=cm.gray)                         # Plot the image
    spy(position_overlay, marker='.', color='b')    # Plot the expected position
    spy(particle_overlay, marker=',', color='r')    # Plot the particles
  show()