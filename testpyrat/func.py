__all__ = ['RatPartFloat','createObjectDf', 'achaArquivos', 'interact' , 'RatDistanceSpeed', 'scaleConverter', 'rltr']

def RatPartFloat (dataX,dataY):
  import pandas as pd
  import numpy as np
  x = dataX[3:]
  y = dataY[3:]

  x = pd.to_numeric(x)
  resultX  = x.to_numpy()

  y = pd.to_numeric(y)
  resultY = y.to_numpy()
  return resultX,resultY

def createObjectDf(numObjects):
  import pandas as pd
  objects = pd.DataFrame(columns=['obj','center_x','center_y', 'radius', 'a_x', 'a_y' , 'height', 'width'])
  for i in range(numObjects):
    print('Digite o tipo do objeto '+ str(i+1) + " (0 - circulo, 1 - quadrado):")
    objectType = int(input())
    if objectType == 0:
      print('Digite o valor de X do centro do objeto ' + str(i+1) + ':')
      centerX = int(input())
      print('Digite o valor de Y do centro do objeto ' + str(i+1) + ':')
      centerY = int(input())
      print('Digite o valor do raio do objeto ' + str(i+1) + ':')
      radius = int(input())
      df2 = pd.DataFrame([[objectType, centerX, centerY,radius,'null','null','null','null']], columns=['obj','center_x','center_y', 'radius', 'a_x', 'a_y' , 'height', 'width'])
    else:
      print('Digite o valor de X do vértice inferior esquerdo do objeto ' + str(i+1) + ':')
      aX = int(input())
      print('Digite o valor de Y do vértice inferior esquerdo do objeto ' + str(i+1) + ':')
      aY = int(input())
      print('Digite o valor da altura do objeto ' + str(i+1) + ':')
      height = int(input())
      print('Digite o valor da largura do objeto ' + str(i+1) + ':')
      width = int(input())
      df2 = pd.DataFrame([[objectType, 'null','null' , 'null' ,aX,aY,height,width]], columns=['obj','center_x','center_y', 'radius', 'a_x', 'a_y' , 'height', 'width'])
    objects = objects.append(df2, ignore_index=True)
  return objects

def achaArquivos(root, elements):
  import os

  c = []
  b = []
  for path, subdirs, files in os.walk(root):
    matching = [s for s in files if elements in s]
    c = c + [os.path.join(root, names) for names in matching]
    b = b + matching
  return c, b

def interact(objects, dataX, dataY):
  import numpy as np
  import pandas as pd
  numObjects = len(objects.index)
  interact = np.zeros(len(dataX))

  for i in range(len(interact)):
    for j in range(numObjects):
      if objects['obj'][0] == 0:
        if ((dataX[i] - objects['center_x'][j])**2 + (dataY[i] - objects['center_y'][j])**2 <= objects['radius'][j]**2):
          interact[i] = j +1
      else:
        if objects['a_x'][j] <= dataX[i] <= (objects['a_x'][j] + objects['width'][j]) and objects['a_y'][j] <= dataY[i] <= (objects['a_y'][j] + objects['height'][j]): 
          interact[i] = j +1
    interactsDf = pd.DataFrame(columns=['start','end','obj'])
  obj = 0
  start = 0
  end = 0
  fps =30
  for i in range(len(interact)):
    if obj != interact[i]:
      end = ((i-1)/fps)
      df = pd.DataFrame([[start,end,obj]],columns=['start','end','obj'])
      obj =  interact[i]
      start = end
      interactsDf = interactsDf.append(df, ignore_index=True)

  return interactsDf

def RatDistanceSpeed (dataXfloat,dataYfloat,fps=30, filter = .5):
  import math
  import numpy as np
  distance = 0
  teste = []
  filtro = []

  esquerda = dataXfloat.min()
  direita = dataXfloat.max()
  baixo = dataYfloat.min()
  cima = dataYfloat.max()

  for i in range(len(dataYfloat)-1):
    caixa = (dataYfloat[i] - cima)
    teste.append(distance + (math.sqrt( ((dataXfloat[i]-dataXfloat[i+1])**2)+((dataYfloat[i]-dataYfloat[i+1])**2) )) )   
    if teste[i] <= filter:
      filtro.append(0)
    else:
      filtro.append(teste[i])

  filtro = sum(np.asarray(filtro))
  VideoInSec =  dataXfloat.size/fps
  dist = filtro
  speedCMperS = filtro/VideoInSec
  speedKMperH = ((filtro/VideoInSec)/27.778)


  return dist, speedCMperS, speedKMperH

def scaleConverter(dado, pixel_max,pixel_min,max_real, min_real=0):
  return min_real + ((dado-pixel_min)/(pixel_max-pixel_min)) * (max_real-min_real)

def rltr (path, key, bodypart, objs, max_realX, max_realY, min_realX = 0, min_realY = 0, fps = 30):
  import numpy as np
  import pandas as pd
  paths, files = achaArquivos(path, key)
  relatorio = pd.DataFrame(columns=['file','time','dist', 'speed'])
  
   
  for i in range(len(objs)):
    relatorio["obj{0}".format(i+1)] = []
    relatorio["time_obj{0}".format(i+1)] = []
  for i in range(len(paths)):
    lista = [files[i]]
    data = pd.read_csv(paths[i])
    ind, = np.where(data.iloc[0].values == bodypart)
    dataX =  pd.to_numeric(data.iloc[2:, ind[0]]).values
    dataY = pd.to_numeric(data.iloc[2:, ind[1]]).values
     
    tempo = time(dataX, fps)
    lista.append(tempo)

    pixel_maxX = dataX.max()
    pixel_minX = dataX.min()
    pixel_maxY = dataY.max()
    pixel_minY = dataY.min()
    newX = scaleConverter(dataX,pixel_maxX,pixel_minX,max_realX, min_realX)
    newY = scaleConverter(dataY,pixel_maxY,pixel_minY,max_realY, min_realY)

    dist, vMedia, vMediaKM = RatDistanceSpeed(newX, newY)
    lista.append(dist)
    lista.append(vMedia)

    interacts = interact2(objs, dataX, dataY)
    for i in range(len(objs)):
      lista.append(interacts["obj"][interacts["obj"] == i+1].count())
      lista.append((interacts["end"][interacts["obj"] == i+1]-interacts["start"][interacts["obj"] == i+1]).sum())
    relatorio_temp = pd.DataFrame([lista], columns=relatorio.columns)
    relatorio = relatorio.append(relatorio_temp, ignore_index=True)
  return relatorio
