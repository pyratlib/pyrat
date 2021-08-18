import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

def Trajectory (data,bodyPartTraj,bodyPartBox,cmapType='viridis',figureTitle=None, 
                hSize=6,wSize=8,fontsize=15,invertY=True,saveName=None,figformat=".eps"):
    """
    Plots the trajectory of the determined body part.

    Parameters
    ----------
    data : pandas DataFrame
        The input tracking data.
    bodyPartTraj : str
        Body part you want to plot the tracking.
    bodyPartBox : str
        The body part you want to use to estimate the limits of the environment, 
        usually the base of the tail is the most suitable for this determination.
    cmapType : str, optional
        matplotlib colormap.
    figureTitle : str, optional
        Figure title.
    hSize : int, optional
        Determine the figure height size (x).
    wSize : int, optional
        Determine the figure width size (y).
    fontsize : int, optional
        Determine of all font sizes.
    invertY : bool, optional
        Determine if de Y axis will be inverted (used for DLC output).
    saveName : str, optional
        Determine the save name of the plot.        
    figformat : str, optional
        Determines the type of file that will be saved. Used as base the ".eps", 
        which may be another supported by matplotlib. 

    Returns
    -------
    out : plot
        The output of the function is the figure with the tracking plot of the 
        selected body part.

    See Also
    --------
    For more information and usage examples: https://github.com/pyratlib/pyrat

    Notes
    -----
    This function was developed based on DLC outputs and is able to support 
    matplotlib configurations."""
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib import cm
    values = (data.iloc[2:,1:].values).astype(np.float)
    lista1 = (data.iloc[0][1:].values +" - " + data.iloc[1][1:].values).tolist()

    x = values[:,lista1.index(bodyPartTraj+" - x")]
    y = values[:,lista1.index(bodyPartTraj+" - y")]

    cmap = plt.get_cmap(cmapType)

    c = np.linspace(0, x.size/30, x.size)
    esquerda = values[:,lista1.index(bodyPartBox+" - x")].min()
    direita = values[:,lista1.index(bodyPartBox+" - x")].max()
    baixo = values[:,lista1.index(bodyPartBox+" - y")].min()
    cima = values[:,lista1.index(bodyPartBox+" - y")].max()

    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(wSize, hSize), dpi=80)
    plt.title(figureTitle, fontsize=fontsize)
    plt.scatter(x, y, c=c, cmap=cmap, s=3)
    plt.plot([esquerda,esquerda] , [baixo,cima],"r")
    plt.plot([esquerda,direita]  , [cima,cima],"r")
    plt.plot([direita,direita]   , [cima,baixo],"r")
    plt.plot([direita,esquerda]  , [baixo,baixo],"r")
    #ax1.set_ylim(480,0)
    #ax1.set_xlim(0,640)
    cb = plt.colorbar()
    #plt.xticks(rotation=45)
    #plt.yticks(rotation=90)

    if invertY == True:
        plt.gca().invert_yaxis()
    cb.set_label('Time (s)',fontsize=fontsize)
    cb.ax.tick_params(labelsize=fontsize*0.8)
    plt.xlabel("X (px)",fontsize=fontsize)
    plt.ylabel("Y (px)",fontsize=fontsize)
    plt.xticks(fontsize = fontsize*0.8)
    plt.yticks(fontsize = fontsize*0.8)
    if saveName != None:
        plt.savefig(saveName+figformat)
    plt.show()


def Heatmap(data,bodyPart,cmapType='viridis',figureTitle=None,hSize=6,wSize =8,
            bins=40,vmax=1000,fontsize=15,invertY=True,saveName=None,figformat=".eps"):
    """
    Plots the trajectory heatmap of the determined body part.

    Parameters
    ----------
    data : pandas DataFrame
        The input tracking data.
    bodyPart : str
        Body part you want to plot the heatmap.
    cmapType : str, optional
        matplotlib colormap.
    figureTitle : str, optional
        Figure title.
    hSize : int, optional
        Determine the figure height size (x).
    wSize : int, optional
        Determine the figure width size (y).
    bins : int, optional
        Determine the heatmap resolution, the higher the value, the higher the 
        resolution.
    vmax : int, optional
        Determine the heatmap scale.
    fontsize : int, optional
        Determine of all font sizes.
    invertY : bool, optional
        Determine if de Y axis will be inverted (used for DLC output).
    saveName : str, optional
        Determine the save name of the plot.        
    figformat : str, optional
        Determines the type of file that will be saved. Used as base the ".eps", 
        which may be another supported by matplotlib. 

    Returns
    -------
    out : plot
        The output of the function is the figure with the heatpmat of trackin.

    See Also
    --------
    For more information and usage examples: https://github.com/pyratlib/pyrat

    Notes
    -----
    This function was developed based on DLC outputs and is able to support 
    matplotlib configurations."""
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib import cm

    values = (data.iloc[2:,1:].values).astype(np.float)
    lista1 = (data.iloc[0][1:].values +" - " + data.iloc[1][1:].values).tolist()

    x = values[:,lista1.index(bodyPart+" - x")]
    y = values[:,lista1.index(bodyPart+" - y")]

    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(wSize, hSize), dpi=80)
    plt.hist2d(x,y, bins = bins, vmax = vmax,cmap=plt.get_cmap(cmapType))

    cb = plt.colorbar()

    plt.title(figureTitle, fontsize=fontsize)
    cb.ax.tick_params(labelsize=fontsize*0.8)
    plt.xlabel("X (px)",fontsize=fontsize)
    plt.ylabel("Y (px)",fontsize=fontsize)
    plt.xticks(fontsize = fontsize*0.8)
    plt.yticks(fontsize = fontsize*0.8)
    if invertY == True:
        plt.gca().invert_yaxis()

    if saveName != None:
        plt.savefig(saveName+figformat)
    plt.show()

def ScaleConverter(dado, pixel_max,pixel_min,max_real, min_real=0):
    """
    It performs the pixel conversion to the determined real scale (meter, 
    centimeter, millimeter...).

    Parameters
    ----------
    data : pandas DataFrame
        The input tracking data.
    pixel_max : int
        Pixel maximum value. 
    pixel_min : int
        Pixel minimum value.   
    max_real : int
        Box maxixum value (eg., box wall).
    min_real : int
        Box maxixum value, usually zero.

    Returns
    -------
    out : int
        Scale factor of your box.

    See Also
    --------
    For more information and usage examples: https://github.com/pyratlib/pyrat

    Notes
    -----
    This function was developed based on DLC outputs and is able to support 
    matplotlib configurations."""    

    return min_real + ((dado-pixel_min)/(pixel_max-pixel_min)) * (max_real-min_real)

def MotionMetrics (data,bodyPart,filter=0,fps=30,max_real=60,min_real=0):
    """
    Performs motion-related metrics such as velocity, acceleration, and distance.

    Parameters
    ----------
    data : pandas DataFrame
        The input tracking data.
    bodyPart : str
        Body part you want use as reference.
    filter : float
        Threshold to remove motion artifacts. Adjust according to the tracking 
        quality and speed of what is moving.   
    fps : int
        The recording frames per second.
    max_real : int
        Box maxixum value (eg., box wall).
    min_real : int
        Box maxixum value, usually zero.
        
    Returns
    -------
    out : pandas DataFrame
        All metrics in the df.

    See Also
    --------
    For more information and usage examples: https://github.com/pyratlib/pyrat

    Notes
    -----
    This function was developed based on DLC outputs and is able to support 
    matplotlib configurations."""
    import numpy as np
    import pandas as pd

    values = (data.iloc[2:,1:].values).astype(np.float)
    lista1 = (data.iloc[0][1:].values +" - " + data.iloc[1][1:].values).tolist()

    dataX = values[:,lista1.index(bodyPart+" - x")]
    dataY = values[:,lista1.index(bodyPart+" - y")]

    dataX = ScaleConverter(dataX,dataX.max(),dataX.min(), max_real,0)
    dataY = ScaleConverter(dataY,dataY.max(),dataY.min(), min_real,0)

    time = np.arange(0,((1/fps)*len(dataX)), (1/fps))
    df = pd.DataFrame(time, columns = ["Time"])
    dist = np.hypot(np.diff(dataX, prepend=dataX[0]), np.diff(dataY, prepend=dataY[0]))
    dist[dist>=filter] = 0
    dist[0] = "nan"
    df["Distance"] = dist
    df['Speed'] = df['Distance']/(1/fps)
    df['Acceleration'] =  df['Speed'].diff().abs()/(1/fps)

    return df

def FieldDetermination(Fields):
    """
    Creates a data frame with the desired dimensions to extract information about
    an area. Therefore, you must determine the area in which you want to extract 
    information. Works perfectly with objects. We suggest using ImageJ, DLC GUI 
    or other image software that is capable of informing the coordinates of a frame.

    Parameters
    ----------
    Fields : int
        Determines the number of fields or objects you want to create.
  
    Returns
    -------
    out : pandas DataFrame
        The coordinates of the created fields.

    See Also
    --------
    For more information and usage examples: https://github.com/pyratlib/pyrat

    Notes
    -----
    This function was developed based on DLC outputs and is able to support 
    matplotlib configurations."""

    import numpy as np
    import pandas as pd

    null = .001
    fields = pd.DataFrame(columns=['fields','center_x','center_y', 'radius', 'a_x', 'a_y' , 'height', 'width'])
    for i in range(Fields):
        print('Enter the object type '+ str(i+1) + " (0 - circular, 1 - rectangular):")
        objectType = int(input())
        if objectType == 0:
            print('Enter the X value of the center of the field ' + str(i+1) + ':')
            centerX = int(input())
            print('Enter the Y value of the center of the field ' + str(i+1) + ':')
            centerY = int(input())
            print('Enter the radius value of the field ' + str(i+1) + ':')
            radius = int(input())
            df2 = pd.DataFrame([[objectType, centerX, centerY,radius,null,null,null,null]], columns=['fields','center_x','center_y', 'radius', 'a_x', 'a_y' , 'height', 'width'])
        else:
            print('Enter the X value of the field\'s lower left vertex ' + str(i+1) + ':')
            aX = int(input())
            print('Enter the Y value of the field\'s lower left vertex ' + str(i+1) + ':')
            aY = int(input())
            print('Enter the field height value ' + str(i+1) + ':')
            height = int(input())
            print('Enter the field\'s width value ' + str(i+1) + ':')
            width = int(input())
            df2 = pd.DataFrame([[objectType, null,null, null ,aX,aY,height,width]], columns=['fields','center_x','center_y', 'radius', 'a_x', 'a_y' , 'height', 'width'])
        fields = fields.append(df2, ignore_index=True)
        
    return fields

def Interaction(data,bodyPart,fields):
    """
    Performs the metrification of the interaction of the point of the determined 
    body part and the marked area.

    Parameters
    ----------
    data : pandas DataFrame
        The input tracking data.
    bodyPart : str
        Body part you want use as reference.
    fields : pandas DataFrame
        The DataFrame with the coordinates of the created fields (output of FieldDetermination()).   

    Returns
    -------
    out : int
        Scale factor of your box.

    See Also
    --------
    For more information and usage examples: https://github.com/pyratlib/pyrat

    Notes
    -----
    This function was developed based on DLC outputs and is able to support 
    matplotlib configurations.""" 

    import numpy as np
    import pandas as pd

    values = (data.iloc[2:,1:].values).astype(np.float)
    lista1 = (data.iloc[0][1:].values +" - " + data.iloc[1][1:].values).tolist()

    dataX = values[:,lista1.index(bodyPart+" - x")]
    dataY = values[:,lista1.index(bodyPart+" - y")]

    numObjects = len(fields.index)
    interact = np.zeros(len(dataX))

    for i in range(len(interact)):
        for j in range(numObjects):
            if fields['fields'][0] == 0:
                if ((dataX[i] - fields['center_x'][j])**2 + (dataY[i] - fields['center_y'][j])**2 <= fields['radius'][j]**2):
                    interact[i] = j +1
            else:
                if fields['a_x'][j] <= dataX[i] <= (fields['a_x'][j] + fields['width'][j]) and fields['a_y'][j] <= dataY[i] <= (fields['a_y'][j] + fields['height'][j]): 
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

    start = end
    end = (len(interact)-1)/fps
    obj = interact[-1]
    df = pd.DataFrame([[start,end,obj]],columns=['start','end','obj'])
    interactsDf = interactsDf.append(df, ignore_index=True)

    return interactsDf

def Reports(df_list,list_name,bodypart,fields=None,filter=0.3,fps = 30):
    """
    Produces a report of all data passed along the way with movement and interaction metrics in a 
    given box space.

    Parameters
    ----------
    df_list : list
        List with all DataFrames.
    list_name : list
        List with names of each data.
    bodypart : str
        Body part you want use as reference.
    fields : pandas DataFrame
        The DataFrame with the coordinates of the created fields (output of FieldDetermination()).   
    filter : float
        Threshold to remove motion artifacts. Adjust according to the tracking 
        quality and speed of what is moving.
    fps : int
        The recording frames per second.

    Returns
    -------
    out : pandas DataFrame
        DataFrame with report of each data in one line.

    See Also
    --------
    For more information and usage examples: https://github.com/pyratlib/pyrat

    Notes
    -----
    This function was developed based on DLC outputs and is able to support 
    matplotlib configurations.""" 

    import numpy as np
    import pandas as pd
    import pyratlib as rat

    relatorio = pd.DataFrame(columns=['file','time','dist', 'speed'])
  
    if type(fields) != type(None):  
        for i in range(len(fields)):
            relatorio["obj{0}".format(i+1)] = []
            relatorio["time_obj{0}".format(i+1)] = []

    for i,v in enumerate(df_list):
        lista = [list_name[i]]

        DF = rat.MotionMetrics(df_list[i], bodypart, filter=filter, fps=fps)

        time = DF.Time.iloc[-1]
        dist = DF.Distance.sum()
        vMedia = DF.Speed.mean()
        
        lista.append(time)
        lista.append(dist)
        lista.append(vMedia)
        if type(fields) != type(None): 
            interacts = rat.Interaction(df_list[i], bodypart, fields)
            for i in range(len(fields)):
                lista.append(interacts["obj"][interacts["obj"] == i+1].count())
                lista.append((interacts["end"][interacts["obj"] == i+1]-interacts["start"][interacts["obj"] == i+1]).sum())
        relatorio_temp = pd.DataFrame([lista], columns=relatorio.columns)
        relatorio = relatorio.append(relatorio_temp, ignore_index=True)

    return relatorio