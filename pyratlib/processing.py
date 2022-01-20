import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

def Trajectory(data,bodyPartTraj,bodyPartBox = None, **kwargs):
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
    start : int, optional
        Moment of the video you want tracking to start, in seconds. If the variable 
        is empty (None), the entire video will be processed.
    end : int, optional
        Moment of the video you want tracking to end, in seconds. If the variable is 
        empty (None), the entire video will be processed.
    fps : int
        The recording frames per second.
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
    limit_boundaries : bool, optional.
        Limits the points to the box boundary.
    xLimMin : int, optional
      Determines the minimum size on the axis X.
    xLimMax : int, optional
        Determines the maximum size on the axis X.
    yLimMin : int, optional
        Determines the minimum size on the axis Y.
    yLimMax : int, optional
        Determines the maximum size on the axis Y.
    saveName : str, optional
        Determine the save name of the plot.        
    figformat : str, optional
        Determines the type of file that will be saved. Used as base the ".eps", 
        which may be another supported by matplotlib. 
    res : int, optional
        Determine the resolutions (dpi), default = 80.
    ax : fig, optional
        Creates an 'axs' to be added to a figure created outside the role by the user.
    fig : fig, optional
        Creates an 'fig()' to be added to a figure created outside the role by the user.
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
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    saveName= kwargs.get('saveName')
    start= kwargs.get('start')
    end= kwargs.get('end')
    figureTitle = kwargs.get('figureTitle')
    fps = kwargs.get('fps')
    ax = kwargs.get('ax')
    limit_boundaries = kwargs.get('limit_boundaries')
    xLimMin = kwargs.get('xLimMin')
    xLimMax = kwargs.get('xLimMax')
    yLimMin = kwargs.get('yLimMin')
    yLimMax = kwargs.get('yLimMax')

    if type(limit_boundaries) == type(None):
      limit_boundaries = False
    fig = kwargs.get('fig')
    if type(fps) == type(None):
      fps = 30
    cmapType = kwargs.get('cmapType')
    if type(cmapType) == type(None):
      cmapType = 'viridis'
    hSize = kwargs.get('hSize')
    if type(hSize) == type(None):
      hSize = 6
    wSize = kwargs.get('wSize')
    if type(wSize) == type(None):
      wSize = 8
    bins = kwargs.get('bins')
    if type(bins) == type(None):
      bins = 30
    fontsize = kwargs.get('fontsize')
    if type(fontsize) == type(None):
      fontsize = 15
    invertY = kwargs.get('invertY')
    if type(invertY) == type(None):
      invertY = True
    figformat = kwargs.get('figformat')
    if type(figformat) == type(None):
      figformat = '.eps'
    res = kwargs.get('res')
    if type(res) == type(None):
      res = 80  

    values = (data.iloc[2:,1:].values).astype(np.float)
    lista1 = (data.iloc[0][1:].values +" - " + data.iloc[1][1:].values).tolist()

    if type(start) == type(None):
        x = values[:,lista1.index(bodyPartTraj+" - x")]
        y = values[:,lista1.index(bodyPartTraj+" - y")]
    else:
        init = int(start*fps)
        finish = int(end*fps)
        x = values[:,lista1.index(bodyPartTraj+" - x")][init:finish]
        y = values[:,lista1.index(bodyPartTraj+" - y")][init:finish]


    cmap = plt.get_cmap(cmapType)


    if type(bodyPartBox) == type(None):
      c = np.linspace(0, x.size/fps, x.size)
      esquerda = xLimMin
      direita = xLimMax
      baixo = yLimMin
      cima = yLimMax
    else:
      c = np.linspace(0, x.size/fps, x.size)
      esquerda = values[:,lista1.index(bodyPartBox+" - x")].min()
      direita = values[:,lista1.index(bodyPartBox+" - x")].max()
      baixo = values[:,lista1.index(bodyPartBox+" - y")].min()
      cima = values[:,lista1.index(bodyPartBox+" - y")].max()

    if limit_boundaries:
        testeX = []
        for i in range(len(x)):
            if x[i] >= direita:
                testeX.append(direita)
            elif x[i] <= esquerda:
                testeX.append(esquerda)
            else:
                testeX.append(x[i])
        
        testeY = []
        for i in range(len(x)):
            if y[i] >= cima:
                testeY.append(cima)
            elif y[i] <= baixo:
                testeY.append(baixo)
            else:
                testeY.append(y[i])
    else:
        testeX = x
        testeY = y

    if type(ax) == type(None): 
        plt.figure(figsize=(wSize, hSize), dpi=res)
        plt.title(figureTitle, fontsize=fontsize)
        plt.scatter(testeX, testeY, c=c, cmap=cmap, s=3)
        plt.plot([esquerda,esquerda] , [baixo,cima],"k")
        plt.plot([esquerda,direita]  , [cima,cima],"k")
        plt.plot([direita,direita]   , [cima,baixo],"k")
        plt.plot([direita,esquerda]  , [baixo,baixo],"k")
        cb = plt.colorbar()

        if invertY == True:
            plt.gca().invert_yaxis()
        cb.set_label('Time (s)',fontsize=fontsize)
        cb.ax.tick_params(labelsize=fontsize*0.8)
        plt.xlabel("X (px)",fontsize=fontsize)
        plt.ylabel("Y (px)",fontsize=fontsize)
        plt.xticks(fontsize = fontsize*0.8)
        plt.yticks(fontsize = fontsize*0.8)
        plt.show()

        if type(saveName) != type(None):
            plt.savefig(saveName+figformat)

    else:
        ax.set_aspect('equal')
        plot = ax.scatter(testeX, testeY, c=c, cmap=cmap, s=3)
        ax.plot([esquerda,esquerda] , [baixo,cima],"k")
        ax.plot([esquerda,direita]  , [cima,cima],"k")
        ax.plot([direita,direita]   , [cima,baixo],"k")
        ax.plot([direita,esquerda]  , [baixo,baixo],"k")
        ax.tick_params(axis='both', which='major', labelsize=fontsize*0.8)
        ax.set_title(figureTitle, fontsize=fontsize)
        ax.set_xlabel("X (px)", fontsize = fontsize)
        ax.set_ylabel("Y (px)", fontsize = fontsize)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right',size='5%', pad=0.05)
        cb = fig.colorbar(plot,cax=cax)
        cb.ax.tick_params(labelsize=fontsize*0.8)
        cb.set_label(label='Time (s)', fontsize=fontsize)

        if invertY == True:
            ax.invert_yaxis()

def Heatmap(data, bodyPart, **kwargs):
    """
    Plots the trajectory heatmap of the determined body part.

    Parameters
    ----------
    data : pandas DataFrame
        The input tracking data.
    bodyPart : str
        Body part you want to plot the heatmap.
    bodyPartBox : str
        The body part you want to use to estimate the limits of the environment, 
        usually the base of the tail is the most suitable for this determination.
    start : int, optional
        Moment of the video you want tracking to start, in seconds. If the variable 
        is empty (None), the entire video will be processed.
    end : int, optional
        Moment of the video you want tracking to end, in seconds. If the variable is 
        empty (None), the entire video will be processed.
    fps : int
        The recording frames per second.
    cmapType : str, optional
        matplotlib colormap.
    limit_boundaries : bool, optional.
        Limits the points to the box boundary.
    figureTitle : str, optional
        Figure title.
    hSize : int, optional
        Determine the figure height size (x).
    wSize : int, optional
        Determine the figure width size (y).
    ax : fig axs, optional
        Allows the creation of an out-of-function figure to use this plot.
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
    res : int, optional
        Determine the resolutions (dpi), default = 80.
    ax : fig, optional
        Creates an 'axs' to be added to a figure created outside the role by the user.
    fig : fig, optional
        Creates an 'fig()' to be added to a figure created outside the role by the user.

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
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    saveName= kwargs.get('saveName')
    start= kwargs.get('start')
    end= kwargs.get('end')
    figureTitle = kwargs.get('figureTitle')
    fps = kwargs.get('fps')
    ax = kwargs.get('ax')
    fig = kwargs.get('fig')
    bodyPartBox = kwargs.get('bodyPartBox')
    limit_boundaries = kwargs.get('limit_boundaries')
    xLimMin = kwargs.get('xLimMin')
    xLimMax = kwargs.get('xLimMax')
    yLimMin = kwargs.get('yLimMin')
    yLimMax = kwargs.get('yLimMax')

    if type(limit_boundaries) == type(None):
      limit_boundaries = False
    if type(fps) == type(None):
      fps = 30
    if type(bodyPartBox) == type(None):
      bodyPartBox = bodyPart
    cmapType = kwargs.get('cmapType')
    if type(cmapType) == type(None):
      cmapType = 'viridis'
    hSize = kwargs.get('hSize')
    if type(hSize) == type(None):
      hSize = 6
    wSize = kwargs.get('wSize')
    if type(wSize) == type(None):
      wSize = 8
    bins = kwargs.get('bins')
    if type(bins) == type(None):
      bins = 30
    fontsize = kwargs.get('fontsize')
    if type(fontsize) == type(None):
      fontsize = 15
    invertY = kwargs.get('invertY')
    if type(invertY) == type(None):
      invertY = True
    figformat = kwargs.get('figformat')
    if type(figformat) == type(None):
      figformat = '.eps'
    vmax = kwargs.get('vmax')
    if type(vmax) == type(None):
      vmax = 1000
    res = kwargs.get('res')
    if type(res) == type(None):
      res = 80  

    values = (data.iloc[2:,1:].values).astype(np.float)
    lista1 = (data.iloc[0][1:].values +" - " + data.iloc[1][1:].values).tolist()

    esquerda = values[:,lista1.index(bodyPartBox+" - x")].min()
    direita = values[:,lista1.index(bodyPartBox+" - x")].max()
    baixo = values[:,lista1.index(bodyPartBox+" - y")].min()
    cima = values[:,lista1.index(bodyPartBox+" - y")].max()

    if type(start) == type(None):
        x = values[:,lista1.index(bodyPart+" - x")]
        y = values[:,lista1.index(bodyPart+" - y")]
    else:
        init = int(start*fps)
        finish = int(end*fps)
        x = values[:,lista1.index(bodyPart+" - x")][init:finish]
        y = values[:,lista1.index(bodyPart+" - y")][init:finish]

    if limit_boundaries:
        testeX = []
        for i in range(len(x)):
            if x[i] >= direita:
                testeX.append(direita)
            elif x[i] <= esquerda:
                testeX.append(esquerda)
            else:
                testeX.append(x[i])
        
        testeY = []
        for i in range(len(x)):
            if y[i] >= cima:
                testeY.append(cima)
            elif y[i] <= baixo:
                testeY.append(baixo)
            else:
                testeY.append(y[i])
    else:
        testeX = x
        testeY = y
    
    if type(ax) == type(None):
        plt.figure(figsize=(wSize, hSize), dpi=res)

        if type(xLimMin) != type(None):
            plt.hist2d(testeX,testeY, bins = bins, vmax = vmax,cmap=plt.get_cmap(cmapType), range=[[xLimMin,xLimMax],[yLimMin,yLimMax]])
        else:
            plt.hist2d(testeX,testeY, bins = bins, vmax = vmax,cmap=plt.get_cmap(cmapType))

        cb = plt.colorbar()

        plt.title(figureTitle, fontsize=fontsize)
        cb.ax.tick_params(labelsize=fontsize*0.8)
        plt.xlabel("X (px)",fontsize=fontsize)
        plt.ylabel("Y (px)",fontsize=fontsize)
        plt.xticks(fontsize = fontsize*0.8)
        plt.yticks(fontsize = fontsize*0.8)
        if invertY == True:
            plt.gca().invert_yaxis()
        plt.show()

        if type(saveName) != type(None):
            plt.savefig(saveName+figformat)
    else:
        if type(xLimMin) != type(None):
            ax.hist2d(testeX,testeY, bins = bins, vmax = vmax,cmap=plt.get_cmap(cmapType), range=[[xLimMin,xLimMax],[yLimMin,yLimMax]])
        else:
            ax.hist2d(testeX,testeY, bins = bins, vmax = vmax,cmap=plt.get_cmap(cmapType))
        ax.tick_params(axis='both', which='major', labelsize=fontsize*0.8)
        ax.set_title(figureTitle, fontsize=fontsize)
        ax.set_xlabel("X (px)", fontsize = fontsize)
        ax.set_ylabel("Y (px)", fontsize = fontsize)
        if invertY == True:
            ax.invert_yaxis()

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right',size='5%', pad=0.05)

        im = ax.imshow([testeX,testeY], cmap=plt.get_cmap(cmapType))
        cb = fig.colorbar(im,cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fontsize*0.8)

def pixel2centimeters(data, pixel_max,pixel_min,max_real, min_real=0):
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

    return min_real + ((data-pixel_min)/(pixel_max-pixel_min)) * (max_real-min_real)

def MotionMetrics (data,bodyPart,filter=1,fps=30,max_real=60,min_real=0):
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

    dataX = pixel2centimeters(dataX,dataX.max(),dataX.min(), max_real,0)
    dataY = pixel2centimeters(dataY,dataY.max(),dataY.min(), min_real,0)

    time = np.arange(0,((1/fps)*len(dataX)), (1/fps))
    df = pd.DataFrame(time/60, columns = ["Time"])
    dist = np.hypot(np.diff(dataX, prepend=dataX[0]), np.diff(dataY, prepend=dataY[0]))
    dist[dist>=filter] = 0
    dist[0] = "nan"
    df["Distance"] = dist
    df['Speed'] = df['Distance']/(1/fps)
    df['Acceleration'] =  df['Speed'].diff().abs()/(1/fps)

    return df

def FieldDetermination(Fields=1,plot=False,**kwargs):
    """
    Creates a data frame with the desired dimensions to extract information about
    an area. Therefore, you must determine the area in which you want to extract 
    information. Works perfectly with objects. We suggest using ImageJ, DLC GUI 
    or other image software that is capable of informing the coordinates of a frame.
    If you have difficulty in positioning the areas, this parameter will plot the 
    graph where the areas were positioned. It needs to receive the DataFrame of the 
    data and the part of the body that will be used to determine the limits of the 
    environment (usually the tail).
    ***ATTENTION***
    If plot = True, the user must pass the variable 'data' and 'bodypart' as input 
    to the function.   
    
    Parameters
    ----------
    Fields : int
        Determines the number of fields or objects you want to create.
    plot : bool, optional
        Plot of objects created for ease of use. If you have difficulty in positioning 
        the areas, this parameter will plot the graph where the areas were positioned. 
        It needs to receive the DataFrame of the data and the part of the body that will
        be used to determine the limits of the environment (usually the tail).
    data : pandas DataFrame, optional
        The input tracking data.
    bodyPartBox : str, optional
        The body part you want to use to estimate the limits of the environment, 
        usually the base of the tail is the most suitable for this determination.
    posit : dict, optional
        A dictionary to pass objects with directions and not need to use input. It
        must contain a cache and 8 dice ('objt_type','center_x','center_y', 'radius',
        'a_x', 'a_y' , 'height', 'width'), 'obj_type' must be 0 or 1 (0 = circle and 
        1 = rectangle). An example of this dictionary is in Section examples.       
    obj_color : str, optional
        Allows you to determine the color of the objects created in the plot.
    invertY : bool, optional
        Determine if de Y axis will be inverted (used for DLC output).

    Returns
    -------
    out : pandas DataFrame
        The coordinates of the created fields.
          plot
        Plot of objects created for ease of use.

    Examples
    --------
    Dictionary :
    >>>> posições = {'circ': [0,200,200,50,0  ,0  ,0 ,0 ],
    >>>>             'rect': [1,0  ,0  ,0 ,400,200,75,75],}

    See Also
    --------
    For more information and usage examples: https://github.com/pyratlib/pyrat

    Notes
    -----
    This function was developed based on DLC outputs and is able to support 
    matplotlib configurations."""

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
      
    ax = kwargs.get('ax')
    ret = kwargs.get('ret')
    posit = kwargs.get('posit')
    data = kwargs.get('data')
    bodyPartBox = kwargs.get('bodyPartBox')
    invertY = kwargs.get('invertY')
    if type(invertY) == type(None):
      invertY = True
    obj_color = kwargs.get('obj_color')
    if type(obj_color) == type(None):
      obj_color = 'r'
    if type(ret) == type(None):
      ret = True

    null = 0
    fields = pd.DataFrame(columns=['fields','center_x','center_y', 'radius', 'a_x', 'a_y' , 'height', 'width'])
    circle = []
    rect = []
    if plot:
        values = (data.iloc[2:,1:].values).astype(np.float)
        lista1 = (data.iloc[0][1:].values +" - " + data.iloc[1][1:].values).tolist()
        ax = plt.gca()
        esquerda = values[:,lista1.index(bodyPartBox+" - x")].min()
        direita = values[:,lista1.index(bodyPartBox+" - x")].max()
        baixo = values[:,lista1.index(bodyPartBox+" - y")].min()
        cima = values[:,lista1.index(bodyPartBox+" - y")].max()

    if type(posit) == type(None):
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
                circle.append(plt.Circle((centerX, centerY), radius, color=obj_color,fill = False))
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
                rect.append(patches.Rectangle((aX, aY), height, width, linewidth=1, edgecolor=obj_color, facecolor='none'))
                df2 = pd.DataFrame([[objectType, null,null, null ,aX,aY,height,width]], columns=['fields','center_x','center_y', 'radius', 'a_x', 'a_y' , 'height', 'width'])
            fields = fields.append(df2, ignore_index=True)
    else:
        for i,v in enumerate(posit):
            df2 = pd.DataFrame([[posit[v][0], posit[v][1], posit[v][2],posit[v][3],posit[v][4],posit[v][5],posit[v][6],posit[v][7]]], 
                                 columns=['fields','center_x','center_y', 'radius', 'a_x', 'a_y','height', 'width'])
            if posit[v][0] == 1:
                rect.append(patches.Rectangle((float(posit[v][4]),float(posit[v][5])), float(posit[v][6]), float(posit[v][7]), linewidth=1, edgecolor=obj_color, facecolor='none'))
            if posit[v][0] == 0:
                circle.append(plt.Circle((float(posit[v][1]),float(posit[v][2])), float(posit[v][3]), color=obj_color,fill = False))
            fields = fields.append(df2, ignore_index=True)

    if plot:
        ax.plot([esquerda,esquerda] , [baixo,cima],"k")
        ax.plot([esquerda,direita]  , [cima,cima],"k")
        ax.plot([direita,direita]   , [cima,baixo],"k")
        ax.plot([direita,esquerda]  , [baixo,baixo],"k")
        if invertY == True:
            ax.invert_yaxis()
        for i in range(len(circle)):
            ax.add_patch(circle[i])
        for i in range(len(rect)):
            ax.add_patch(rect[i])

    if ret:
            return fields

def Interaction(data,bodyPart,fields,fps=30):
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
    fps : int, optional
        The recording frames per second.   

    Returns
    -------
    interactsDf: DataFrame
        DataFrame with all interactions. 0 = no interaction; 1 = first object; 2 = second object...
    interacts: list
        List with the interactions without processing.

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
                if fields['a_x'][j] <= dataX[i] <= (fields['a_x'][j] + fields['height'][j]) and fields['a_y'][j] <= dataY[i] <= (fields['a_y'][j] + fields['width'][j]):
                    interact[i] = j +1

        interactsDf = pd.DataFrame(columns=['start','end','obj'])

    obj = 0
    start = 0
    end = 0
    fps =fps

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

    return interactsDf, interact

def Reports(df_list,list_name,bodypart,fields=None,filter=0.3,fps=30):
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

    Examples
    --------
    DataFrame :
    >>>> columns = 'file','video time (min)', 'dist (cm)','speed (cm/s)','field','time_field'
    >>>> file = file name
    >>>> video time (min) = video duration
    >>>> dist (cm) = ditance traveled in centimeters
    >>>> speed (cm/s) = animal velocity in centimeters per second
    >>>> field = how many times did you interact with the field, organized by order (if fields != None)
    >>>> time_field = time spent in each field

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

    relatorio = pd.DataFrame(columns=['file','video time (min)','dist (cm)', 'speed (cm/s)'])
  
    if type(fields) != type(None):  
        for i in range(len(fields)):
            relatorio["field_{0}".format(i+1)] = []
            relatorio["time_field_{0}".format(i+1)] = []

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
            interacts,_ = rat.Interaction(df_list[i], bodypart, fields)
            for i in range(len(fields)):
                lista.append(interacts["obj"][interacts["obj"] == i+1].count())
                lista.append((interacts["end"][interacts["obj"] == i+1]-interacts["start"][interacts["obj"] == i+1]).sum())
        relatorio_temp = pd.DataFrame([lista], columns=relatorio.columns)
        relatorio = relatorio.append(relatorio_temp, ignore_index=True)

    return relatorio

def DrawLine(x, y, angle, **kwargs):
    """
    Makes the creation of arrows to indicate the orientation of the animal's head in a superior 
    view. Used in the HeadOrientation() function.

    Parameters
    ----------
    x : float
        X axis coordinates.
    y : float
        Y axis coordinates.
    angle : float
        Angle in radians, output of the arctan2 function.
    ax : fig, optional
        Creates an 'axs' to be added to a figure created outside the role by the user.  
    arrow_width : int, optional
        Determines the width of the arrow's body.
    head_width : int, optional
        Determines the width of the arrow head.
    arrow_color : str, optional
        Determines the arrow color.
    arrow_size : int, optional
        Determines the arrow size.

    Returns
    -------
    out : plot
        Arrow based on head coordinates.

    See Also
    --------
    For more information and usage examples: https://github.com/pyratlib/pyrat

    Notes
    -----
    This function was developed based on DLC outputs and is able to support 
    matplotlib configurations."""

    import numpy as np
    import matplotlib.pyplot as plt
    ax = kwargs.get('ax')
    arrow_color = kwargs.get('arrow_color')
    arrow_width = kwargs.get('arrow_width')
    if type(arrow_width) == type(None):
      arrow_width = 2
    head_width = kwargs.get('head_width')
    if type(head_width) == type(None):
      head_width = 7
    arrow_size = kwargs.get('arrow_size')
    if type(arrow_size) == type(None):
      arrow_size = 10

    if type(ax) == type(None):
        return plt.arrow(x, y, arrow_size*np.cos(angle), arrow_size*np.sin(angle),width = arrow_width,head_width=head_width,fc = arrow_color)
    else:
        return ax.arrow(x, y, arrow_size*np.cos(angle), arrow_size*np.sin(angle),width = arrow_width,head_width=head_width,fc = arrow_color)

def HeadOrientation(data, step, head = None, tail = None, **kwargs):
    """
    Plots the trajectory of the determined body part.

    Parameters
    ----------
    data : pandas DataFrame
        The input tracking data.
    step : int
        Step used in the data, will use a data point for each 'x' steps. The 
        smaller the step, the greater the amount of arrows and the more difficult 
        the interpretation.
    head : str
        Head coordinates to create the arrow. You can use data referring to another
        part of the body that you want to have as a reference for the line that will
        create the arrow. The angulation will be based on the arrow.
    tail : str
        Tail coordinates to create the arrow. You can use data referring to another
        part of the body that you want to have as a reference for the line that will
        create the arrow. The angulation will be based on the arrow.
    bodyPartBox : str, optional
        The body part you want to use to estimate the limits of the environment, 
        usually the base of the tail is the most suitable for this determination.
    start : int, optional
        Moment of the video you want tracking to start, in seconds. If the variable 
        is empty (None), the entire video will be processed.
    end : int, optional
        Moment of the video you want tracking to end, in seconds. If the variable is 
        empty (None), the entire video will be processed.
    fps : int
        The recording frames per second.
    limit_boundaries : bool, optional.
        Limits the points to the box boundary.
    xLimMin : int, optional
      Determines the minimum size on the axis X.
    xLimMax : int, optional
        Determines the maximum size on the axis X.
    yLimMin : int, optional
        Determines the minimum size on the axis Y.
    yLimMax : int, optional
        Determines the maximum size on the axis Y.
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
    res : int, optional
        Determine the resolutions (dpi), default = 80.
    ax : fig, optional
        Creates an 'axs' to be added to a figure created outside the role by the user.  
    arrow_width : int, optional
        Determines the width of the arrow's body.
    head_width : int, optional
        Determines the width of the arrow head.
    arrow_color : str, optional
        Determines the arrow color.
    arrow_size : int, optional
        Determines the arrow size.

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
    import pyratlib as rat

    ax = kwargs.get('ax')
    start= kwargs.get('start')
    end= kwargs.get('end')
    figureTitle = kwargs.get('figureTitle')
    saveName = kwargs.get('saveName')
    hSize = kwargs.get('hSize')
    bodyPartBox = kwargs.get('bodyPartBox')
    arrow_color = kwargs.get('arrow_color')
    limit_boundaries = kwargs.get('limit_boundaries')
    xLimMin = kwargs.get('xLimMin')
    xLimMax = kwargs.get('xLimMax')
    yLimMin = kwargs.get('yLimMin')
    yLimMax = kwargs.get('yLimMax')
    if type(limit_boundaries) == type(None):
      limit_boundaries = False
    if type(bodyPartBox) == type(None):
      bodyPartBox = tail
    fps = kwargs.get('fps')
    if type(fps) == type(None):
      fps = 30
    res = kwargs.get('res')
    if type(res) == type(None):
      res = 80  
    if type(hSize) == type(None):
      hSize = 6
    wSize = kwargs.get('wSize')
    if type(wSize) == type(None):
      wSize = 8
    fontsize = kwargs.get('fontsize')
    if type(fontsize) == type(None):
      fontsize = 15
    invertY = kwargs.get('invertY')
    if type(invertY) == type(None):
      invertY = True
    figformat = kwargs.get('figformat')
    if type(figformat) == type(None):
      figformat = '.eps'
    arrow_width = kwargs.get('arrow_width')
    if type(arrow_width) == type(None):
      arrow_width = 2
    head_width = kwargs.get('head_width')
    if type(head_width) == type(None):
      head_width = 7 
    arrow_size = kwargs.get('arrow_size')
    if type(arrow_size) == type(None):
      arrow_size = 10

    values = (data.iloc[2:,1:].values).astype(np.float)
    lista1 = (data.iloc[0][1:].values +" - " + data.iloc[1][1:].values).tolist()

    if type(start) == type(None):
        tailX = values[:,lista1.index(tail+" - x")] 
        tailY = values[:,lista1.index(tail+" - y")]

        cervicalX = values[:,lista1.index(head+" - x")]
        cervicalY = values[:,lista1.index(head+" - y")]
    else:
        init = int(start*fps)
        finish = int(end*fps)

        tailX = values[:,lista1.index(tail+" - x")][init:finish] 
        tailY = values[:,lista1.index(tail+" - y")][init:finish]

        cervicalX = values[:,lista1.index(head+" - x")][init:finish]
        cervicalY = values[:,lista1.index(head+" - y")][init:finish]

    boxX = values[:,lista1.index(bodyPartBox+" - x")]
    boxY = values[:,lista1.index(bodyPartBox+" - y")]

    if type(bodyPartBox) == type(None):
      
      esquerda = xLimMin
      direita = xLimMax
      baixo = yLimMin
      cima = yLimMax
    else:
      
      esquerda = values[:,lista1.index(bodyPartBox+" - x")].min()
      direita = values[:,lista1.index(bodyPartBox+" - x")].max()
      baixo = values[:,lista1.index(bodyPartBox+" - y")].min()
      cima = values[:,lista1.index(bodyPartBox+" - y")].max()

    if limit_boundaries:
        testeX = []
        for i in range(len(tailX)):
            if tailX[i] >= direita:
                testeX.append(direita)
            elif tailX[i] <= esquerda:
                testeX.append(esquerda)
            else:
                testeX.append(tailX[i])
        
        testeY = []
        for i in range(len(tailY)):
            if tailY[i] >= cima:
                testeY.append(cima)
            elif tailY[i] <= baixo:
                testeY.append(baixo)
            else:
                testeY.append(tailY[i])
    else:
        testeX = tailX
        testeY = tailY

    if limit_boundaries:
        tX = []
        for i in range(len(cervicalX)):
            if cervicalX[i] >= direita:
                tX.append(direita)
            elif cervicalX[i] <= esquerda:
                tX.append(esquerda)
            else:
                tX.append(cervicalX[i])
        
        tY = []
        for i in range(len(cervicalY)):
            if cervicalY[i] >= cima:
                tY.append(cima)
            elif cervicalY[i] <= baixo:
                tY.append(baixo)
            else:
                tY.append(cervicalY[i])
    else:
        tX = cervicalX
        tY = cervicalY

    rad = np.arctan2((np.asarray(tY) - np.asarray(testeY)),(np.asarray(tX) - np.asarray(testeX)))

    if type(ax) == type(None):
        plt.figure(figsize=(wSize, hSize), dpi=res)
        plt.title(figureTitle, fontsize=fontsize)
        plt.gca().set_aspect('equal')
      
        if invertY == True:
            plt.gca().invert_yaxis()
      
        plt.xlabel("X (px)",fontsize=fontsize)
        plt.ylabel("Y (px)",fontsize=fontsize)
        plt.xticks(fontsize = fontsize*0.8)
        plt.yticks(fontsize = fontsize*0.8)
      
        for i in range(0,len(tailY),step):
            rat.DrawLine(tX[i], tY[i], (rad[i]), ax = ax,arrow_color = arrow_color, arrow_size = arrow_size)
            
        plt.plot([esquerda,esquerda] , [baixo,cima],"k")
        plt.plot([esquerda,direita]  , [cima,cima],"k")
        plt.plot([direita,direita]   , [cima,baixo],"k")
        plt.plot([direita,esquerda]  , [baixo,baixo],"k")

        if type(saveName) != type(None):
            plt.savefig(saveName+figformat)

    else:
        ax.set_aspect('equal')
        for i in range(0,len(tailY),step):
            rat.DrawLine(tX[i], tY[i], (rad[i]), ax =ax,arrow_color = arrow_color,arrow_size = arrow_size)
        ax.plot([esquerda,esquerda] , [baixo,cima],"k")
        ax.plot([esquerda,direita]  , [cima,cima],"k")
        ax.plot([direita,direita]   , [cima,baixo],"k")
        ax.plot([direita,esquerda]  , [baixo,baixo],"k")
        ax.set_title(figureTitle, fontsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize*0.8)
        ax.set_xlabel("X (px)", fontsize = fontsize)
        ax.set_ylabel("Y (px)", fontsize = fontsize)
        if invertY == True:
            ax.invert_yaxis()

def SignalSubset(sig_data,freq,fields, **kwargs):
    """
    Performs the extraction of substes from electrophysiology data. For proper functioning, the 
    data must be organized in a data. To subset the data with its own list of markers, the 
    variable fields must be empty (fields = None).

    Parameters
    ----------
    sig_data : pandas DataFrame
        The input electrophysiology data organized with the channels in columns. We use the function
        LFP().
    freq : pandas DataFrame
        Frequency of electrophysiology data collection.
    fields : str
        Event time markers. Developed to use the output of the "Interaction()" function. But with 
        standardized data like the output of this function, it is possible to assemble the dataframe.
    start_time : list, optional
        Moment of the video you want subset to start, in seconds. If the variable is empty (None), 
        the entire video will be processed.
    end_time : list, optional
        Moment of the video you want subset to end, in seconds. If the variable is empty (None), 
        the entire video will be processed.
        
    Returns
    -------
    out : dict
        if fields = None:
            return a dictionary with size equal to the list in start_time or end_time lenght, with others
            dictionaries with the subset in each channel.
        else:
            Dictionary with the objects/places passed in the input, inside each one will have the channels
            passed with the subset of the data.

    See Also
    --------
    For more information and usage examples: https://github.com/pyratlib/pyrat

    Notes
    -----
    This function was based using data from the electrophysiology plexon. Input data was formatted using
    the LFP() function0. Lista first index is to enter the list; Lista second index is to choose between
    start([0]) ou end([1]); Lista third index is to choose between object/fields([0],[1] ...).
    """ 

    start_time= kwargs.get('start_time')
    end_time = kwargs.get('end_time')

    if type(fields) == type(None):
        dicts = {}
        if type(start_time) == type(None):
            keys = range(len(end_time))
        else:
            keys = range(len(start_time))
        for j in keys:
            cortes = {}
            for ç,canal in enumerate(sig_data.columns):
                if type(start_time) == type(None):
                    cortes[ç] = sig_data[canal][None:end_time[j]*freq]
                    dicts[j] = cortes 
                elif type(end_time) == type(None):
                    cortes[ç] = sig_data[canal][start_time[j]*freq:None]
                    dicts[j] = cortes 
                else: 
                    cortes[ç] = sig_data[canal][start_time[j]*freq:end_time[j]*freq]
                    dicts[j] = cortes

    else:
        lista = []
        start = []
        end = []

        for i in fields['obj'].unique():
            if i != 0:
                start.append(fields.start.loc[(fields.obj == i)].values)
                end.append(fields.end.loc[(fields.obj == i)].values)
                lista.append((start,end))

        dicts = {}
        keys = range(len(list(fields['obj'].unique())[1:]))

        for j in keys:
            cortes = {}
            for ç,canal in enumerate(sig_data.columns):
                for i in range(len(lista[0][0][j])):
                    cortes[ç] = sig_data[canal][int(lista[0][0][j][i]*freq):int(lista[0][1][j][i]*freq)]
                    dicts[j] = cortes

    return dicts

def LFP(data):
    """
    Performs LFP data extraction from a MATLAB file (.mat) and returns all channels arranged in 
    columns of a DataFrame. This data was converted from a plexon file (.plx).

    Parameters
    ----------
    data : .mat
        Plexon data in .mat format.
       
    Returns
    -------
    out : pandas DataFrame
        A DataFrame with the data arranged in columns by channels.

    See Also
    --------
    For more information and usage examples: https://github.com/pyratlib/pyrat

    Notes
    -----
    This function was based using data from the electrophysiology plexon. Input data was formatted using
    the LFP() function.""" 

    import numpy as np
    column_name = []
    if len(data['allad'][0]) == 192:
        time = np.arange(0,len(data['allad'][0][128])/data['adfreq'][0][0],1/data['adfreq'][0][0]) 
        values = np.zeros((len(data['allad'][0][128]),64))
        ç = 128
        for j in range(64):
            for r in range(len(data['allad'][0][128])):
                values[r][j] = data['allad'][0][ç][r]
            column_name.append(data['adnames'][ç])
            ç +=1
    elif len(data['allad'][0]) == 96:
        time = np.arange(0,len(data['allad'][0][64])/data['adfreq'][0][0],1/data['adfreq'][0][0]) 
        values = np.zeros((len(data['allad'][0][64]),32))
        ç = 64
        for j in range(32):
            for r in range(len(data['allad'][0][64])):
                values[r][j] = data['allad'][0][ç][r]
            column_name.append(data['adnames'][ç])
            ç +=1

    df = pd.DataFrame(values,columns=column_name,index= None) 
    df.insert(0, "Time", time, True)

    return df  

def PlotInteraction(interactions, **kwargs):
  """
  Plots a bar with interactions times of the determined body with the fields.

  Parameters
  ----------
  interactions : pandas DataFrame
      The DataFrame with the interactions of the fields (output of Interaction()). 
  barH : float, optional
      Bar height.
  start : int, optional
      Moment of the video you want tracking to start, in seconds. If the variable 
      is empty (None), the entire video will be processed.
  end : int, optional
      Moment of the video you want tracking to end, in seconds. If the variable is 
      empty (None), the entire video will be processed.
  fps : int
      The recording frames per second.
  figureTitle : str, optional
      Figure title.
  hSize : int, optional
      Determine the figure height size (x).
  wSize : int, optional
      Determine the figure width size (y).
  fontsize : int, optional
      Determine of all font sizes.
  saveName : str, optional
      Determine the save name of the plot.        
  figformat : str, optional
      Determines the type of file that will be saved. Used as base the ".eps", 
      which may be another supported by matplotlib. 
  res : int, optional
      Determine the resolutions (dpi), default = 80.
  ax : fig, optional
      Creates an 'axs' to be added to a figure created outside the role by the user.

  Returns
  -------
  out : plot
      The output of the function is the figure with the interactions times with fields.
      
  See Also
  --------
  For more information and usage examples: https://github.com/pyratlib/pyrat
  Notes
  -----
  This function was developed based on DLC outputs and is able to support 
  matplotlib configurations."""

    
  import pyratlib as rat
  import matplotlib.pyplot as plt

  saveName= kwargs.get('saveName')
  start= kwargs.get('start')
  end= kwargs.get('end')
  figureTitle = kwargs.get('figureTitle')
  fps = kwargs.get('fps')
  ax = kwargs.get('ax')
  aspect = kwargs.get('aspect')
  if type(aspect) == type(None):
    aspect = 'equal'
  if type(fps) == type(None):
    fps = 30
  hSize = kwargs.get('hSize')
  if type(hSize) == type(None):
    hSize = 2
  wSize = kwargs.get('wSize')
  if type(wSize) == type(None):
    wSize = 8
  fontsize = kwargs.get('fontsize')
  if type(fontsize) == type(None):
    fontsize = 15
  figformat = kwargs.get('figformat')
  if type(figformat) == type(None):
    figformat = '.eps'
    res = kwargs.get('res')
  if type(res) == type(None):
    res = 80 
  barH = kwargs.get('barH')
  if type(barH) == type(None):
    barH = .5

  if type(start) == type(None):
      init = 0
      finish = interactions.end.iloc[-1]
  else:
      init = int(start)
      finish = int(end) 

  times = []
  starts = []
  for i in range (int(interactions.obj.max())+1):
      times.append((interactions.end.loc[(interactions.obj == i) & (interactions.start >= init) & (interactions.start <= finish)])-(interactions.start.loc[(interactions.obj == i) & (interactions.start >= init) & (interactions.start <= finish)]).values)
      starts.append((interactions.start.loc[(interactions.obj == i) & (interactions.start >= init) & (interactions.start <= finish)]).values)

  barHeight = barH

  if type(ax) == type(None):
    plt.figure(figsize=(wSize,hSize))
    for i in range (1,int(interactions.obj.max())+1):
      plt.barh(0,times[i], left=starts[i], height = barHeight, label = "Field "+str(i))

    plt.title(figureTitle,fontsize=fontsize)
    plt.legend(ncol=int(interactions.obj.max()))
    plt.xlim(init, finish)
    plt.yticks([])
    plt.xticks(fontsize = fontsize*0.8)
    plt.xlabel("Time (s)",fontsize=fontsize)
    plt.ylim([-barHeight,barHeight])
    plt.show()
    if type(saveName) != type(None):
        plt.savefig(saveName+figformat)
  else:
    for i in range (1,int(interactions.obj.max())+1):
      ax.barh(0,times[i], left=starts[i], height = barHeight, label = "Field "+str(i))
    ax.set_title(figureTitle, fontsize=fontsize)
    if aspect == type(None):
      ax.set_aspect(aspect)
    ax.set_xlim([init, finish])
    ax.set_yticklabels([])
    ax.get_yaxis().set_visible(False)
    ax.tick_params(axis='x', labelsize=fontsize*.8)
    ax.tick_params(axis='y', labelsize=fontsize*.8)
    ax.legend(ncol=int(interactions.obj.max()),fontsize=fontsize*.8)
    ax.set_xlabel('Time (s)',fontsize=fontsize)

    ax.set_ylim([-barHeight,barHeight])

def Blackrock(data_path, freq): 
    """
    Transform a Blackrock file (.ns2) to a pandas DataFrame with all LFP channels.

    Parameters
    ----------
    data_path : path
        Str with data path.
    freq : int
        Aquisition frequency.

    Returns
    -------
    out : pandas DataFrame
        The output of the function is the figure with the interactions times with fields.
        
    See Also
    --------
    For more information and usage examples: https://github.com/pyratlib/pyrat
    """
    from neo.io import BlackrockIO
    import numpy as np
    import pandas as pd 

    reader = BlackrockIO(data_path)
    seg = reader.read_segment()

    column_name = []
    time = np.arange(0,len(seg.analogsignals[0])/freq,1/freq)
    values = np.zeros((len(seg.analogsignals[0]),len(seg.analogsignals[0][0])))

    for i in range(len(seg.analogsignals[0][0])):
        for ç in range(len(seg.analogsignals[0])):
            values[ç][i] = float(seg.analogsignals[0][ç][i])

    channels = []
    for i in range(len(seg.analogsignals[0][0])):
        channels.append('Channel ' + str(i+1))

    df = pd.DataFrame(values,columns=channels,index= None) 
    df.insert(0, "Time", time, True)

    return df

def ClassifyBehavior(data,video, bodyparts_list, dimensions = 2,distance=28,**kwargs):
    """
    Returns an array with the cluster by frame, an array with the embedding data in low-dimensional 
    space and the clusterization model.
    
    Parameters
    ----------
    data : pandas DataFrame
        The input tracking data.
    video : str
        Video directory
    bodyparts_list : list
        List with name of body parts.
    dimensions : int
        Dimension of the embedded space.
    distance : int
        The linkage distance threshold above which, clusters will not be merged.
    startIndex : int, optional
        Initial index.
    endIndex : int, optional
        Last index.
    n_jobs : int, optional
        The number of parallel jobs to run for neighbors search.
    verbose : int, optional
        Verbosity level.
    perplexity : float, optional
        The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity.
    learning_rate : float, optional
        t-SNE learning rate.
    directory : str, optional
        Path where frame images will be saved.
    return_metrics : bool, optional
         Where True, returns t-SNE metrics, otherwise does not return t-SNE metrics.
    knn_n_neighbors : int, optional
        Number of neighbors to use by default for kneighbors queries in KNN metric.
    knc_n_neighbors : int, optional
        Number of neighbors to use by default for kneighbors queries in KNC metric.
    n : int, optional
        Number of N randomly chosen points in CPD metric.
    Returns
    -------
    cluster_labels : array
        Array with the cluster by frame.
    X_transformed : array
        Embedding of the training data in low-dimensional space.
    model : Obj
        AgglomerativeClustering model.
    d  : array
        High dimension data.
    knn : int, optional
        The fraction of k-nearest neighbours in the original highdimensional data that are preserved as k-nearest neighbours in the embedding.
    knc : int, optional
        The fraction of k-nearest class means in the original data that are preserved as k-nearest class means in the embedding. This is computed for class means only and averaged across all classes.
    cpd : Obj, optional
        Spearman correlation between pairwise distances in the high-dimensional space and in the embedding.
    
    See Also
    --------
    For more information and usage examples: https://github.com/pyratlib/pyrat
    
    Notes
    -----
    This function was developed based on DLC outputs and is able to support 
    matplotlib configurations."""
    from sklearn.manifold import TSNE
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    import os
    import cv2
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from scipy import stats

    startIndex = kwargs.get('startIndex')
    endIndex = kwargs.get('endIndex')
    n_jobs = kwargs.get('n_jobs')
    verbose = kwargs.get('verbose')
    perplexity = kwargs.get("perplexity")
    learning_rate = kwargs.get("learning_rate")
    directory = kwargs.get("directory")
    return_metrics = kwargs.get("return_metrics")
    knn_n_neighbors = kwargs.get("knn_n_neighbors")
    knc_n_neighbors = kwargs.get("knc_n_neighbors")
    n = kwargs.get("n")
    k = 1
    if type(startIndex) == type(None):
      startIndex = 0
    if type(endIndex) == type(None):
      endIndex = data.shape[0]-3
    if type(n_jobs) == type(None):
      n_jobs=-1
    if type(verbose) == type(None):
      verbose=0
    if type(perplexity) == type(None):
      perplexity = data[startIndex:endIndex].shape[0]//100
    if type(learning_rate) == type(None):
      learning_rate = (data[startIndex:endIndex].shape[0]//12)/4
    if type(directory) == type(None):
      directory = os.getcwd()
    if type(return_metrics) == type(None):
      return_metrics == 0
    f = 1
    directory=directory+"/images"
    for i in range(2,len(directory.split(sep = "/"))):
      if os.path.exists("/"+"/".join(directory.split(sep = "/")[1:i+1])) == False:
        f = i
        break
    
    for i in range(f,len(directory.split(sep = "/"))):
      os.mkdir("/"+"/".join(directory.split(sep = "/")[1:i+1]))
    
    values = (data.iloc[2:,1:].values).astype(np.float)
    lista1 = (data.iloc[0][1:].values +" - " + data.iloc[1][1:].values).tolist()

    bodyparts = []
    for i in range(len(bodyparts_list)):
      bodyparts.append(np.concatenate(((values[:,lista1.index(bodyparts_list[i]+" - x")]).reshape(1,-1).T,(values[:,lista1.index(bodyparts_list[i]+" - y")]).reshape(1,-1).T), axis=1))

    distances = []

    for k in range(len(bodyparts[0])):
        frame_distances = []
        for i in range(len(bodyparts)):
            distance_row = []
            for j in range( len(bodyparts) ):
                distance_row.append(np.linalg.norm(bodyparts[i][k] - bodyparts[j][k]))
            frame_distances.append(distance_row)
        distances.append(frame_distances)

    distances2 = np.asarray(distances)

    for i in range(len(bodyparts)):
      for k in range(len(bodyparts)):
          distances2[:, i, j] = distances2[:, i, j]/np.max(distances2[:, i, j])

    d = []
    for i in range(distances2.shape[0]):
        d.append(distances2[i, np.triu_indices(len(bodyparts), k = 1)[0], np.triu_indices(len(bodyparts), k = 1)[1]])
    
    d = StandardScaler().fit_transform(d)

    embedding = TSNE(n_components=dimensions, n_jobs=n_jobs, verbose=verbose, perplexity=perplexity, random_state = 42, n_iter = 5000, learning_rate=learning_rate, init = "pca", early_exaggeration = 12)
    X_transformed = embedding.fit_transform(d[startIndex:endIndex])
    model = AgglomerativeClustering(n_clusters=None,distance_threshold=distance)
    model = model.fit(d[startIndex:endIndex])
    cluster_labels = model.labels_   

    frames = data.scorer[2:].values.astype(np.int)
    for i in np.unique(cluster_labels):
      os.mkdir(directory+"/cluster"+ str(i))

    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 0
    position = (10,50)
    ind = 0
    while success:

      if (np.isin(count, frames[startIndex:endIndex])):
          a = cv2.imwrite(directory+"/cluster"+str(model.labels_[ind])+"/frame%d.jpg" % count, image)
          ind = ind +1 
      success,image = vidcap.read()
      count += 1

    for i in np.unique(cluster_labels):
      plt.bar(i, cluster_labels[cluster_labels==i].shape, color = "C0")
    plt.xticks(np.arange(model.n_clusters_))
    plt.xlabel("Clusters")
    plt.ylabel("Frames")
    plt.show()
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    plt.figure(figsize=(10,5))
    plt.title("Hierarchical Clustering Dendrogram")
    dendrogram(linkage_matrix, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

    fig, ax = plt.subplots(figsize=(10,10), dpi =80)
    i = 0

    color = plt.cm.get_cmap("rainbow", model.n_clusters_)
    for x in range(model.n_clusters_):
      sel = cluster_labels == x
      
      pontos = ax.scatter(X_transformed[sel,0], X_transformed[sel,1], label=str(x), s=1, color = color(i))
      i = i+1

    plt.legend()
    plt.title('Clusters')
    plt.show()

    if return_metrics == 1:
      if type(knn_n_neighbors) == type(None):
        knn_n_neighbors = model.n_clusters_//2
      if type(knc_n_neighbors) == type(None):
        knc_n_neighbors = model.n_clusters_//2
      if type(n) == type(None):
        n = 1000
      data_HDim = d[startIndex:endIndex]
      data_emb = X_transformed

      neigh = NearestNeighbors(n_neighbors=knn_n_neighbors)
      neigh.fit(data_HDim)
      neigh2 = NearestNeighbors(n_neighbors=knn_n_neighbors)
      neigh2.fit(data_emb)
      intersections = 0.0
      for i in range(len(data_HDim)):
        intersections += len(set(neigh.kneighbors(data_HDim, return_distance = False)[i]) & set(neigh2.kneighbors(data_emb, return_distance = False)[i]))
      knn = intersections / len(data_HDim) / knn_n_neighbors


      clusters =  len(np.unique(cluster_labels))
      clusters_HDim = np.zeros((clusters,data_HDim.shape[1]))
      clusters_tsne = np.zeros((clusters,data_emb.shape[1]))
      for i in np.unique(cluster_labels):
        clusters_HDim[i,:] = np.mean(data_HDim[np.unique(cluster_labels, return_inverse=True)[1] == np.unique(cluster_labels, return_inverse=True)[0][i], :], axis = 0)
        clusters_tsne[i,:] = np.mean(data_emb[np.unique(cluster_labels, return_inverse=True)[1] == np.unique(cluster_labels, return_inverse=True)[0][i], :], axis = 0)
      neigh = NearestNeighbors(n_neighbors=knc_n_neighbors)
      neigh.fit(clusters_HDim)
      neigh2 = NearestNeighbors(n_neighbors=knc_n_neighbors)
      neigh2.fit(clusters_tsne)
      intersections = 0.0
      for i in range(clusters):
        intersections += len(set(neigh.kneighbors(clusters_HDim, return_distance = False)[i]) & set(neigh2.kneighbors(clusters_tsne, return_distance = False)[i]))
      knc = intersections / clusters / knc_n_neighbors


      dist_alto = np.zeros(n)
      dist_tsne = np.zeros(n)
      for i in range(n):
        a = np.random.randint(0,len(data_HDim), size = 1)
        b = np.random.randint(0,len(data_HDim), size = 1)
        dist_alto[i] = np.linalg.norm(data_HDim[a] - data_HDim[b])
        dist_tsne[i] = np.linalg.norm(data_emb[a] - data_emb[b])
      cpd = stats.spearmanr(dist_alto, dist_tsne)

      return cluster_labels, data_emb, model, data_HDim, knn, knc, cpd

    else:
      return cluster_labels, X_transformed, model, d[startIndex:endIndex]