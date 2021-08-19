import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

def Trajectory(data,bodyPartTraj,bodyPartBox, **kwargs):
  # ,start=None,end=None,fps=30,cmapType='viridis',
                # figureTitle=None,hSize=6,wSize=8,fontsize=15,invertY=True,saveName=None,
                # figformat=".eps"):
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

    c = np.linspace(0, x.size/fps, x.size)
    esquerda = values[:,lista1.index(bodyPartBox+" - x")].min()
    direita = values[:,lista1.index(bodyPartBox+" - x")].max()
    baixo = values[:,lista1.index(bodyPartBox+" - y")].min()
    cima = values[:,lista1.index(bodyPartBox+" - y")].max()

    if type(ax) == type(None): 
        plt.figure(figsize=(wSize, hSize), dpi=res)
        plt.title(figureTitle, fontsize=fontsize)
        plt.scatter(x, y, c=c, cmap=cmap, s=3)
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

        if type(saveName) == type(None):
            plt.savefig(saveName+figformat)

    else:
        ax.set_aspect('equal')
        ax.scatter(x, y, c=c, cmap=cmap, s=3)
        ax.plot([esquerda,esquerda] , [baixo,cima],"k")
        ax.plot([esquerda,direita]  , [cima,cima],"k")
        ax.plot([direita,direita]   , [cima,baixo],"k")
        ax.plot([direita,esquerda]  , [baixo,baixo],"k")
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

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
    fig : fig,optional
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
    vmax = kwargs.get('vmax')
    if type(vmax) == type(None):
      vmax = 1000
    res = kwargs.get('res')
    if type(res) == type(None):
      res = 80  


    values = (data.iloc[2:,1:].values).astype(np.float)
    lista1 = (data.iloc[0][1:].values +" - " + data.iloc[1][1:].values).tolist()

    if type(start) == type(None):
        x = values[:,lista1.index(bodyPart+" - x")]
        y = values[:,lista1.index(bodyPart+" - y")]
    else:
        init = int(start*fps)
        finish = int(end*fps)
        x = values[:,lista1.index(bodyPart+" - x")][init:finish]
        y = values[:,lista1.index(bodyPart+" - y")][init:finish]

    if type(ax) == type(None):
        plt.figure(figsize=(wSize, hSize), dpi=res)
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
        plt.show()

        if type(saveName) != type(None):
            plt.savefig(saveName+figformat)
    else:
        ax.hist2d(x,y, bins = bins, vmax = vmax,cmap=plt.get_cmap(cmapType))
        ax.tick_params(axis='both', which='major', labelsize=fontsize)

        if invertY == True:
            ax.invert_yaxis()

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right',size='5%', pad=0.05)

        im = ax.imshow([x,y], cmap=plt.get_cmap(cmapType))
        cb = fig.colorbar(im,cax=cax, orientation='vertical')
        cb.ax.tick_params(labelsize=fontsize)

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
    df = pd.DataFrame(time/60, columns = ["Time"])
    dist = np.hypot(np.diff(dataX, prepend=dataX[0]), np.diff(dataY, prepend=dataY[0]))
    dist[dist>=filter] = 0
    dist[0] = "nan"
    df["Distance"] = dist
    df['Speed'] = df['Distance']/(1/fps)
    df['Acceleration'] =  df['Speed'].diff().abs()/(1/fps)

    return df

def FieldDetermination(Fields,plot=False,**kwargs):
    """
    Creates a data frame with the desired dimensions to extract information about
    an area. Therefore, you must determine the area in which you want to extract 
    information. Works perfectly with objects. We suggest using ImageJ, DLC GUI 
    or other image software that is capable of informing the coordinates of a frame.
    If you have difficulty in positioning the areas, this parameter will plot the 
    graph where the areas were positioned. It needs to receive the DataFrame of the 
    data and the part of the body that will be used to determine the limits of the 
    environment (usually the tail).
    
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
    invertY : bool, optional
        Determine if de Y axis will be inverted (used for DLC output).
    Returns
    -------
    out : pandas DataFrame
        The coordinates of the created fields.
          plot
        Plot of objects created for ease of use.

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
 
    data = kwargs.get('data')
    bodyPartBox = kwargs.get('bodyPartBox')
    invertY = kwargs.get('invertY')
    if type(invertY) == type(None):
      invertY = True

    values = (data.iloc[2:,1:].values).astype(np.float)
    lista1 = (data.iloc[0][1:].values +" - " + data.iloc[1][1:].values).tolist()
    null = .001
    fields = pd.DataFrame(columns=['fields','center_x','center_y', 'radius', 'a_x', 'a_y' , 'height', 'width'])
    circle = []
    rect = []
    if plot:
        ax = plt.gca()
        esquerda = values[:,lista1.index(bodyPartBox+" - x")].min()
        direita = values[:,lista1.index(bodyPartBox+" - x")].max()
        baixo = values[:,lista1.index(bodyPartBox+" - y")].min()
        cima = values[:,lista1.index(bodyPartBox+" - y")].max()


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
            circle.append(plt.Circle((centerX, centerY), radius, color='r',fill = False))
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
            rect.append(patches.Rectangle((aX, aY), height, width, linewidth=1, edgecolor='r', facecolor='none'))
            df2 = pd.DataFrame([[objectType, null,null, null ,aX,aY,height,width]], columns=['fields','center_x','center_y', 'radius', 'a_x', 'a_y' , 'height', 'width'])
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

    return interactsDf

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

def HeadOrientation(data, step, head = "cervical", tail = "tailBase", **kwargs):
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

    rad = np.arctan2((cervicalY - tailY),(cervicalX - tailX))

    esquerda = boxX.min()
    direita = boxX.max()
    baixo = boxY.min()
    cima = boxY.max()

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
            rat.DrawLine(tailX[i], tailY[i], (rad[i]), ax = ax,arrow_color = arrow_color, arrow_size = arrow_size)

        plt.plot([esquerda,esquerda] , [baixo,cima],"k")
        plt.plot([esquerda,direita]  , [cima,cima],"k")
        plt.plot([direita,direita]   , [cima,baixo],"k")
        plt.plot([direita,esquerda]  , [baixo,baixo],"k")

        if type(saveName) != type(None):
            plt.savefig(saveName+figformat)

    else:
        ax.set_aspect('equal')
        for i in range(0,len(tailY),step):
            rat.DrawLine(tailX[i], tailY[i], (rad[i]), ax =ax,arrow_color = arrow_color,arrow_size = arrow_size)
        ax.plot([esquerda,esquerda] , [baixo,cima],"k")
        ax.plot([esquerda,direita]  , [cima,cima],"k")
        ax.plot([direita,direita]   , [cima,baixo],"k")
        ax.plot([direita,esquerda]  , [baixo,baixo],"k")
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        if invertY == True:
            ax.invert_yaxis()        
