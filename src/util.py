import logging, sys
import pandas as pd
import numpy as np

## DatetimeIndex Manipulation Routines ##
def pick_extrema_indices(series,T=pd.Timedelta(6,unit='hour'),granularity=0.2):
    """Get the extrema within cycles of a specified periodic signal

    :param series: series with data values and a DatetimeIndex index
    :type series: pandas.Series
    :param T: window in which to assess min/max, dominant period of signal,
        defaults to pd.Timedelta(6,unit='hour')
    :type T: pands.Timedelta, optional
    :param granularity: step size for scanning time series, defaults to 0.2
    :type granularity: float, optional
    :return:
     - **output** (*dict*) -- dictionary with the following keys
        - **I_max** -- timestamp indices of max values
        - **V_max** -- maximum values
        - **I_min** -- timestamp indices of min values
        - **V_min** -- minimum values
    """ 
    if isinstance(series,pd.DataFrame):
        series = series[series.columns[0]]

    t0 = series.index.min()
    tf = series.index.max()
    ti = t0
    min_ind = []; min_vals = []
    max_ind = []; max_vals = []
    while ti + T <= tf:
        iS = series[(series.index >= ti) & (series.index <=ti + T)]
        imax = iS.max(); Imax = iS.idxmax()
        imin = iS.min(); Imin = iS.idxmin()
        # If maximum value is greater than edge values
        if imax > iS.values[0] and imax > iS.values[-1]:
            # And maximum index is not already in the output list
            if Imax not in max_ind:
                max_ind.append(Imax)
                max_vals.append(imax)
        if imin < iS.values[0] and imin < iS.values[-1]:
            if Imin not in min_ind:
                min_ind.append(Imin)
                min_vals.append(imin)

        ti += T*granularity

    return {'I_max':pd.DatetimeIndex(max_ind),'V_max':np.array(max_vals),'I_min':pd.DatetimeIndex(min_ind),'V_min':np.array(min_vals)}


def fit_dateline(datetimes,values):
    """Fit a linear trend line to values indexed by a 
    pands.DatetimeIndex

    :param datetimes: datetime index
    :type datetimes: pandas.DatetimeIndex
    :param values: time series values
    :type values: array-like
    :return: 
     - **slope** (*scalar*) -- slope of the fit line
    """    
    # Convert datetime array into total seconds relative to minimum datetime
    tvsec = (datetimes - datetimes.min()).total_seconds()
    # Get slope
    mod = np.polyfit(tvsec,values,1)
    return mod[0]


def reduce_series_by_slope(series,slope,reference_time,reference_value):
    """Given a slope and a reference intercept, remove the modeled line
    from arbitrarily ordered values associated to Timestamps

    :param series: series of values with a pandas.DatetimeIndex index
    :type series: pandas.Series
    :param slope: slope to remove, in units of [data unit] seconds**-1
    :type slope: float-like
    :param reference_time: reference time to use for the model line
        intercept
    :type reference_time: pandas.Timestamp
    :param reference_value: reference value to use for the model ine
        intercept
    :type reference_value: scalar
    :return: 
     - **s_out** (*pandas.Series*) -- **series** with the prescribed
        trend line removed
    """    
    if isinstance(series,pd.DataFrame):
        series = series[series.columns[0]]
    series = series.sort_index()
    dt_ind = (series.index - reference_time).total_seconds()
    poly = [slope,reference_value]
    mfun = np.poly1d(poly)
    y_hat = mfun(dt_ind)
    y_red = series.values - y_hat
    s_out = pd.Series(y_red,index=series.index,name='%s red'%(series.name))
    return s_out

## Plotting Subroutines ##
def plot_cycles(axis,Xdata,Ydata,Tindex,t0,cmaps,ncycles=5,T=pd.Timedelta(24,unit='hour'),zorder=10):
    """Plotting routine for hysteresis cross-plots featured in Figures 7 and 8
    in Stevens and others (accepted)

    :param axis: plotting axis
    :type axis: matplotlib.pyplot.axes.Axes
    :param Xdata: x-axis data points
    :type Xdata: array-like
    :param Ydata: y-axis data points
    :type Ydata: array-like
    :param Tindex: time stamp index
    :type Tindex: pandas.DatetimeIndex
    :param t0: reference time
    :type t0: pandas.Timestamp
    :param cmaps: list of color map names to use for each cycle
    :type cmaps: list
    :param ncycles: number of cycles in this record, defaults to 5
    :type ncycles: int, optional
    :param T: period of cycles, defaults to pd.Timedelta(24,unit='hour')
    :type T: pandas.Timedelta, optional
    :param zorder: zorder level of plotted data, defaults to 10
    :type zorder: int, optional
    :return: 
     - **chs** (*list*) -- cycle handles (output of **axis.scatter**) for 
        each plotted cycle
    """
    chs = []
    for I_ in range(ncycles):
        TS = t0 + I_*T
        TE = t0 + (I_ + 1)*T
        IND = (Tindex >= TS) & (Tindex < TE)
        XI = Xdata[IND]
        YI = Ydata[IND]
        cbl = axis.scatter(XI,YI,c=(Tindex[IND] - TS)/T,cmap=cmaps[I_],s=1,zorder=zorder)
        chs.append(cbl)
    return chs

def get_lims(XI, YI, PADXY): 
    """Get the xlimits and ylimits of input data vectors
    plus a fractional padding based on the range of these
    vectors

    :param XI: x data vector
    :type XI: array-like
    :param YI: y data vector
    :type YI: array-like
    :param PADXY: fractional range to use as padding beyond
        the min/max of XI and YI values
        e.g., PADXY=0.05 adds a 5% padding to the limits of XI and YI
    :type PADXY: float
    :returns:
     - **xlims** (*tuple*) -- padded x-min and x-max values
     - **ylims** (*tuple*) -- padded y-min and y-max values
    """    
    xlims = (np.nanmin(XI) - PADXY*(np.nanmax(XI) - np.nanmin(XI)),\
            np.nanmax(XI) + PADXY*(np.nanmax(XI) - np.nanmin(XI)))
    ylims = (np.nanmin(YI) - PADXY*(np.nanmax(YI) - np.nanmin(YI)),\
            np.nanmax(YI) + PADXY*(np.nanmax(YI) - np.nanmin(YI)))
    return xlims, ylims

# Logging Subroutines
class CriticalExitHandler(logging.Handler):
    """A custom :class:`~logging.Handler` sub-class that emits a sys.exit
    if a logging instance emits a logging.CRITICAL level message

    Constructed through a prompt to ChatGPT and independently
    tested by the author.

    """
    def __init__(self, exit_code=1, **kwargs):
        """Initialize a CritiaclExitHandler object

        :param exit_code: exit code to associate with system-exit, defaults to 1
        :type exit_code: int, optional
        """        
        super().__init__(**kwargs)
        self.exit_code = exit_code
        
    def emit(self, record):
        """Update the :meth:`~logging.Handler.emit` method
        to trigger a :meth:`~sys.exit` call if a CRITICAL
        level logger message is raised by any script this
        handler/logger is associated with

        :param record: message record emitted from a script
            associated with this handler/logger
        :type record: logging.LogRecord
        """        
        if record.levelno == logging.CRITICAL:
            sys.exit(self.exit_code)

def rich_error_message(e):
    """Given the raw output of an "except Exception as e"
    formatted clause, return a string that emulates
    the error message print-out from python

    e.g., 
    'TypeError: input "input" is not type int'

    :param e: Error object
    :type e: _type_
    :return: error message
    :rtype: str
    """    
    etype = type(e).__name__
    emsg = str(e)
    return f'{etype}: {emsg}'

def setup_terminal_logger(name, level=logging.INFO):
    """QuickStart setup for a write-to-command-line-only
    logger that has safety catches against creating
    multiple StreamHandlers with repeat calls of a
    script in ipython or similar interactive python session.

    :param name: name for the logger
        e.g., __name__ is fairly customary
    :type name: str
    :param level: logging level, defaults to logging.INFO
    :type level: int, optional
    :return: Logger
    :rtype: logging.RootLogger
    """    
    ## SET UP LOGGING
    Logger = logging.getLogger(name)
    # Set logging level to INFO
    Logger.setLevel(level)

    # Prevent duplication during testing
    # Solution from https://stackoverflow.com/questions/31403679/python-logging-module-duplicated-console-output-ipython-notebook-qtconsole
    # User Euclides (Sep 8, 2015)
    handler_console = None
    handlers = Logger.handlers
    for h in handlers:
        if isinstance(h, logging.StreamHandler):
            handler_console = h
            break
    # Set up logging to terminal
    if handler_console is None:
        ch = logging.StreamHandler()
        # Set up logging line format
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(fmt)
        # Add formatting & handler
        Logger.addHandler(ch)
    return Logger