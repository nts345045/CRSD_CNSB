import logging, sys
import pandas as pd
import numpy as np

## Plotting Subroutines ##
def plot_cycles(axis,Xdata,Ydata,Tindex,t0,cmaps,ncycles=5,T=pd.Timedelta(24,unit='hour'),zorder=10):
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
        super().__init__(**kwargs)
        self.exit_code = exit_code
        
    def emit(self, record):
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

# def setup_logger(name, level=logging.INFO, log_file=None, to_terminal=True):
#     bckw = {'level': level}
    
#     if isinstance(log_file, str):
#         bckw.update({'filename': log_file})
    
#     logging.basicConfig(filename = log_file, format='%(asctime)s - %(file)s'
#     Logger = logging.getLogger(name)


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