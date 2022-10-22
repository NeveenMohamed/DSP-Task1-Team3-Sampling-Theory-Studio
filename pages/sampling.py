import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.tools as tls

def add_to_plot(ax,time,f_amplitude, shape):
  ax.plot( time , f_amplitude, shape, alpha=0.7, linewidth=2)  # 'shape' express the color or the shape , alpha represent the brightness of the line
            
def show_plot(f):
    plotly_fig = tls.mpl_to_plotly(f)     
    st.plotly_chart(plotly_fig, use_container_width=True, sharing="streamlit")

def init_plot():
  f, ax = plt.subplots(1,1,figsize=(10,10))                       # increment phase based on current frequency
  ax.set_xlabel('Time (second)')                                  # the x_axis title
  ax.yaxis.set_tick_params(length=5)                              # to draw the y-axis line (-) for points
  ax.xaxis.set_tick_params(length=0)                              # to draw the y-axis line (-) for points
  ax.grid(c='#D3D3D3', lw=1, ls='--')                             #lw represent the line width of the grid
  legend = ax.legend()
  legend.get_frame().set_alpha(1)
  for spine in ('top', 'right', 'bottom', 'left'):                #control the border lines to be visible or not and the width of them
    ax.spines[spine].set_visible(False)
  
  return f,ax

# read the df from the csv file and store it in variable named df
# df = pd.read_csv("1_2.csv",nrows=250) #read only the firt nrows row from the file 
uploaded_file = st.file_uploader(label="Upload your Signal",
        type=['csv', 'xslx'])

if uploaded_file is not None:
  df = pd.read_csv(uploaded_file, nrows=250)
  time = df['time']
  f,ax = init_plot()

  # converting column df to list
  time = df['time'].tolist() # time will carry the values of the time 
  f_amplitude = df['signal'].tolist() # f_amplitude will carry the values of the amplitude of the signal
  #plt.plot(time,f_amplitude) #draw time and f_amplitude
  add_to_plot(ax, time, f_amplitude, 'c')  #draw time and f_amplitude      'c': the wanted color
  # show_plot(f)   #show the drawing


  Number_Of_Samples = st.slider('Enter the number of samples required', min_value= 2, max_value =len(df))  #number of samples we want to take from the df
  time_samples = [] # the list which will carry the values of the samples of the time
  signal_samples = [] # the list which will carry the values of the samples of the amplitude
  for i in range(0, df.shape[0], df.shape[0]//Number_Of_Samples): #take only the specific Number_Of_Samples from the df
      time_samples.append(df.iloc[:,0][i])  # take the value of the time
      signal_samples.append(df.iloc[:,1][i]) #take the value of the amplitude
  add_to_plot(ax, time_samples, signal_samples, 'ko') #draw the samples of time and f_amplitude as small black circles

  show_plot(f) #show the drawing


# function that make the interpolation
def interpolate(time_domain, samples_of_time, samples_of_amplitude, left = None, right = None):
    """One-dimensional Whittaker-Shannon interpolation.

    This uses the Whittaker-Shannon interpolation formula to interpolate the
    value of samples_of_amplitude (array), which is defined over samples_of_time (array), at time_domain (array or
    float).

    Returns the interpolated array with dimensions of time_domain.

    """
    scalar = np.isscalar(time_domain)
    if scalar:
        time_domain = np.array(time_domain)
        time_domain.resize(1)
    # shape = (nsamples_of_time, ntime_domain), nsamples_of_time copies of time_domain df span axis 1
    u = np.resize(time_domain, (len(samples_of_time), len(time_domain)))
    # Must take transpose of u for proper broadcasting with samples_of_time.
    # shape = (ntime_domain, nsamples_of_time), v(samples_of_time) df spans axis 1
    v = (samples_of_time - u.T) / (samples_of_time[1] - samples_of_time[0])
    # shape = (nx, nxp), m(v) df spans axis 1
    m = samples_of_amplitude * np.sinc(v)
    # Sum over m(v) (axis 1)
    samples_of_amplitude_at_time_domain = np.sum(m, axis = 1)

    # Enforce left and right
    if left is None: left = samples_of_amplitude[0]
    samples_of_amplitude_at_time_domain[time_domain < samples_of_time[0]] = left
    if right is None: right = samples_of_amplitude[-1]
    samples_of_amplitude_at_time_domain[time_domain > samples_of_time[-1]] = right

    # Return a float if we got a float
    if scalar: return float(samples_of_amplitude_at_time_domain)

    return samples_of_amplitude_at_time_domain

f,ax = init_plot()  # make a new graph to draw in it

time_domain = np.linspace(0, max(time), Number_Of_Samples)  # the domain we want to draw the recounstructed signal in it
ans = interpolate(time_domain, time_samples, signal_samples) # result of reconstruction


add_to_plot(ax, time_domain, ans, 'c')  #draw the reconstructed signal   'c': the wanted color
show_plot(f)   #show the drawing