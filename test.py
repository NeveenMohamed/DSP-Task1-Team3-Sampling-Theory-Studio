#requred libraries --------------------------------------------------
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.tools as tls
import csv 
from itertools import zip_longest
from math import ceil





global time,signal

st.set_page_config(layout="wide")

def addNoise(snr):
       df = pd.read_csv("numbers.csv", nrows=1000)
       power=df['sum_of_yaxis']**2
       snr_dp=snr
       signal_average_power=np.mean(power)
       signal_averagepower_dp=10*np.log10(signal_average_power)
       noise_dp=signal_averagepower_dp-snr_dp
       noise_watts=10**(noise_dp/10)
       mean_noise=0
       noise=np.random.normal(mean_noise,np.sqrt(noise_watts),len(df['sum_of_yaxis']))
       df['sum_of_yaxis']=df['sum_of_yaxis']+noise
       st.write("Signal with noise")
       add_to_plot(ax,time,df['sum_of_yaxis'],'r')

def add_to_plot(ax,time,f_amplitude, shape):
  ax.plot( time , f_amplitude, shape, alpha=0.7, linewidth=2)  # 'shape' express the color or the shape , alpha represent the brightness of the line
            
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



def draw(time,y):
    d={'t':time,'y':y}
    signal=pd.DataFrame(data=d)
    fig,go=plt.subplots()
    go.plot(signal.iloc[:,0],signal.iloc[:,1])
    go.set_title("Signal Digram")
    go.set_xlabel("time")
    go.set_ylabel("Amplitude")
    go.grid(True)
    # st.pyplot(fig) 
    # st.write(fig)
    plotly_fig = tls.mpl_to_plotly(fig)
    # st.write(plotly_fig)
    st.plotly_chart(plotly_fig, use_container_width=True, sharing="streamlit")       

col1, col2 = st.columns([1,2])
with col1:
  # CSV Folder Path For Signal Information
  file_dir = r'C:\Users\Mazen Tarek\Desktop\DSP-Task-1-webApplication-signalViewer'
  file_name = 'test1.csv'
  filepath = f"{file_dir}/{file_name}"
  #read the csv files from the pc
  df1 = pd.read_csv(filepath)

  

  #variables defintion ------------------------------------------------
  signal_values = []                            #list to save the signal values along y axis
  dt = 0.01                                     # time step
  phi = 0                                       # phase accumulator
  phase=[]                                      #list to save the omega values along the time domain
  c=[]                       
  signal_name = df1['signal_name']              #import the signal names from the csv file into a list
  signal_type = df1['signal_type']              #import the signal type from the csv file into a list
  signal_freq = df1['signal_freq']              #import the freq values into a list
  signal_amp = df1['signal_amp']                #import the amp values into a list
  x=np.arange(0,1,0.01)                         # x axis domain from 0 to 1 and with step 0.01
  colours=['r','g','b']                         # list of coloue=rs used to draw the signals
  color_index=0                                 # init the index value for the colours list

      

 
snr = st.slider('Select SNR', 1, 30, key=0)
f,ax = init_plot()

#signal generation---------------------------------------------------
st.title("Generate Your Own Signal")
with st.form(key='df1'):                       #generate form with unique key called df1
      name_of_signal = st.text_input(label = "Enter the Signal Name")                      #pass the signal name from the user
      Type_of_signal= st.selectbox('Select The Signal Type', ['None','sin', 'cos'], key=1) #det the sugnal type
      frequency = st.slider("Enter the Frequency", min_value=0.0, max_value = 1.0 )          #det the frequency value by slider
      amplitude = st.slider("Enter the Amplitude", min_value=0.0, max_value = 1.0 )          #det the amplitudes value by slider
      submit= st.form_submit_button(label = "Submit")                                      #submit to save the data in cache memory

      #update the list of signals information------------------------------
      if submit:
        new_data = {'signal_name': name_of_signal, 'signal_type': Type_of_signal, 'signal_freq': frequency, 'signal_amp': amplitude}
        df1 = df1.append(new_data, ignore_index=True)      #add the new information from the user to the csv file 
        df1.to_csv(filepath, index=False)

      for i in range(0, len(signal_freq)):     #convert the list of freq string values '4' to an integer value 4  by looping
        signal_freq[i] = int(signal_freq[i])   

      for i in range(0, len(signal_amp)):      #convert the list of amp string values '4' to an integer value 4  by looping
        signal_amp[i] = int(signal_amp[i])  


      for n in range(0,len(signal_amp)):            #for loop to save the x and y axis values for signals
         phase.append(2*np.pi*signal_freq[n]*dt)     #det the omega value for each signal in csv file row
         for i in np.arange(0,1,0.01):
            if signal_type[n]=='cos':                 # cos of current phase
              c.append(signal_amp[n]*np.cos(phi))     
            else:
              c.append(signal_amp[n]*np.sin(phi))     # sine of current phase
            phi=phi+phase[n]                          #implement the omega t within the time domain
         signal_values.append(c)
         c=[]
    #function to implement the sum of the signal values(y-axis) within the x-axis
sum_of_signal_values=np.zeros(len(signal_values[0]))  #array of zeros to add the list in it   
for i in range(len(signal_values)): 
    sum_of_signal_values+=np.array(signal_values[i])    #transmit the values in the array   
    color_index=(color_index+1)%3
   
    #add the two column of tha y amplitude
    sum_of_yaxis= sum_of_signal_values
    d = [x, sum_of_signal_values,sum_of_yaxis]
    export_data = zip_longest(*d, fillvalue = '')
    with open('numbers.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("x", "sum_of_signal_values","signal","sum_of_yaxis"))
        sum_of_yaxis = sum_of_signal_values
        wr.writerows(export_data)
    myfile.close()
    add_to_plot(ax,x,sum_of_yaxis,colours[color_index])
    addNoise(snr)      
    plotly_fig = tls.mpl_to_plotly(f)


      


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

# f,ax = init_plot()  # make a new graph to draw in it
df = pd.read_csv("numbers.csv", nrows=1000)
time = df['x']
signal=df['sum_of_yaxis']
max_of_time = (max(time))
Number_Of_Samples = st.slider('Enter the number of samples required', min_value= 2, max_value =int(len(df)/(ceil(max_of_time))))  #number of samples we want to take from the df
time_samples = [] # the list which will carry the values of the samples of the time
signal_samples = [] # the list which will carry the values of the samples of the amplitude
for i in range(0, df.shape[0], df.shape[0]//(Number_Of_Samples*(ceil(max_of_time)))): #take only the specific Number_Of_Samples from the df
          time_samples.append(df.iloc[:,0][i])  # take the value of the time
          signal_samples.append(df.iloc[:,1][i]) #take the value of the amplitude
add_to_plot(ax, time_samples, signal_samples, 'ko') #draw the samples of time and f_amplitude as small black circles

time_domain = np.linspace(0, max(time), Number_Of_Samples)  # the domain we want to draw the recounstructed signal in it
ans = interpolate(time_domain, time_samples, signal_samples) # result of reconstruction


with col2:
  add_to_plot(ax, time_domain, ans, 'c')  #draw the reconstructed signal   'c': the wanted color
  show_plot(f)   #show the drawing
  addNoise(snr)  #draw time and f_amplitude      'c': the wanted color

    