#requred libraries --------------------------------------------------
from asyncio.windows_events import NULL
import csv
from itertools import zip_longest
from math import ceil
from operator import index
from tkinter import font
from turtle import width
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.tools as tls

st.set_page_config(page_title='Signal Sampler Studio', layout = 'wide', initial_sidebar_state = 'auto')


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .css-18e3th9 {
            flex: 1 1 0%;
            width: 100%;
            padding: 1.5rem 4rem 0rem;
            min-width: auto;
            max-width: initial;
              }
                      .css-18ni7ap {
            position: fixed;
            top: 0px;
            left: 0px;
            right: 0px;
            height: 0rem;
            background: rgb(255, 255, 255);
            outline: none;
            z-index: 999990;
            display: block;
             }
            .css-k3w14i {
            font-size: 18px;
            color: rgb(250, 250, 250);
            display: flex;
            visibility: visible;
            margin-bottom: 0.5rem;
            height: auto;
            min-height: 1.5rem;
            vertical-align: middle;
            flex-direction: row;
            -webkit-box-align: center;
            align-items: center;
             }
            .css-1inwz65 {
            line-height: 1.6;
            font-weight: normal;
            font-size: 18px;
            font-family: "Source Code Pro", monospace;
            color: inherit;
            } 
            .css-1avcm0n {
            position: fixed;
            top: 0px;
            left: 0px;
            right: 0px;
            height: 0rem;
            background: rgb(14, 17, 23);
            outline: none;
            z-index: 999990;
            display: block;
             }
            </style>
            """


st.markdown(hide_streamlit_style, unsafe_allow_html=True) 




def interpolate(time_domain, samples_of_time, samples_of_amplitude):
    
    # shape = (nsamples_of_time, ntime_domain), nsamples_of_time copies of time_domain df span axis 1
    resizing = np.resize(time_domain, (len(samples_of_time), len(time_domain)))

    
    # Must take transpose of u for proper broadcasting with samples_of_time.
    # shape = (ntime_domain, nsamples_of_time), v(samples_of_time) df spans axis 1
    matrix_of_pre_interpolation = (samples_of_time - resizing.T) / (samples_of_time[1] - samples_of_time[0])
    # shape = (nx, nxp), m(v) df spans axis 1
    matrix_of_interpolation = samples_of_amplitude * np.sinc(matrix_of_pre_interpolation)
    
    # Sum over m(v) (axis 1)
    samples_of_amplitude_at_time_domain = np.sum(matrix_of_interpolation, axis = 1)

    

   

    return samples_of_amplitude_at_time_domain

def add_to_plot(ax,time,f_amplitude, shape, label):
  ax.plot( time , f_amplitude, shape, alpha=1, linewidth=2, label=label)  # 'shape' express the color or the shape , alpha represent the brightness of the line
            
def show_plot(f):
    f.set_figwidth(4)
    f.set_figheight(8)
    plotly_fig = tls.mpl_to_plotly(f)
    plotly_fig.update_layout(font=dict(size=16), xaxis_title= "Time (second)", yaxis_title= 'Voltage (mV)', showlegend = True)     
    # plotly_fig['data'][0]['name']='Number of Samples'
    st.plotly_chart(plotly_fig, use_container_width=True, sharing="streamlit")

def init_plot():
  f, ax = plt.subplots(1,1,figsize=(10,10))                       # increment phase based on current frequency
  ax.yaxis.set_tick_params(length=5)                              # to draw the y-axis line (-) for points
  ax.xaxis.set_tick_params(length=0)                              # to draw the y-axis line (-) for points
  ax.grid(c='gray', lw=0.1, ls='--')                             #lw represent the line width of the grid
  legend = ax.legend()
  legend.get_frame().set_alpha(1)
  for spine in ('top', 'right', 'bottom', 'left'):                #control the border lines to be visible or not and the width of them
    ax.spines[spine].set_visible(False)
  return f,ax

global time,signal
def addNoise(snr):
       power=df['signal']**2
       snr_dp=snr
       signal_average_power=np.mean(power)
       signal_averagepower_dp=10*np.log10(signal_average_power)
       noise_dp=signal_averagepower_dp-snr_dp
       noise_watts=10**(noise_dp/10)
       mean_noise=0
       noise=np.random.normal(mean_noise,np.sqrt(noise_watts),len(df['signal']))
       df['signal']=df['signal']+noise
       ax.plot(label='Signal with noise')
       add_to_plot(ax,time,df['signal'],'r', label='Signal with noise')


col1, col2 = st.columns([1,3])
with col1:
  # read the df from the csv file and store it in variable named df
  # df = pd.read_csv("1_2.csv",nrows=250) #read only the firt nrows row from the file 
  uploaded_file = st.file_uploader(label="Upload your Signal",
          type=['csv', 'xslx'])

  
  if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, nrows=1000)
    f,ax = init_plot()
    snr = st.slider('Select SNR', 1, 50, key=0, value=50)
    # converting column df to list
    time = df['time'] # time will carry the values of the time 
    signal=df['signal'] # f_amplitude will carry the values of the amplitude of the signal
    #plt.plot(time,f_amplitude) #draw time and f_amplitude
    addNoise(snr)  #draw time and f_amplitude      'c': the wanted color
    # show_plot(f)   #show the drawing



    max_of_time = (max(time))
    
    Number_Of_Samples = st.slider('Sampling Rate', min_value= 2, max_value =int(len(df)/(ceil(max_of_time))))  #number of samples we want to take from the df
    time_samples = [] # the list which will carry the values of the samples of the time
    signal_samples = [] # the list which will carry the values of the samples of the amplitude
    for i in range(0, df.shape[0], df.shape[0]//(Number_Of_Samples*(ceil(max_of_time)))): #take only the specific Number_Of_Samples from the df
        time_samples.append(df.iloc[:,0][i])  # take the value of the time
        signal_samples.append(df.iloc[:,1][i]) #take the value of the amplitude
    add_to_plot(ax, time_samples, signal_samples, 'ko', label='Number of Samples') #draw the samples of time and f_amplitude as small black circles

    def convert_df(dff):
      # IMPORTANT: Cache the conversion to prevent computation on every rerun
      return dff.to_csv().encode('utf-8')

    data = {'time':time,'signal':df['signal']}
    dff = pd.DataFrame(data)
    csv = convert_df(dff)
    st.download_button(
        label="Download reconstructed data",
        data=csv,
        file_name='large_df.csv',
        mime='text/csv',
    )
  else:
    st.write("Generate Your Own Signal")
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
    time=np.arange(0,1,0.01)                         # x axis domain from 0 to 1 and with step 0.01
    
    

    with st.form(key='df1'):                                    #generate form with unique key called df1
      Type_of_signal= st.selectbox('Select The Signal Type', ['None','sin', 'cos'], key=1) #det the sugnal type
      frequency = st.slider("Enter the Frequency", min_value=0, max_value = 100 )          #det the frequency value by slider
      amplitude = st.slider("Enter the Amplitude", min_value=0, max_value = 100 )          #det the amplitudes value by slider
      submit= st.form_submit_button(label = "Submit")                                      #submit to save the data in cache memory

      #update the list of signals information------------------------------
      if submit:
        n="signal"+str(len(signal_amp))
        new_data = {'signal_name':n,'signal_type': Type_of_signal, 'signal_freq': frequency, 'signal_amp': amplitude}
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
    
    f,ax = init_plot() 
    if len(signal_name)!= 0:   
      sum_of_signal_values=np.zeros(len(signal_values[0]))  #array of zeros to add the list in it   
      for i in range(len(signal_values)): 
        sum_of_signal_values+=np.array(signal_values[i])    #transmit the values in the array   
    

      collected_data=[time,sum_of_signal_values]
      export_data=zip_longest(*collected_data,fillvalue='')
      with open('num.csv','w',encoding="ISO-8859-1", newline='') as myfile:
        wr=csv.writer(myfile)
        wr.writerow(("time","signal"))
        wr.writerows(export_data)
      myfile.close()



    df = pd.read_csv('num.csv')
    time = df['time']
    signals=df['signal']



    max_of_time = (max(time))
    snr = st.slider('Select SNR', 1, 50, key=0, value=50)
  
    Number_Of_Samples = st.slider('Sampling Rate', min_value= 2, max_value =int(len(df)/(ceil(max_of_time))))  #number of samples we want to take from the df
    time_samples = [] # the list which will carry the values of the samples of the time
    signal_samples = [] # the list which will carry the values of the samples of the amplitude
    for i in range(0, df.shape[0], df.shape[0]//(Number_Of_Samples*(ceil(max_of_time)))): #take only the specific Number_Of_Samples from the df
        time_samples.append(df.iloc[:,0][i])  # take the value of the time
        signal_samples.append(df.iloc[:,1][i]) #take the value of the amplitude
    add_to_plot(ax, time_samples, signal_samples, 'ko', label='Number of Samples') #draw the samples of time and f_amplitude as small black circles
    ans = interpolate(time, time_samples, signal_samples) # result of reconstruction
    # plt.legend(['Constructed Signal'])
    add_to_plot(ax, time, ans, 'c', label='Reconstructed Signal')  #draw the reconstructed signal   'c': the wanted color in BLUE
    
    
    
    if len(signal_name)!=0:
      add_to_plot(ax,time,sum_of_signal_values,'y', label='Original Signal')  
      def convert_df(dff):
      # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return dff.to_csv().encode('utf-8')
      data = {'time':time,'signal':ans}
      dff = pd.DataFrame(data)
      csv = convert_df(dff)
      st.download_button(
          label="Download reconstructed data",
          data=csv,
          file_name='large_df.csv',
          mime='text/csv',
      )

      addNoise(snr)
  with col2:

    if uploaded_file is not None:
      time_domain = np.linspace(0, max_of_time, (Number_Of_Samples*(ceil(max_of_time))))  # the domain we want to draw the recounstructed signal in it
      ans = interpolate(time_domain, time_samples, signal_samples) # result of reconstruction
      add_to_plot(ax, time_domain, ans, 'c', label="Constructed Signal")  #draw the reconstructed signal   'c': the wanted color in BLUEEEEEEEEE

    else:
      delete = st.selectbox("Select signal to be removed",signal_name)
      delete_button=st.button(label="Delete")
      if delete_button:
        df1 =  df1[df1.signal_name != delete] 
        df1.to_csv(filepath, index=False)
        
    show_plot(f)   #show the drawing














