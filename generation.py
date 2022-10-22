#requred libraries --------------------------------------------------
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import plotly.tools as tls

# CSV Folder Path For Signal Information
file_dir = r'C:\Users\Mazen Tarek\Desktop\DSP-Task-1-webApplication-signalViewer'
file_name = 'test1.csv'
filepath = f"{file_dir}/{file_name}"

st.set_page_config(layout="wide")


# function to implement the grid and the interfacing of signal plotting
def init_plot():
  f, ax = plt.subplots(1,1,figsize=(10,10))                       
  ax.set_xlabel('Time (second)')              # the x_axis title
  ax.yaxis.set_tick_params(length=5)          # to draw the y-axis line (-) for points
  ax.xaxis.set_tick_params(length=0)          # to draw the y-axis line (-) for points
  ax.grid(c='#D3D3D3', lw=1, ls='--')         #lw represent the line width of the grid
  legend = ax.legend()      
  legend.get_frame().set_alpha(1)
  for spine in ('top', 'right', 'bottom', 'left'):  #control the border lines to be visible or not and the width of them
    ax.spines[spine].set_visible(False)
  return  f,ax


#function to pass the needed parameters for plotting
def add_to_plot(ax,x,y,colour,label):
  ax.plot(x,y,colour, alpha=0.7, linewidth=2, label=label)  # 'b' express the blue color , alpha represent the brightness of the line

#function to show the plotting signal
def show_plot(f):
  plotly_fig = tls.mpl_to_plotly(f)
  # st.pyplot(f) 
  st.plotly_chart(plotly_fig, use_container_width=True, sharing="streamlit")       

#our main function
def main():
  
  #read the csv files from the pc
  df1 = pd.read_csv(filepath)


  #form to generate the signal with det. freq and amplitude -----------
  with st.sidebar.form(key='df1', clear_on_submit=True):                                    #generate form with unique key called df1
       name_of_signal = st.text_input(label = "Enter the Signal Name")                      #pass the signal name from the user
       Type_of_signal= st.selectbox('Select The Signal Type', ['None','sin', 'cos'], key=1) #det the sugnal type
       frequency = st.slider("Enter the Frequency", min_value=0, max_value = 100 )          #det the frequency value by slider
       amplitude = st.slider("Enter the Amplitude", min_value=0, max_value = 100 )          #det the amplitudes value by slider
       submit= st.form_submit_button(label = "Submit")                                      #submit to save the data in cache memory

       #update the list of signals information------------------------------
       if submit:
          new_data = {'signal_name': name_of_signal, 'signal_type': Type_of_signal, 'signal_freq': frequency, 'signal_amp': amplitude}
          df1 = df1.append(new_data, ignore_index=True)      #add the new information from the user to the csv file 
          df1.to_csv(filepath, index=False)
  
  #signal generation---------------------------------------------------
  st.title("Generate Your Own Signal")
  #variables defintion ------------------------------------------------
  signal_values = []                 #list to save the signal values along y axis
  dt = 0.01                          # time step
  phi = 0                            # phase accumulator
  phase=[]                           #list to save the omega values along the time domain
  c=[]                       
  signal_name = df1['signal_name'].tolist()     #import the signal names from the csv file into a list
  signal_type = df1['signal_type'].tolist()     #import the signal type from the csv file into a list
  signal_freq = df1['signal_freq'].tolist()     #import the freq values into a list
  signal_amp = df1['signal_amp'].tolist()       #import the amp values into a list
  x=np.arange(0,1,0.01)              # x axis domain from 0 to 1 and with step 0.01
  colours=['r','g','b']              # list of coloue=rs used to draw the signals
  color_index=0                      # init the index value for the colours list




  for i in range(0, len(signal_freq)):     #convert the list of freq string values '4' to an integer value 4  by looping
    signal_freq[i] = int(signal_freq[i])   

  for i in range(0, len(signal_amp)):      #convert the list of amp string values '4' to an integer value 4  by looping
    signal_amp[i] = int(signal_amp[i])  

  f, ax = init_plot()                      #call the init function to implement the grid
  
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
    add_to_plot(ax,x,signal_values[n],colours[color_index],signal_name[n])
    color_index=(color_index+1)%3

  show_plot(f) 

#function to implement the sum of the signal values(y-axis) within the x-axis 
  f,ax=init_plot()                                      #init a new grid to plot the sum of the signals
  sum_of_signal_values=np.zeros(len(signal_values[0]))  #array of zeros to add the list in it   
  for i in range(len(signal_values)): 
    sum_of_signal_values+=np.array(signal_values[i])    #transmit the values in the array   

  add_to_plot(ax,x,sum_of_signal_values,'b','sum')      
  show_plot(f)

if __name__ == '__main__':
	main()  

      
   