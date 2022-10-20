from ast import Try
import streamlit as st
import numpy as np
import plotly_express as px
import pandas as pd
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.graph_objects as go
global time,signal
def addNoise(snr):
       power=signal**2
       snr_dp=snr
       signal_average_power=np.mean(power)
       signal_averagepower_dp=10*np.log10(signal_average_power)
       noise_dp=signal_averagepower_dp-snr_dp
       noise_watts=10**(noise_dp/10)
       mean_noise=0
       noise=np.random.normal(mean_noise,np.sqrt(noise_watts),len(signal))
       noise_signal=signal+noise
       st.write("signal with noise")
       draw(time,noise_signal)
def draw(time,y):
    d={'t':time,'y':y}
    signal=pd.DataFrame(data=d)
    fig,ax=plt.subplots()
    ax.plot(signal.iloc[:,0],signal.iloc[:,1])
    ax.set_title("Signal Digram")
    ax.set_xlabel("time")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    st.pyplot(fig) 
    
st.title("Signal Viewer App")

st.write("Signal 1")
# st.sidebar.subheader("Visulization Settings")

# File upload

uploaded_file = st.file_uploader(label="Upload your Signal",
type=['csv', 'xslx'])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    time = df['Time']
    signal=df['Voltage']


    timeval = 'Time'
    x_min = '-3'
    x_max = '3'
    y_min = '0.0000000000000022'
    y_max = '-0.0000000000000022'
    voltage_value=['Voltage']
    name1 = voltage_value[0]
    group1 = df[name1].tolist()


    trace = go.Scatter (
        x= df[timeval][:-3],
        y= time[:0],
        mode='lines',
        line = dict(width=1.5))

    frames = [dict(data= [dict(type='scatter',
                            x=df[timeval][:k],
                            y=group1[:k]),],
                traces= [0, 1],  # frames[k]['data'][0]  updates trace
                )
            for k  in  range(1, len(group1)-1)] 

    layout = go.Layout(width=800,
                    height=500,
                    showlegend=True,
                    hovermode='closest',
                    updatemenus=[dict(type='buttons', showactive=False,
                                    y=1.05,
                                    x=1.15,
                                    xanchor='right',
                                    yanchor='bottom',
                                    pad=dict(t=0, r=10),
                                    buttons=[dict(label='Play',
                                                method='animate',
                                                args=[None, 
                                                        dict(frame=dict(duration=0.1, 
                                                                        redraw=False),
                                                            transition=dict(duration=0),
                                                            fromcurrent=True,
                                                            mode='immediate')
                                                    ])
                                            ])
                                ,dict(type='buttons', showactive=False,
                                    y=0.55,
                                    x=1.15,
                                    xanchor='right',
                                    yanchor='bottom',
                                    pad=dict(t=0, r=10),
                                    buttons=[dict(label='Stop',
                                                method='restyle',
                                                args=[None, 
                                                        dict(frame=dict(duration=0.1, 
                                                                        redraw=False),
                                                            transition=dict(duration=0),
                                                            fromcurrent=True,
                                                            mode='immediate')
                                                    ])
                                            ])
                                ],
                    )

    fig_1 = go.Figure(data=[trace], frames=frames, layout=layout)  
    layout.update(xaxis =dict(range=[x_min, x_max], autorange=False),
                yaxis =dict(range=[-3,3]), 
                title="Signal 1",
                )
    # fig.add_annotation(text = (f"@reshamas / {today}<br>Source: JHU CSSE"), showarrow=False, x = 0, 
    #                    y = -0.11, xref='paper', yref='paper', xanchor='left', yanchor='bottom', xshift=-3,
    #                    yshift=-15, font=dict(size=10, color="grey"), align="left")


    st.write(fig_1)


    snr = st.slider('Select SNR', 1, 30)
#     st.button('Add',snr,'HZ')
    addNoise(snr)
    