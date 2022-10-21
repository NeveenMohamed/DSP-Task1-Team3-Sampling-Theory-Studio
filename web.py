from ast import Try
from turtle import width
import streamlit as st
import plotly_express as px
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.tools as tls

st.set_page_config(layout="wide")

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
       st.write("Signal with noise")
       draw(time,noise_signal)
def draw(time,y):
    d={'t':time,'y':y}
    signal=pd.DataFrame(data=d)
    fig,go=plt.subplots()
    go.plot(signal.iloc[:,0],signal.iloc[:,1])
    go.set_title("Signal Digram")
    go.set_xlabel("Time")
    go.set_ylabel("Amplitude")
    go.grid(True)
    # st.pyplot(fig) 
    # st.write(fig)
    plotly_fig = tls.mpl_to_plotly(fig)
    # st.write(plotly_fig)
    st.plotly_chart(plotly_fig, use_container_width=True, sharing="streamlit")

st.title("Signal Viewer App")

tab1, tab2 = st.tabs(["Signal 1", "Signal 2"])


with tab1:
    col1, col2 = st.columns([1,2])

    with col1:

        # st.write("Signal 1")
        # st.sidebar.subheader("Visulization Settings")

        # File upload

        uploaded_file = st.file_uploader(label="Upload your Signal",
        type=['csv', 'xslx'])

        snr = st.slider('Select SNR', 1, 30, key="1")

    with col2:
        if uploaded_file is not None:

            df = pd.read_csv(uploaded_file)
            
            time = df['Time']
            signal=df['Voltage']

            low = df['Time'].tolist()

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
                y= low[:0],
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

            layout.update(xaxis =dict(range=[x_min, x_max], autorange=False),
                        yaxis =dict(range=[-3,3]), 
                        title="Signal 1",
                        )
            fig_1 = go.Figure(data=[trace], frames=frames, layout=layout)  
            # fig.add_annotation(text = (f"@reshamas / {today}<br>Source: JHU CSSE"), showarrow=False, x = 0, 
            #                    y = -0.11, xref='paper', yref='paper', xanchor='left', yanchor='bottom', xshift=-3,
            #                    yshift=-15, font=dict(size=10, color="grey"), align="left")


            st.write(fig_1)


            
            # st.button('Add',addNoise(snr),'HZ')
            addNoise(snr)


# col1,ce, col2 = st.columns([1,0.8,2])

with tab2:
    col1, col2 = st.columns([1,2])

    with col1:
        st.write("Signal 2")
        uploaded_file_2 = st.file_uploader(label="Upload Signal 2",
        type=['csv', 'xslx'])
        snr = st.slider('Select SNR', 1, 30, key="2")

    with col2:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file_2)

            time = df['Time']
            signal=df['Voltage']

            low = df['Time'].tolist()

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
                y= low[:0],
                mode='lines',
                line = dict(width=1.5))

            frames = [dict(data= [dict(type='scatter',
                                    x=df[timeval][:k],
                                    y=group1[:k]),],
                        traces= [0, 1],   
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
            layout.update(xaxis =dict(range=[x_min, x_max], autorange=False),
                        yaxis =dict(range=[-3,3]), 
                        title="Signal 2",
                        )
            fig_2 = go.Figure(data=[trace], frames=frames, layout=layout)  
        
            st.write(fig_2)
            addNoise(snr)
                        