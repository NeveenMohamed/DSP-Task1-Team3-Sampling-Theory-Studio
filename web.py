from ast import Try
import streamlit as st
import plotly_express as px
import pandas as pd
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.title("Signal Viewer App")

st.write("Signal 1")
# st.sidebar.subheader("Visulization Settings")

# File upload

uploaded_file = st.file_uploader(label="Upload your Signal",
type=['csv', 'xslx'])


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

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

    fig_1 = go.Figure(data=[trace], frames=frames, layout=layout)  
    layout.update(xaxis =dict(range=[x_min, x_max], autorange=False),
                yaxis =dict(range=[-3,3]), 
                title="Signal 1",
                )
    # fig.add_annotation(text = (f"@reshamas / {today}<br>Source: JHU CSSE"), showarrow=False, x = 0, 
    #                    y = -0.11, xref='paper', yref='paper', xanchor='left', yanchor='bottom', xshift=-3,
    #                    yshift=-15, font=dict(size=10, color="grey"), align="left")


    st.write(fig_1)


    noise = st.slider('Select noise', 1, 30)
    st.button('Add',noise,'HZ')

    st.write("Signal 2")
    uploaded_file_2 = st.file_uploader(label="Upload Signal 2",
    type=['csv', 'xslx'])


    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file_2)

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

        fig_2 = go.Figure(data=[trace], frames=frames, layout=layout)  
        layout.update(xaxis =dict(range=[x_min, x_max], autorange=False),
                    yaxis =dict(range=[-3,3]), 
                    title="Signal 2",
                    )
        st.write(fig_2)
                    