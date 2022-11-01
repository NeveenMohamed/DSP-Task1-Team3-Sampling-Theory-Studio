# Signal Viewer
## introduction 
### sampling an analog signal is a crucial step for any digital signal processing system 
## description 
### it is a web application that illustrates the signal sampling and recovery showing the importance and validation of the nyquist rate
### Our application have the following features:
#### -> visualize and sample a signal and use the sampled points to recover the original signal
#### -> prepare mixed signal by adding sinusoidal signals with different frequancy and magnitudes 
#### -> remove any component from the mixed signal
#### -> adding noise to the loaded signal 
#### -> resize the signals without missing the UI
## technology used 
### python with streamlit
## Team Members:
### Saeed Elsayed   
###  Maryam Megahed
###  Mazen Tarek
###  Neveen Mohamed

## Screenshots of the web app
generating sin wave with frequency 4hz and amplitude 2v 
![Screenshot 2022-11-01 180335](https://user-images.githubusercontent.com/92316869/199279986-e0faf0f2-4f02-46d3-9f7d-b4de2fe9c164.png)
sampling it with 2 fmax(Nyquist rate)
![Screenshot 2022-11-01 181008](https://user-images.githubusercontent.com/92316869/199281708-43bbdda2-a048-4792-8dc8-94b70aff34d8.png)
reconstructing the signal
![Screenshot 2022-11-01 181215](https://user-images.githubusercontent.com/92316869/199282014-4d7f8b9c-166c-4c04-89a6-88eda4bd3a14.png)
adding noise to the generated signal
![Screenshot 2022-11-01 181526](https://user-images.githubusercontent.com/92316869/199282599-b53323a2-40ee-48dd-8269-daccc74a6c5b.png)




