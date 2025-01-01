import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import *
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from sklearn.cluster import KMeans
main = tkinter.Tk()
main.title("CUSTOMER SEGMENTATION ANALYSIS")
main.geometry("1300x1200")
global filename
global df
global features,Y,kmeans
def upload():
    global filename
    global df
    filename=filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0',END)
    text.insert(END,filename+" loaded\n\n")
    df = pd.read_csv(filename)
    text.insert(END,str(df.head())+"\n")
def processDataset():
    global df
    df.isnull().sum()
    df.dropna(inplace=True)
    text.insert(END,str(df.head())+"\n")
def calculate():
    global features,kmeans,Y
    features = df.iloc[:,[3,4]].values
    kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
    Y=kmeans.fit_predict(features)
    print(Y)
    text.insert(END,str(features[:10])+"\n")
def graph():
    global features,Y,kmeans
    plt.figure(figsize=(8,8))
    plt.scatter(features[Y==0,0],features[Y==0,1],s=50,c='green',label='Cluster 1')
    plt.scatter(features[Y==1,0],features[Y==1,1],s=50,c='red',label='Cluster 2')
    plt.scatter(features[Y==2,0],features[Y==2,1],s=50,c='yellow',label='Cluster 3')
    plt.scatter(features[Y==3,0],features[Y==3,1],s=50,c='violet',label='Cluster 4')
    plt.scatter(features[Y==4,0],features[Y==4,1],s=50,c='blue',label='Cluster 5')
    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=100, c='cyan',label='Centroids')
    plt.title('Customer Groups')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='CUSTOMER SEGMENTATION ANALYSIS')
title.config(bg='yellow3', fg='white')  
title.config(font=font)           
title.config(height=3, width=110)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="UPLOAD DATASET", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=460,y=250)

processButton = Button(main, text="PREPROCESS DATA", command=processDataset)
processButton.place(x=50,y=150)
processButton.config(font=font1)

processButton = Button(main, text="FEAUTURE&SCALING", command=calculate)
processButton.place(x=750,y=100)
processButton.config(font=font1)

processButton = Button(main, text="CLUSTERING GRAPH", command=graph)
processButton.place(x=750,y=150)
processButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=151)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=30,y=300)
text.config(font=font1)


main.config(bg='burlywood2')
main.mainloop()


