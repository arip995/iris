from sklearn.datasets import load_iris
iris=load_iris()      #data type of iris is bunch
print(iris)
####################################################################separate input and output
x=iris.data
y=iris.target
print(x)                             ####### x and y both are numpy array
print(y)
print(x.shape)      #(150,4)
print(y.shape)      #(150)
###############################################split the data set for training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)###########random state is used to fix tedting and training data
print(x_train.shape)        #(120,4)
print(y_train.shape)        #(120,)
print(x_test.shape)         #(30,4)
print(y_test.shape)         #(30,)
                                                                                                                                            
def knn():
    global K
    global acc_Knn
    #####################create a model KNN-K nearest neighbors algorithm
    from sklearn.neighbors import KNeighborsClassifier
    K=KNeighborsClassifier(n_neighbors=5)
    #######################################################train the model by training dataset
    K.fit(x_train,y_train)
    #######################################test the model by testing dataset
    y_pred=K.predict(x_test)
    #########################################find accuracy
    from sklearn.metrics import accuracy_score
    acc_Knn=accuracy_score(y_test,y_pred)
    acc_Knn=round(acc_Knn*100,2)
    m.showinfo(title="KNN",message="accuracy_score in KNN is"+str(acc_Knn)+"%")
def lg():
    global L
    global acc_lg
    from sklearn.linear_model import LogisticRegression
    #######create the model
    L=LogisticRegression(solver='liblinear',multi_class='auto')
    #########train the model
    L.fit(x_train,y_train)
    #######test the model
    y_pred_lg=L.predict(x_test)
    ##################find the accuracy
    from sklearn.metrics import accuracy_score
    acc_lg=accuracy_score(y_test,y_pred_lg)
    acc_lg=round(acc_lg*100,2)
    m.showinfo(title='LG',message="accuracy is"+str(acc_lg)+"%")
def dt():
    global D
    global acc_dt
    from sklearn.tree import DecisionTreeClassifier
    #######create the model
    D=DecisionTreeClassifier()
    #########train the model
    D.fit(x_train,y_train)
    #######test the model
    y_pred_dt=D.predict(x_test)
    #################find the accuracy
    from sklearn.metrics import accuracy_score
    acc_dt=accuracy_score(y_test,y_pred_dt)
    acc_dt=round(acc_dt*100,2)
    m.showinfo(title='DT',message="accuracy is"+str(acc_dt)+"%")
def nb():
    global N
    global acc_nb
    from sklearn.naive_bayes import GaussianNB
    #######create the model
    N=GaussianNB()
    #########train the model
    N.fit(x_train,y_train)
    #######test the model
    y_pred_nb=N.predict(x_test)
    #################find the accuracy
    from sklearn.metrics import accuracy_score
    acc_nb=accuracy_score(y_test,y_pred_nb)
    acc_nb=round(acc_nb*100,2)
    m.showinfo(title='NB',message="accuracy is"+str(acc_nb)+"%")
def compare():
    import matplotlib.pyplot as plt
    model=["KNN","LG","NB","DT"]
    accuracy=[acc_Knn,acc_lg,acc_nb,acc_dt]
    plt.bar(model,accuracy,color=["orange","blue","green","yellow"])
    plt.xlabel("MODELS")
    plt.ylabel("ACCURACY")
    plt.show()
def submit():
    sl=float(v1.get())
    sw=float(v2.get())
    pl=float(v3.get())
    pw=float(v4.get())
    result=K.predict([[sl,sw,pl,pw]])
    if result==0:
        flower="setosa"
    if result==1:
        flower="versicolor"
    else:
        flower="virginica"
    m.showinfo(title="IRIS FLOWER",message=flower)
def reset():
    v1.set("")
    v2.set("")
    v3.set("")
    v4.set("")
##########################################design the gui
from tkinter import *
import tkinter.messagebox as m
w=Tk()
v1=StringVar()
v2=StringVar()
v3=StringVar()
v4=StringVar()
w.title("Comparing ML Algorithms for IRIS Project")
b1=Button(w,text="KNN", font=('arial',15,'bold'),command=knn)
b1.grid(row=1,column=1,columnspan=1)
b2=Button(w,text="LG", font=('arial',15,'bold'),command=lg)
b2.grid(row=2,column=1,columnspan=1)
b3=Button(w,text="DT", font=('arial',15,'bold'),command=dt)
b3.grid(row=3,column=1,columnspan=1)
b4=Button(w,text="NB", font=('arial',15,'bold'),command=nb)
b4.grid(row=4,column=1,columnspan=1)
b5=Button(w,pady=10,text="Compare",font=('arial',15,'bold'),command=compare)
b5.grid(row=6,column=1,columnspan=1)
b5=Button(w,pady=10,text="Submit",font=('arial',15,'bold'),command=submit)
b5.grid(row=6,column=2,columnspan=1)
b5=Button(w,pady=10,text="Reset",font=('arial',15,'bold'),command=reset)
b5.grid(row=6,column=3,columnspan=1)
L1=Label(w,text="Enter the following data",font=('arial',15,'bold'))
L1.grid(row=1,ipadx=10,column=2)
L2=Label(w,text="Sepal Width",font=('arial',15,'bold'))
L2.grid(row=2,ipadx=10,column=2)
L3=Label(w,text="Sepal Length",font=('arial',15,'bold'))
L3.grid(row=3,ipadx=10,column=2)
L4=Label(w,text="Petal Length",font=('arial',15,'bold'))
L4.grid(row=4,ipadx=10,column=2)
L5=Label(w,text="Petal Width",font=('arial',15,'bold'))
L5.grid(row=5,ipadx=10,column=2)
E1=Entry(w,textvariable=v1,font=('arial',15,'bold'))
E1.grid(row=2,column=3)
E1=Entry(w,textvariable=v2,font=('arial',15,'bold'))
E1.grid(row=3,column=3)
E1=Entry(w,textvariable=v3,font=('arial',15,'bold'))
E1.grid(row=4,column=3)
E1=Entry(w,textvariable=v4,font=('arial',15,'bold'))
E1.grid(row=5,column=3)
w.mainloop()




           

















































































