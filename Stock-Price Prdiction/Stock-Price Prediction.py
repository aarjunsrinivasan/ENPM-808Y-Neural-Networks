import numpy as np
with open('Stock Price.txt', 'r') as file:
    data = file.read()
txdata= np.asarray([data])
date_arr = []
price_arr = []
day_arr = []
for i in txdata:
    for cnt in range(0, len(i), 21):
        if i[cnt] != "\t" and i[cnt] != "." and i[cnt] != ",":
            c1 = i[cnt] + i[cnt+1] + i[cnt+2] + i[cnt+3] + i[cnt+5] + i[cnt+6] + i[cnt+8] + i[cnt+9]
            c2 = i[cnt+11] + i[cnt+13] + i[cnt+14] + i[cnt+15]
            date_arr.append(int(c1))
            price_arr.append(int(c2))
count = 1
for date in date_arr:
    day_arr.append(count)
    count += 1

def Preprocess(arr_val, N, T):
    """
    :param N: length of window
    :param T: predict forecast size 
    :return: X_train and Y_train
    """
    x_ar = []
    y_ar = []
    ix=[]
    for i in range(len(arr_val) - (N+T) + 1): # we need 246 - 7 + 1
        x = arr_val[i: i+N]
        y = arr_val[i+(N+T)-1]
        x_ar.append(x)
        ix.append(i+(N+T)-1)
        y_ar.append(y)
    return x_ar, y_ar,ix
def Linear_Forward(x, w, b):
    return np.dot(w, x) + b

def relu(z):
    return np.maximum(0,z)

def mean_squared_error(y_hat, y):
    return np.square(np.subtract(y_hat, y)).mean() 

def front_propogation(x,w1, w2, w3, b1, b2, b3):
    
    z_1 = Linear_Forward(x, w1, b1)
    a_1 = relu(z_1)
    
    z_2 = Linear_Forward(a_1, w2, b2)
    a_2 = relu(z_2)

    z_3 = Linear_Forward(a_2, w3, b3)
    a_3 = relu(z_3)    

    return a_1, a_2, a_3,z_1, z_2, z_3

def diff_relu(z):
    dZ = np.array(z, copy=True)
    dZ[z <= 0] = 0
    dZ[z > 0] = 1
    return dZ

def back_propagation(x, y, w1, w2, w3, b1, b2, b3, a1, a2, a3, z1, z2, z3, alpha):
    m = x.shape[1]
    dz3 = (-2/m) *(y - a3) * diff_relu(z3)
    da2 = np.dot(w3.T, dz3)
    dw3 = (1/m) *np.dot(dz3, a2.T)
    db3 = (1/m) *np.dot(dz3, np.ones([m, 1]))

    dz2 = da2 * diff_relu(z2)
    da1 = np.dot(w2.T, dz2)
    dw2 = (1/m) *np.dot(dz2, a1.T)
    db2 = (1/m) *np.dot(dz2, np.ones([m, 1]))

    dz1 = da1 * diff_relu(z1)
    dx = np.dot(w1.T, dz1)
    dw1 = (1/m) *np.dot(dz1, x.T)
    db1 = (1/m) *np.dot(dz1, np.ones([m ,1]))

    w1 = w1 - alpha* dw1
    w2 = w2 - alpha * dw2
    w3 = w3 - alpha* dw3
    b1 = b1 - alpha* db1
    b2 = b2 - alpha* db2
    b3 = b3 - alpha* db3

    return w1, w2, w3, b1, b2, b3

def model_fit(x, y, N, alpha,batch_size, epochs):
    np.random.seed(0)
    mean_squared_list = []

    # Input layer has N neurons
    # Hidden Layer 1 has 2 neurons
    # Hidden Layer 2 has 2 neurons
    # Output Layer has 1 neuron

    w1 = np.random.randn(2, N)
    b1 = np.zeros((2, 1))
    
    w2 = np.random.randn(2, 2)
    b2 = np.zeros((2, 1))

    w3 = np.random.randn(1, 2)
    b3 = np.zeros((1, 1))

        
    for iterator in range(0, epochs):
        f=0
        for i in range(0,x.shape[0],batch_size):
                     s=i
                     l=i+batch_size
                     if(l>x.shape[0]):
                               l=x.shape[0]
                               f=1
                               
                     xb = (x[s:l]).T
                     yb = (y[s:l]).T
                     a1, a2, a3, z1, z2, z3 = front_propogation(xb, w1, w2, w3, b1, b2, b3)
                     mse = mean_squared_error(a3, yb)
                     mean_squared_list.append(mse)
                     w1, w2, w3, b1, b2, b3 = back_propagation(xb, yb, w1, w2, w3, b1, b2, b3, a1, a2, a3, z1, z2, z3,alpha)

    print("Mean Error after " + str(x.shape[0]) + " iterations is " + str(mse))
    return w1, w2, w3, b1, b2, b3

#Similar to model fit except weights and biases can be passed into function for initialisation
def val_fit(x, y, N, alpha,batch_size, epochs,w1, w2, w3, b1, b2, b3):
    np.random.seed(0)
    mean_squared_list = []      
    for iterator in range(0, epochs):
        f=0
        for i in range(0,x.shape[0],batch_size):
                     s=i
                     l=i+batch_size
                     if(l>x.shape[0]):
                               l=x.shape[0]
                               f=1
                               
                     xb = (x[s:l]).T
                     yb = (y[s:l]).T
                     a1, a2, a3, z1, z2, z3 = front_propogation(xb, w1, w2, w3, b1, b2, b3)
                     mse = mean_squared_error(a3, yb)

                     mean_squared_list.append(mse)
                     w1, w2, w3, b1, b2, b3 = back_propagation(xb, yb, w1, w2, w3, b1, b2, b3, a1, a2, a3, z1, z2, z3,alpha)

    print("Mean Error after " + str(x.shape[0]) + " iterations is " + str(mse))

    return w1, w2, w3, b1, b2, b3


# ## 100 % Test and Train Data

# In[18]:


# 100 test and train
T = 1 # To test 1,10,30,80
N = 4 
x_train, y_train,ix = Preprocess(price_arr, N, T)
x_train=np.array(x_train)/1000
y_train=np.array(y_train)/1000
batch_size=1#Used for training Mini batch Gradient Descent if batch_size=1 Stochastic gradient descent 
alpha=0.001#Learning Rate
print("\n")
print("100% train -test data T=1,N=4")
w1, w2, w3, b1, b2, b3=model_fit(x_train, y_train, N, alpha,batch_size, epochs=1)
a1, a2, a3, z1, z2, z3=front_propogation(x_train.T,w1, w2, w3, b1, b2, b3)
y_pred=np.squeeze(a3*1000)




# In[4]:


T = 10 # To test 1,10,30,80
N = 4
x_train, y_train,ix = Preprocess(price_arr, N, T)
x_train=np.array(x_train)/1000
y_train=np.array(y_train)/1000
batch_size=1#Used for training Mini batch Gradient Descent if batch_size=1 Stochastic gradient descent 
alpha=0.001#Learning Rate
print("\n")
print("100% train -test data T=10,N=4")
w1, w2, w3, b1, b2, b3=model_fit(x_train, y_train, N, alpha,batch_size, epochs=1)
a1, a2, a3, z1, z2, z3=front_propogation(x_train.T,w1, w2, w3, b1, b2, b3)
y_pred=np.squeeze(a3*1000)


# In[6]:


T = 30 # To test 1,10,30,80
N = 4
x_train, y_train,ix = Preprocess(price_arr, N, T)
x_train=np.array(x_train)/1000
y_train=np.array(y_train)/1000
batch_size=1#Used for training Mini batch Gradient Descent if batch_size=1 Stochastic gradient descent 
alpha=0.001#Learning Rate
print("\n")
print("100% train -test data T=30,N=4")
w1, w2, w3, b1, b2, b3=model_fit(x_train, y_train, N, alpha,batch_size, epochs=1)
a1, a2, a3, z1, z2, z3=front_propogation(x_train.T,w1, w2, w3, b1, b2, b3)
y_pred=np.squeeze(a3*1000)




# In[67]:


T = 80 # To test 1,10,30,80
N = 5 
x_train, y_train,ix = Preprocess(price_arr, N, T)
x_train=np.array(x_train)/1000
y_train=np.array(y_train)/1000
batch_size=1#Used for training Mini batch Gradient Descent if batch_size=1 Stochastic gradient descent 
alpha=0.001#Learning Rate
print("\n")
print("100% train -test data T=80,N=5")
w1, w2, w3, b1, b2, b3=model_fit(x_train, y_train, N, alpha,batch_size, epochs=1)
a1, a2, a3, z1, z2, z3=front_propogation(x_train.T,w1, w2, w3, b1, b2, b3)
y_pred=np.squeeze(a3*1000)





# ## 60%-40% Train-Test split

# In[46]:


T = 1 
N = 4 
x_train1, y_train1,ix = Preprocess(price_arr, N, T)
x_train=x_train1[0:np.round((len(price_arr) - (N+T) + 1)*0.6).astype(int)]
y_train=y_train1[0:np.round((len(price_arr) - (N+T) + 1)*0.6).astype(int)]
x_test=x_train1[np.round((len(price_arr) - (N+T) + 1)*0.6).astype(int):]
y_test=y_train1[np.round((len(price_arr) - (N+T) + 1)*0.6).astype(int):]

x_train=np.array(x_train)/1000
y_train=np.array(y_train)/1000
x_test=np.array(x_test)/1000
y_test=np.array(y_test)/1000

batch_size=1#Used for training Mini batch Gradient Descent if batch_size=1 Stochastic gradient descent 
alpha=0.001#Learning Rate
print("\n")
print("60% train -40%test data T=1,N=4")
w1, w2, w3, b1, b2, b3=model_fit(x_train, y_train, N, alpha,batch_size, epochs=1)
a1, a2, a3, z1, z2, z3=front_propogation(x_test.T,w1, w2, w3, b1, b2, b3)
y_pred=np.squeeze(a3*1000)
y_test=y_test*1000




# ## 60 -40 random shuffle

# In[50]:
T = 1 
N = 4 
x_train1, y_train1,ix = Preprocess(price_arr, N, T)
x_train1=np.array(x_train1)
y_train1=np.array(y_train1)
np.random.seed(0)
np.random.shuffle(x_train1)
np.random.shuffle(y_train1)
x_train=x_train1[0:np.round((x_train1.shape[0]*0.6)).astype(int)]
x_test=x_train1[np.round((x_train1.shape[0]*0.6)).astype(int):]
y_train=y_train1[0:np.round((y_train1.shape[0]*0.6)).astype(int)]
y_test=y_train1[np.round((y_train1.shape[0]*0.6)).astype(int):]
x_train=np.array(x_train)/1000
y_train=np.array(y_train)/1000
x_test=np.array(x_test)/1000
y_test=np.array(y_test)/1000
batch_size=1#Used for training Mini batch Gradient Descent if batch_size=1 Stochastic gradient descent 
alpha=0.001#Learning Rate
print("\n")
print("60% train -40%test data Random Shuffle T=1,N=4")
w1, w2, w3, b1, b2, b3=model_fit(x_train, y_train, N, alpha,batch_size, epochs=1)
a1, a2, a3, z1, z2, z3=front_propogation(x_test.T,w1, w2, w3, b1, b2, b3)
y_pred=np.squeeze(a3*1000)
y_test=y_test*1000


# In[51]:




# ## 50%-10%-40% Random Shuffle of Data

# In[55]:


T = 1 
N = 4 
x_train1, y_train1,ix = Preprocess(price_arr, N, T)
x_train1=np.array(x_train1)
y_train1=np.array(y_train1)
np.random.seed(0)
np.random.shuffle(x_train1)
x_train=x_train1[0:np.round((x_train1.shape[0]*0.5)).astype(int)]
x_val=x_train1[np.round((x_train1.shape[0]*0.5)).astype(int):np.round((x_train1.shape[0]*0.6)).astype(int)]
x_test=x_train1[np.round((x_train1.shape[0]*0.6)).astype(int):]
y_train=y_train1[0:np.round((y_train1.shape[0]*0.5)).astype(int)]
y_val=y_train1[np.round((y_train1.shape[0]*0.5)).astype(int):np.round((x_train1.shape[0]*0.6)).astype(int)]
y_test=y_train1[np.round((y_train1.shape[0]*0.6)).astype(int):]
x_train=np.array(x_train)/1000
y_train=np.array(y_train)/1000
x_test=np.array(x_test)/1000
y_test=np.array(y_test)/1000
x_val=np.array(x_val)/1000
y_val=np.array(y_val)/1000
batch_size=1#Used for training Mini batch Gradient Descent if batch_size=1 Stochastic gradient descent 
alpha=0.001#Learning Rate
print("\n")
print("50%-10%-40% Random Shuffle of Data T=1,N=4")
print("training...")
w1, w2, w3, b1, b2, b3=model_fit(x_train, y_train, N, alpha,batch_size, epochs=1)

# In[58]:
print("validation ...")

w1, w2, w3, b1, b2, b3=val_fit(x_val, y_val, N, alpha,batch_size,1,w1, w2, w3, b1, b2, b3)
a1, a2, a3, z1, z2, z3=front_propogation(x_test.T,w1, w2, w3, b1, b2, b3)
y_pred=np.squeeze(a3*1000)
y_test=y_test*1000



