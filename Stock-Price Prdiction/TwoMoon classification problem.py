
# In[1]:
import numpy as np
  
## Parsing Data from Text file
with open('two_moon.txt', 'r') as f:
    lines=f.readlines()

xpos=[]
ypos=[]
clas=[]
for i in range(4,len(lines)):
        j=lines[i].strip().find('\t')
        k=(lines[i].strip())[0]
        for l in range(1,j):
                k+=(lines[i].strip())[l]
        xpos.append(k)
        j2=lines[i].strip().find('\t',j+2)
        q=(lines[i].strip())[j+1]
        for l in range(j+2,j2):
                q+=(lines[i].strip())[l]
        ypos.append(q)
        clas.append((lines[i].strip())[j2+1])


# In[2]:


#Reshaping Data to be used by the Sofmax model
x=np.array(xpos)
x1= x.astype(np.float64)
y=np.array(ypos)
x2= y.astype(np.float64)
c=np.array(clas)
y=c.astype(np.int)
X=np.vstack((x1,x2))
X=X.T
a=np.array([0,1])#if 0
b=np.array([1,0])#if 1
if y[0]==1:
    Y=np.copy(a)
        
if y[0]==0:
    Y=np.copy(b)
for i in range(1,y.shape[0]):
    if y[i]==1:
        Y=np.vstack((Y,a))
        
    if y[i]==0:
        Y=np.vstack((Y,b))



# In[4]:


# Splitting the Data int 50% train,
#                        20% validate
#                        30% test
X_train=X[0:100]
X_val=X[100:140]
X_test=X[140:200]
Y_train=Y[0:100]
Y_val=Y[100:140]
Y_test=Y[140:200]


# In[5]:


def Param_init():
    # initialize parameters randomly
    W = 0.01 * np.random.randn(2,2)
    b = np.zeros((1,2))
    return W,b
def SoftMax(X,Y,W,b):
# # initialize parameters randomly
#  W = 0.01 * np.random.randn(2,2)
#  b = np.zeros((1,2))
# some hyperparameters
 step_size = 1e-0
# gradient descent loop
 num_examples = X.shape[0]
 for i in range(200):
  # evaluate class scores, [N x K]
  scores = np.dot(X, W) + b 
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  lprobs=-np.log(probs)  
  correct_logprobs=np.multiply(Y,lprobs)
  data_loss = np.sum(correct_logprobs)/num_examples
  loss = data_loss 
  if i % 10 == 0:
    print ("iteration %d: loss %f" % (i, loss))  
  # compute the gradient on scores
  dscores = probs
  dscores=np.subtract(dscores,Y)
  dscores /= num_examples  
  # backpropate the gradient to the parameters (W,b)
  dW = np.dot(X.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)  
  # perform a parameter update
  W += -step_size * dW
  b += -step_size * db
 return W,b,probs
def prob_scores(probs):
 for i in probs:
    if (i[0] >0.5):
        i[0]=1
    else:
        i[0]=0
 for i in probs:
    if (i[1] >0.5):
        i[1]=1
    else:
        i[1]=0
 p=(probs==[0,1])
 p=p[:,0]
 q=(probs==[1,0])
 q=q[:,0]
 return probs,p,q
# probs,p,q=prob_scores(probs)
def Accuracy_Softmax(W,b,X,Y):
 scores = np.dot(X, W) + b 
 # compute the class probabilities
 exp_scores = np.exp(scores)
 probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
 prob,p,q=prob_scores(probs)
 acc=Y==prob
 acc=acc[:,0]
 pos=0
 for i in acc:
    if i:
      pos+=1
 accuracy=100*(pos/acc.shape[0])
 print("Accuracy for Softmax Prediction is")
 print(accuracy)

def predict_Softmax(W,b,X,Y):
 scores = np.dot(X, W) + b 
 # compute the class probabilities
 exp_scores = np.exp(scores)
 probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
 prob,p,q=prob_scores(probs)
 return prob,p,q
 
def Conf_Mat(probs,Y):
 a=np.array([0,1])#if 1
 b=np.array([1,0])#if 0
 oy=np.tile(a,(Y.shape[0],1))
 zy=np.tile(b,(Y.shape[0],1))
 yz=Y==zy
 yz=yz[:,0]
 pz=probs==zy
 pz=pz[:,0]
 trneg=pz==yz
 trueneg=0
 falseneg=0
 for i in trneg:
    if i:
        trueneg+=1        
    else:
        falseneg+=1
 yp=Y==oy
 yp=yp[:,0]
 pp=probs==oy
 pp=pp[:,0]
 trpos=pp==yp
 truepos=0
 falsepos=0
 for i in trpos:
    if i:
        truepos+=1        
    else:
        falsepos+=1  
        
 first_row_sum = trueneg + falseneg
 second_row_sum = falseneg + truepos
 first_col_sum = trueneg + falseneg
 second_col_sum = truepos + falsepos
 print("Confusion Matrix : ")
 print("\t    Predicted 0      Predicted 1 \t")
 print("Actual 0 \t " + str(trueneg) + "\t  " + str(falsepos) + "\t  " + str(first_row_sum))
 print("Actual 1 \t " + str(falseneg) + "\t  " + str(truepos) + "\t  " + str(second_row_sum))
 print("\t \t" +str(first_col_sum) + "\t  " + str(second_col_sum))

def plot_pred(x1,x2,q,p):
 plt.plot(x1[q],x2[q], 'ro',label='class 1')
 plt.plot(x1[p],x2[p], 'bo',label='class 0')
 plt.title("Two Moon-Soft Max Prediction Data")
 plt.xlabel('x1')
 plt.ylabel('x2')
 plt.legend()
 plt.show()


# ## Trainging using train dataset

# In[6]:


W,b=Param_init()
Wt,bt,probst=SoftMax(X_train,Y_train,W,b)
print("\n")
print("Training data accuracy")
Accuracy_Softmax(Wt,bt,X_train,Y_train)


# ## Using Validation set for generalising

# In[7]:


Wv,bv,probsv=SoftMax(X_val,Y_val,Wt,bt)
print("\n")
print("Validation Data training accuracy")
Accuracy_Softmax(Wv,bv,X_val,Y_val)


# In[8]:


probsv,p,q=prob_scores(probsv)
Conf_Mat(probsv,Y_val)


# ## Model on Test Data

# In[9]:


Wf,bf,probsf=SoftMax(X_test,Y_test,Wv,bv)
print("\n")
print("Accuracy For Test Data")
Accuracy_Softmax(Wf,bf,X_test,Y_test)


# In[10]:


probsf,p,q=prob_scores(probsf)
Conf_Mat(probsf,Y_test)





# # Logistic Regression

# In[12]:


x=np.array(xpos)
x1= x.astype(np.float64)
y=np.array(ypos)
x2= y.astype(np.float64)
x0=np.ones((x1.shape[0]))
c=np.array(clas)
y=c.astype(np.int)
X=np.vstack((x0,x1,x2))
X=X.T
y=y.reshape(y.shape[0],1)


# In[13]:


# initialize parameters randomly
W = np.random.randn(3,1)
# some hyperparameters
step_size = 1e-0
# gradient descent loop
num_examples = X.shape[0]
for i in range(200):
  # evaluate class scores, [N x K]
  scores = np.dot(X, W) 
  sigmoid=1./(1+np.exp(-scores))
  data_loss = -np.matmul(y[y==1].T,np.log(sigmoid[y==1]))-np.matmul((1-y[y==0]).T,np.log(1-sigmoid[y==0]))
  loss = data_loss/num_examples

  if i % 10 == 0:
    print ("iteration %d: loss %f" % (i, loss))
  
  grad=(1/num_examples)*np.matmul((sigmoid-y).T,X)
  W=W-step_size*(grad.T)


# In[14]:


f_scores = np.dot(X, W) 
pred=1./(1+np.exp(-f_scores))
pred[pred>=0.5]=1
pred[pred<0.5]=0
x1=x1.reshape(200,1)
x2=x2.reshape(200,1)





# In[16]:


acc=y==pred
pos=0
for i in acc:
    if i:
      pos+=1
accuracy=100*(pos/acc.shape[0])
print("Accuracy for Logistic Regression is")
print(accuracy)


# In[17]:


yz=y==0
pz=pred==0
trneg=pz==yz
trueneg=0
falseneg=0
for i in trneg:
    if i:
        trueneg+=1        
    else:
        falseneg+=1
yp=y==1
pp=pred==1
trpos=pp==yp
truepos=0
falsepos=0
for i in trpos:
    if i:
        truepos+=1
        
    else:
        falsepos+=1     
first_row_sum = trueneg + falseneg
second_row_sum = falseneg + truepos
first_col_sum = trueneg + falseneg
second_col_sum = truepos + falsepos
print("Confusion Matrix for Logistic Regresssion is: ")
print("\t    Predicted 0      Predicted 1 \t")
print("Actual 0 \t " + str(trueneg) + "\t  " + str(falsepos) + "\t  " + str(first_row_sum))
print("Actual 1 \t " + str(falseneg) + "\t  " + str(truepos) + "\t  " + str(second_row_sum))
print("\t \t" +str(first_col_sum) + "\t  " + str(second_col_sum))

