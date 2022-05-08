# %% [markdown]
# # KNN Algorithm for Yeast Data
# 

# %%
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd  
from math import floor, ceil, sqrt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# %%
def the_train_test_split(X, test_ratio = 0.2):
    if(test_ratio >= 1 or test_ratio <0):
        test_ratio = 0.2
    row, _ = X.shape
    train_count = floor(row * (1-test_ratio)) 
    train = X[:train_count]
    test = X[train_count:]
    return train, test

# %%
def euclidean_distance(x,y):
    return sqrt(sum(np.square(x-y)))
 
def manhattan_distance(x,y):
    return sum(abs(x - y))
 
def chebyshev_distance(x,y):
    return max(abs(x - y))
 
def minkowski_distance(x,y, p, w):
    return sum(w * abs(x - y)^p)^(1/p)
 
def get_distance(x, y, algorithm ="euclidean"):
    '''
    three valid metrics:
    euclidean:
        sqrt(sum(np.square(x-y)))
    manhattan:
        return sum(abs(x - y))
        
    chebyshev:
        max(abs(x - y))
 
    '''
    if(algorithm == "euclidean"):
        return euclidean_distance(x,y)
    elif(algorithm == "manhattan"):
        return manhattan_distance(x,y)
    elif(algorithm == "chebyshev"):
        return chebyshev_distance(x,y)
    else:
        #print("The algorithm ", algorithm, " couldn't be recognized.\n", "\"euclidean\" algorithm is used instead")
        return euclidean_distance(x,y)
 

# %%
class K_Neigbours_Classifier():
    def __init__(self, neigbour_count = 7, algorithm = "euclidean"):
        self.alg = algorithm
        self.n_count = neigbour_count
 
    def fit(self, train_input, train_output):
        self.train_in = train_input
        self.train_out = train_output
        #
        pd.unique(self.train_out) # since it is array of arrays sized 1
        self.categories = pd.unique(self.train_out.ravel())
   
    def predict(self, single):
        # calculate the distances
        distances = np.apply_along_axis(get_distance, 1, self.train_in, y=single, algorithm=self.alg)
        #print(distances)
        nearest_indices = np.argpartition(distances, self.n_count)[:self.n_count]
        #print(nearest_indices)
        category_dict = dict.fromkeys(self.categories, 0)
        nearest_keys = self.train_out[nearest_indices]
        for neigbour_key in nearest_keys:
            category_dict[neigbour_key] = 1 + category_dict[neigbour_key]
        the_key_with_max = max(category_dict, key=category_dict.get)
        #print("We predict this one to be: ", the_key_with_max)
        return the_key_with_max

# %%
def measure(X_train, Y_train, X_test, Y_test, neigbour_count=13, algorithm="manhattan", output=True ):
    knc = K_Neigbours_Classifier(neigbour_count=neigbour_count, algorithm=algorithm)
    knc.fit(X_train, Y_train[:,0]) # we know that y_train is 1 dimensional 
    correct_pred = 0
    incorrect_pred = 0
    correct_pred_dict = dict.fromkeys(cat,0)
    failed_to_pred_dict = dict.fromkeys(cat,0)
    assumed_to_pred_dict = dict.fromkeys(cat,0)

    predictions = [] #= np.empty(Y_test.size,  dtype="S3")
    for i in range (Y_test.size):
        correct_key = Y_test[i][0]
        predicted_key =knc.predict(X_test[i])
        predictions.append(predicted_key)
        if(  predicted_key== correct_key):
            correct_pred = 1 + correct_pred
            correct_pred_dict[correct_key] = 1 + correct_pred_dict[correct_key]

        else:
            incorrect_pred = 1 + incorrect_pred
            failed_to_pred_dict[correct_key] = 1 + failed_to_pred_dict[correct_key]
            assumed_to_pred_dict[predicted_key] = 1 + assumed_to_pred_dict[predicted_key] 
    accuracy = correct_pred/(correct_pred + incorrect_pred)
    if output:
        print("Accuracy: ", correct_pred/(correct_pred + incorrect_pred) )
        print("Number of correct predictions: ", correct_pred)
        print("Number of incorrect predictions: ", incorrect_pred)
        print("correct predict(ion) count:\n", correct_pred_dict)
        print("failed_to predict(ion) count:\n", failed_to_pred_dict)
        print("assumed_to predict(ion) count:\n", assumed_to_pred_dict)
        

        print("\n                   Classification Report                  \n",
        classification_report(Y_test, predictions, zero_division=1)) # ignores zero division warning
        
        ConfusionMatrixDisplay.from_predictions(Y_test, predictions)
        plt.show()
    return accuracy

# %% [markdown]
# ## Read the data

# %%
file_name = "yeast.csv" 
md = pd.read_csv(file_name)

# md.dropna(inplace = True)
# md.replace('unknown', 0, inplace = True)
md.head()


# %% [markdown]
# ## Prepare the data
# * Separate the input and output variables
# * Seperate the data into training and test sets
# * Normalize the data
# 

# %%
# Shuffle the data to get more fair representative
md = md.reindex(np.random.permutation(md.index))

test_ratio = 0.2
X = md.values[:,1:9]
Y = md.values[:,9:]
cat = pd.unique(Y[:,0])

# normalize X:
for i in range(X.shape[1]):
    X[:,i] = (X[:,i] - X[:,i].mean())/X[:,i].std()

# %%
X_train, X_test = the_train_test_split(X, test_ratio = test_ratio)
Y_train, Y_test = the_train_test_split(Y, test_ratio = test_ratio)

# %%
measure(X_train, Y_train, X_test, Y_test, neigbour_count=25, algorithm="manhattan")

# %%
measure(X_train, Y_train, X_test, Y_test, neigbour_count=13, algorithm="manhattan")

# %% [markdown]
# ## Cross Validation to tune parameters

# %% [markdown]
# Firstly, we will tune $\lambda$ for each metric,\
# then we will compare the best accuracy rates of metrics (which are in those $lambda$ values).

# %%
def cross_validate_knn(X_train, Y_train, min_lambda= 1, max_lambda=None, algorithm="manhattan", fold_k=10):
    size = ceil( X_train.shape[0]/fold_k)
    if max_lambda is None:
        max_lambda = floor(X_train.shape[0]/fold_k) # better to be sorry than  safe
    if min_lambda % 2 == 0:
        min_lambda = min_lambda + 1
    
    acc = {}
    ind = 0;
    for l in range(min_lambda, 1+max_lambda, 2):
        acc[l] = np.zeros(fold_k)
        for i in range(fold_k):
            x_tr = np.concatenate((X_train[0: i*size],X_train[(1+i)*size:]  ))
            x_te = X_train[i*size:(1+i)*size]
            y_tr = np.concatenate((Y_train[0: i*size],Y_train[(1+i)*size:]  ))
            y_te = Y_train[i*size:(1+i)*size]
            #print(y_te[:10])

            acc[l][i] = measure(x_tr,y_tr,x_te,y_te, neigbour_count=l, algorithm=algorithm, output=False)
        print( "l: ",l, " acc: ",acc[l], "avg ", np.mean(acc[l]))
        ind = ind+1
    return acc

# %% [markdown]
# ### Tuning lambda for Manhattan metric 

# %%
man_cv = cross_validate_knn(X_train, Y_train, min_lambda= 1, max_lambda=35, algorithm="manhattan", fold_k=10)

# %%
man_best_l = -1
man_best_acc = 0
for l, arr in man_cv.items():
    m =np.mean(arr) 
    if  m> man_best_acc:
        man_best_l, man_best_acc = l, m
    print(l, " : ",m)
    
print("Best accuracy was: ", man_best_acc, " when lambda=", man_best_l)
#print(np.argmax(man_cv))

# %%
measure(X_train, Y_train, X_test, Y_test, neigbour_count=man_best_l, algorithm="manhattan")

# %% [markdown]
# ### Tuning lambda for Euclidean metric

# %%
euc_cv = cross_validate_knn(X_train, Y_train, min_lambda= 1, max_lambda=35, algorithm="euclidean", fold_k=10)

# %%
euc_best_l = -1
euc_best_acc = 0
for l, arr in euc_cv.items():
    m =np.mean(arr) 
    if  m> euc_best_acc:
        euc_best_l, euc_best_acc = l, m
    print(l, " : ",m)
    
print("Best accuracy was: ", euc_best_acc, " when lambda=", euc_best_l)
#print(np.argmax(euc_cv))

# %%
measure(X_train, Y_train, X_test, Y_test, neigbour_count=euc_best_l, algorithm="euclidean")

# %% [markdown]
# ### Tuning lambda for Chebyshev metric

# %%
cheb_cv = cross_validate_knn(X_train, Y_train, min_lambda= 1, max_lambda=35, algorithm="chebyshev", fold_k=10)

# %%
che_best_l = -1
che_best_acc = 0
for l, arr in cheb_cv.items():
    m =np.mean(arr) 
    if  m> che_best_acc:
        che_best_l, che_best_acc = l, m
    print(l, " : ",m)
    
print("Best accuracy was: ", che_best_acc, " when lambda=", che_best_l)
print(np.argmax(cheb_cv))

# %%
measure(X_train, Y_train, X_test, Y_test, neigbour_count=che_best_l, algorithm="chebyshev")


