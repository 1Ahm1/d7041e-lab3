
import scipy.io
import numpy as np
# pip install matplotlib
import matplotlib.pyplot as plt
# pip install -U scikit-learn
import sklearn.metrics
import pandas as pd
# pip install seaborn
import seaborn as sn
import itertools


    


n_g=3 # n-gram size
d = 1000
permutation = np.random.permutation(d)

def bind(H1, H2):
    return np.multiply(H1, H2)

def permute(H, r):
    for _ in range(r):
        H = H[permutation]
    return H
    
def compute_hd_vector_n_gram(ind):
    n_gram = N_GRAMS[ind]
    result = np.ones(d)
    for i in range(n_g):
        a_ind = alphabet.index(n_gram[i])
        result = bind(permute(H[a_ind], i + 1), result)
    return result

def normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_array = -1 + 2 * (arr - min_val) / (max_val - min_val)
    return normalized_array


data = scipy.io.loadmat('languages_data.mat')
alphabet=(data['alphabet'])
alphabet=str(alphabet[0]) # turn into string
N_GRAMS=[p for p in itertools.product(alphabet, repeat=n_g)] # get all possible n-grams

# get data from the file
chop_size=int(data['chop_size'])
exm_size=int(data['exm_size'])
langLabels=(data['langLabels'])
testing=(data['testing'])
training=(data['training'])

# collect n-gram statitics for all training data 
H = np.zeros((len(alphabet),d), dtype='float') # HD vectors for the alphabet
for i in range(len(alphabet)):
    H[i] = np.random.choice([-1, 1], size = d) # generate a random HD vector

H_n_gram = np.zeros((len(N_GRAMS),d), dtype='float') # HD vectors for the n_grams
for i in range(len(N_GRAMS)):
    H_n_gram[i] = compute_hd_vector_n_gram(i)

lang_H = np.zeros((langLabels.size,d), dtype='float') 
for i in range(langLabels.size):
    buf_chop=training[0][i] #pick the current portion
    buf_chop=np.array2string(buf_chop) # get to string    
    buf_chop=(buf_chop[3:len(buf_chop)-2]) # get rid of artefacts like []
    
    lang_H[i] = np.zeros(d)
    for jj in range(len(buf_chop)-(n_g-1)):        
        ngc = buf_chop [jj:(jj+(n_g))] #pick current n-gram
        ngc1 = tuple(ngc)
        ind_ngc = N_GRAMS.index(ngc1) # find index in  N_GRAMS
        lang_H[i] += H_n_gram[ind_ngc] # update the corresponding HD vector
    lang_H[i] = normalize(lang_H[i])


# collect n-gram statitics for all testing data 

GT = [None] * langLabels.size * exm_size   #ground truth  
PR_encode = [None] * langLabels.size * exm_size   #predicton using encoding
for i in range(langLabels.size):    
    for j in range(exm_size): 
        buf_chop=testing[j][i] #pick the current portion
        buf_chop=np.array2string(buf_chop) # get to string    
        buf_chop=buf_chop[3:len(buf_chop)-2] # get rid of artefacts like []
        
        test_HD_vec = np.zeros(d)
        for jj in range(len(buf_chop)-(n_g-1)): 
            ngc=buf_chop [jj:(jj+(n_g))] #pick current n-gram
            ngc= tuple(ngc)
            ind_ngc=N_GRAMS.index(ngc) # find index in  N_GRAMS            
            test_HD_vec += H_n_gram[ind_ngc]
        test_HD_vec = normalize(test_HD_vec)
        
        lang_ind = -1
        max_wrt = 0
        for l in range(len(lang_H)):
            h= lang_H[l]
            wrt = abs(np.dot(h, test_HD_vec) / (np.linalg.norm(h)*np.linalg.norm(test_HD_vec)))
            if wrt > max_wrt:
                max_wrt = wrt
                lang_ind = l

        PR_encode[ i*exm_size +j ]=np.array2string(langLabels[0][lang_ind])
        GT[ i*exm_size +j ]=np.array2string(langLabels[0][i]) # add ground truth

# accuracy 

acc=0.0
for i in range(len(GT)):
   acc+=(PR_encode[i]==GT[i])

acc=acc/len(GT) 
print("Accuracy encoding: ", acc)


langLabels[0][1][0].tolist()
Labels_arr=np.empty(21).astype(str)
for i in range(21):
    Labels_arr[i]=langLabels[0][i][0]
    
Labels_arr

conf_mat= sklearn.metrics.confusion_matrix(GT, PR_encode, labels=None, sample_weight=None)

#plot confusion matrix
df_cm = pd.DataFrame(conf_mat, index = [i for i in Labels_arr], columns = [j for j in Labels_arr])
plt.figure('Confusion Matrix - Encoding', figsize = (8,8))
sn.heatmap(df_cm, annot=True)
plt.show(block=True)