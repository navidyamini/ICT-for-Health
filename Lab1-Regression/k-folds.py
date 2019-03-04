import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Read_data(object):
    # this is for reading csv files
    def __init__(self, file_name):
        self.file_name = file_name

    def load_file(self): #reading the file
        data = pd.read_csv(self.file_name)
        data.info()
        print(data.describe())
        return data
        
class Pre_process_data(object):
    # for applying the preprocess that we need on the data
    def __init__(self, data):
        self.data = data
    
    def mean_calculation(self, column1, column2):
        #Prepare a new matrix data with 990 rows
        self.data.test_time = self.data.test_time-self.data.test_time.min() #by subtracting minimum test time from others, test time will start from zero
        self.data.test_time = self.data.test_time.abs().round() #there is some negative value, we need to take the absolute one
        mean = self.data.groupby([column1,column2],as_index=False).mean() #calculating the means of the values stored in the 5-6 blocks of the original file
        return mean
    
    def population_number(self,column):
        #calculating the number of patient
        pop_number = self.data.loc[:,column].max()    
        return pop_number
    
    def normalizer(self,vector_train,vector_test):
        #for normalizing the data
        columns = ['subject#','age','sex','test_time']
        #drop unnecessary columns
        vector_train = vector_train.drop(columns,axis=1)
        vector_test = vector_test.drop(columns,axis=1)
        
        mean_vector = vector_train.mean()
        std_vector = vector_train.std()
        
        norm_vector_train = (vector_train-mean_vector) / std_vector 
        norm_vector_test = (vector_test-mean_vector) / std_vector
        norm_vector_test.index = range(len(norm_vector_test))
        return norm_vector_train, norm_vector_test
    
    def setting_feature_to_estimate(self,vector,F0):
        #creating the y and x vectors
        y = vector.loc[:,F0]
        x = vector.drop(F0,axis=1)
        return y,x

class Plots(object):
    def __init__(self):
        pass
    
    def plotting(self,y_train,y_hat_train,y_test,y_hat_test,w_hat,title,target):

        #calculate the difference for plotting histogram
        self.y_train_diff = y_train - y_hat_train
        self.y_test_diff = y_test - y_hat_test
        
        print(title+ " y_train Avg Error: "+str(np.average(self.y_train_diff)))
        print(title+ " y_test Avg Error: "+str(np.average(self.y_test_diff)))

    def plot_error_set(self,k,train_square_error,test_square_error,title):
        k = np.arange(len(train_square_error))
        ax = plt.subplot(111)
        ax.bar(k, train_square_error.values(), width=0.2, color='b', align='center')
        ax.bar(k-0.2, test_square_error.values(), width=0.2, color='g', align='center')     
        ax.legend(('Train Squared Error','Test Squared Error'))
        #plt.figure(figsize=(5, 5))
        plt.xticks(k, train_square_error.keys())
        plt.title(title+" K-Folds Squered Errors", fontsize=17)
        #plt.xlabel("subsets")
        #plt.ylabel("Squared Error")
        plt.show()
        

class MSE(object):
    #^w = [XT.X]^-1 XT.y
    #e(w) = ||y - X.w|| ^ 2
    #y = Xw + v
    def __init__(self):
        
        self.train_square_error={} 
        self.test_square_error={}

        #self.w = [] #to keep the estimated w
        self.e_train = 0.0 # to keep the train's squared error
        self.e_test = 0.0 # to keep the test's squared error
        
    def train(self,X_train,y_train,X_test,y_test):
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        #Compute the (Moore-Penrose) pseudo-inverse of a matrix.
        #Calculate the generalized inverse of a matrix using its singular-value
        #decomposition (SVD) and including all large singular values.
        pseudo_inverse_x = np.linalg.pinv(self.X_train)

        #Dot product of two arrays.
        #For 2-D arrays it is equivalent to matrix multiplication
        self.w_hat = np.dot(pseudo_inverse_x , self.y_train)

        # numpy.linalg.norm to impliment the norm ||y-X.w||
        self.e_train = np.linalg.norm(self.y_train - np.dot(self.X_train , self.w_hat))**2

        #esstimate the y(n) by using the estimated w
        self.y_hat_train = np.dot(self.X_train, self.w_hat)

        #find the difference between estimated target values and real values in train set
        #self.y_train_diff = self.y_train - self.y_hat_train

    def test(self):
        #using the same w that we estimated in the train step to estimate the new target values in test set
        self.e_test = np.linalg.norm(self.y_test - np.dot(self.X_test, self.w_hat)) ** 2
        self.y_hat_test = np.dot(self.X_test , self. w_hat)
        #self.y_test_diff = self.y_test - self.y_hat_test
        
class Iterative_ga(object):
    
    def __init__(self):

        self.train_square_error={} 
        self.test_square_error={}        
        self.e_train = 0.0 # to keep the train's squared error
        self.e_test = 0.0 # to keep the test's squared error
        
        self.i = 0 #counter
        
        #stopping condition
        self.epsilon = 0.0000001;#?????
        #self.epsilon = 0.00001;#?????
        
        #learning coefficient
        self.gamma = 0.000001;#??????
        
        self.e_each_iteration = []
        self.w=[]
        self.final_w=[]
    
    def count(self):
        self.i +=1
        return

    #ge( ^w(i)) = -2XT.y + 2XT.X.^w(i)
    def gradient_def(self,X,y,w_hat_i):
        X_T = X.T
        gradient = -2*np.dot(X_T,y) + 2*np.dot(np.dot(X_T,X),w_hat_i)
        return gradient   

    def train(self,X_train,y_train,X_test,y_test):
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        np.random.seed(0)
        
        self.w_hat_i = np.zeros(len(self.X_train.columns)) #create zero vector for making the while loop works for the first time for the w(i)      
 
        self.w_hat_i1 = np.random.rand(len(self.X_train.columns)) #random value for W_hat(i+1)
        
        while(np.linalg.norm(self.w_hat_i1 - self.w_hat_i) > self.epsilon):
            
            self.count()#COUNTER OF iteration
            
            self.w += [self.w_hat_i] #i = w_hat(i)
            self.w_hat_i = self.w_hat_i1 #i1 = w_hat(i+1)
            
            gradient = self.gradient_def(self.X_train, self.y_train,self.w_hat_i)            
            
            #^w(i + 1) = ^w(i)  - gamma * ge( ^w(i))
            self.w_hat_i1 = self.w_hat_i - (self.gamma * gradient) #Update the guess
            
            self.e_train = np.linalg.norm(self.y_train - np.dot(self.X_train , self.w_hat_i1))**2
             
            self.e_each_iteration += [self.e_train]#saving the error for each itaration
        
        self.y_hat_train = np.dot(self.X_train, self.w_hat_i1)
        #self.y_train_diff = self.y_train - self.y_hat_train
        self.w_hat = self.w_hat_i1
        
    def test(self):
        self.e_test = np.linalg.norm(self.y_test - np.dot(self.X_test, self.w_hat_i1))**2
        self.y_hat_test = np.dot(self.X_test, self.w_hat_i1)
        #self.y_test_diff = self.y_test - self.y_hat_test
    
class Steepest_Descent(object):
    
    def __init__(self):

        self.train_square_error={} 
        self.test_square_error={}        
        self.e_train = 0.0 # to keep the train's squared error
        self.e_test = 0.0 # to keep the test's squared error
        
        self.i = 0 #counter
        
        #stopping condition
        self.epsilon = 0.0000001;#?????
        #self.epsilon = 0.00001;#?????
        
        self.e_each_iteration = []
        self.w=[]
        self.final_w=[]
        
    def count(self):
        self.i +=1
        return

    #ge( ^w(i)) = -2XT.y + 2XT.X.^w(i)
    def gradient_def(self,X,y,w_hat_i):
        X_T = X.T
        gradient = -2*np.dot(X_T,y) + 2*np.dot(np.dot(X_T,X),w_hat_i)
        return gradient
    
    #H( ^w(i)) = 4 XT.X
    def hessian_matrix(self,X):
        X_T = X.T
        hessian = 4*(np.dot(X_T,X))
        return hessian

    def learning_coefficient (self,hessian,gradient):
        #(||gradient|| ^2) / (gradient.T * Hessian matrix * gradient)
        gradient_T = gradient.T
        gamma = (np.linalg.norm(gradient)**2) / np.dot(np.dot(gradient_T,hessian),gradient)
        return gamma
    
    def train(self,X_train,y_train,X_test,y_test):
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        np.random.seed(0)
        
        self.w_hat_i = np.zeros(len(self.X_train.columns)) #create zero vector for making the while loop works for the first time for the w(i)      
 
        self.w_hat_i1 = np.random.rand(len(self.X_train.columns)) #random value for W_hat(i+1)
        
        #H( ^w(i)) = 4 * (XT * X)
        hessian = self.hessian_matrix(X_train)
        
                #||^w(i + 1) - ^w(i)|| < epsilon
        while(np.linalg.norm(self.w_hat_i1 - self.w_hat_i) > self.epsilon):
            #COUNTER OF ITTERATIONS
            self.count()
            
            #saving the w in each iteration
            self.w += [self.w_hat_i] #i = w_hat(i)
            self.w_hat_i = self.w_hat_i1 #i1 = w_hat(i+1)
            
            gradient = self.gradient_def(self.X_train, self.y_train,self.w_hat_i)
            
            gamma = self.learning_coefficient(hessian,gradient)
            
            self.w_hat_i1 = self.w_hat_i - (gamma * gradient)
            
            self.e_train = np.linalg.norm(self.y_train - np.dot(self.X_train , self.w_hat_i1))**2
             
            self.e_each_iteration += [self.e_train]#saving the error for each itaration
        
        self.y_hat_train = np.dot(self.X_train, self.w_hat_i1)
        #self.y_train_diff = self.y_train - self.y_hat_train
        self.w_hat = self.w_hat_i1
        
    def test(self):
        self.e_test = np.linalg.norm(self.y_test - np.dot(self.X_test, self.w_hat_i1))**2
        self.y_hat_test = np.dot(self.X_test, self.w_hat_i1)
        #self.y_test_diff = self.y_test - self.y_hat_test
        
class Ridge_Regression(object):
    
    def __init__(self):

        self.train_square_error={} 
        self.test_square_error={}
        self.lmbda=0.1 #???
        
        self.e_train = 0.0 # to keep the train's squared error
        self.e_test = 0.0 # to keep the test's squared error
    
    #^w = (XT.X + lmbda I)^-1 .XT. ymeas     
    def train(self,X_train,y_train,X_test,y_test):
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        X_train_T = self.X_train.T
        lambda_identity = self.lmbda*np.identity(len(X_train_T))
        inverse = np.linalg.inv(np.dot(X_train_T, self.X_train)+lambda_identity)
        
        self.w_hat = np.dot(np.dot(inverse, X_train_T), self.y_train)
        self.y_hat_train = np.dot(self.X_train, self.w_hat)
        
        self.e_train = np.linalg.norm(self.y_train - np.dot(self.X_train, self.w_hat))**2
        #self.y_train_diff = self.y_train - self.y_hat_train
    
    def test(self):
        self.e_test = np.linalg.norm(self.y_test - np.dot(self.X_test, self.w_hat)) ** 2
        self.y_hat_test = np.dot(self.X_test , self. w_hat)
        #self.y_test_diff = self.y_test - self.y_hat_test
        
class PCR(object):

    def __init__(self):

        self.train_square_error={} 
        self.test_square_error={}
        self.selected_features=0.0
        
        self.e_train = 0.0 # to keep the train's squared error
        self.e_test = 0.0 # to keep the test's squared error
        
    def train(self,X_train,y_train,X_test,y_test):
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        no_features = float(len(self.X_train))
        #Matrix Rx is the estimation of the covariance matrix of the row random
        #vector and Rx(i; k)
        Rx = (1.0/no_features)*(np.dot(self.X_train.T,self.X_train))
        #U the matrix of eigenvectors of Rx
        #print("Rx ",str(Rx))
        U, U_L = np.linalg.eig(Rx)
        #taking the sum of eigenvalues
        eigvals_sum = U.sum()
        #Return the cumulative sum of the elements along a given axis.
        eigvals_cumsum = U.cumsum()
        #changing this for taking diffrent number of features
        percentage = 0.999 #1 #0.999 #0.99 #.97
        #taking L features, corresponding to the largest L eigenvalues of matrix
        L = len(np.where(eigvals_cumsum < eigvals_sum * percentage)[0])
        self.selected_features=L
        #which is the diagonal matrix with the largest L eigenvalues
        A_L = np.diag(U)[:L-1, :L-1]
        A_L_inv = np.linalg.inv(A_L)
        #which is the matrix with the L eigenvectors corresponding to the largest L eigenvalues
        U_L = U_L[:, :L-1]
        #PCR regression with L features page28
        self.w_L = (1.0/no_features)*(np.dot(np.dot(np.dot(U_L,A_L_inv),U_L.T),np.dot(self.X_train.T,self.y_train)))

        self.y_hat_train = np.dot(self.X_train, self.w_L)
        self.e_train = np.linalg.norm(self.y_train - np.dot(self.X_train, self.w_L))**2
        #self.y_train_diff = self.y_train - self.y_train_result  
    
    def test(self):
        self.e_test = np.linalg.norm(self.y_test - np.dot(self.X_test, self.w_L))**2
        self.y_hat_test = np.dot(self.X_test, self.w_L)
        #self.y_test_diff = self.y_test - self.y_hat_test
    

if __name__ == "__main__":

    k=5 #folds
    
    file_name = "parkinsons_updrs.csv"
    data = Read_data(file_name)     #create an object from Read_data class
    original_data = data.load_file()    #read the file
    plots = Plots()
    pre_process= Pre_process_data(original_data)
    
    #setting the target to predict
    feature_to_estimate = "Jitter(%)"
    #feature_to_estimate = "motor_UPDRS"
    #feature_to_estimate = "total_UPDRS"
    
    column1 = "subject#"
    column2 = "test_time"
    
    prepared_data = pre_process.mean_calculation(column1,column2)
    total_patient = pre_process.population_number(column1)
    
    avg = total_patient / k
    last = 0.0
    
    test=[]
    train=[]
    
    counter = 1
    
    #creating objects
    mse = MSE()
    
    iterative_ga = Iterative_ga()
    
    s_descent = Steepest_Descent()
    
    r_reg = Ridge_Regression()
    
    pcr = PCR()
  
    #k-fold
    while ( last < total_patient):
             
      
        chunk = prepared_data[(prepared_data[column1] > int(last)) & (prepared_data[column1] <= int(last + avg))]
        data_test = chunk
        
        data_train = prepared_data.copy()
        #df = df.drop(df[(df.column < condition) & (df.column > condition)].index)
        data_train = data_train.drop(data_train[(data_train[column1] > int(last)) & (data_train[column1] <= int(last + avg))].index )
        last += avg
        
        test.append(data_test)  #to check the subsets
        train.append(data_train)    #to check the subsets
        
        data_train_norm, data_test_norm = pre_process.normalizer(data_train, data_test) #normalize the data
    
        #creating vector X and Y
        y_train, X_train =pre_process.setting_feature_to_estimate (data_train_norm,feature_to_estimate)
        y_test, X_test = pre_process.setting_feature_to_estimate(data_test_norm,feature_to_estimate)
    

        
        mse.train(X_train,y_train,X_test,y_test)
        mse.test()        
        print("************** "+"Fold "+str(counter)+" ***************************")
        print("MSE Train Set Squared Error " +str(counter)+ " : " + str(mse.e_train))
        print("MSE Test Set Squared Error "+str(counter)+ " : " + str(mse.e_test))
        plots.plotting(y_train,mse.y_hat_train,y_test,mse.y_hat_test,mse.w_hat,"MSE",feature_to_estimate)
        mse.train_square_error["k"+str(counter)] = mse.e_train 
        mse.test_square_error["k"+str(counter)] = mse.e_test
        
        
             
       
        
        iterative_ga.train(X_train,y_train,X_test,y_test)
        iterative_ga.test()   
        print("*****************************************")
        print("Iterative, Gradient Train Set Squared Error " +str(counter)+ " : " + str(iterative_ga.e_train))
        print("Iterative, Gradient Test Set Squared Error "+str(counter)+ " : " + str(iterative_ga.e_test))
        plots.plotting(y_train,iterative_ga.y_hat_train,y_test,iterative_ga.y_hat_test,iterative_ga.w_hat,"Iterative, Gradient",feature_to_estimate)    
        iterative_ga.train_square_error["k"+str(counter)] = iterative_ga.e_train 
        iterative_ga.test_square_error["k"+str(counter)] = iterative_ga.e_test
        
        
        
        s_descent.train(X_train,y_train,X_test,y_test)
        s_descent.test()        
        print("*****************************************")
        print("Steepest Descent Train Set Squared Error " +str(counter)+ " : " + str(s_descent.e_train))
        print("Steepest Descent Test Set Squared Error "+str(counter)+ " : " + str(s_descent.e_test))
        plots.plotting(y_train,s_descent.y_hat_train,y_test,s_descent.y_hat_test,s_descent.w_hat,"Steepest Descent",feature_to_estimate)
        s_descent.train_square_error["k"+str(counter)] = s_descent.e_train 
        s_descent.test_square_error["k"+str(counter)] = s_descent.e_test
        
        
        
        r_reg.train(X_train,y_train,X_test,y_test)
        r_reg.test()
        print("*****************************************")
        print("Ridge Regression Train Set Squared Error " +str(counter)+ " : " + str(r_reg.e_train))
        print("Ridge Regression Test Set Squared Error "+str(counter)+ " : " + str(r_reg.e_test))
        plots.plotting(y_train,r_reg.y_hat_train,y_test,r_reg.y_hat_test,r_reg.w_hat,"Ridge Regression",feature_to_estimate)
        r_reg.train_square_error["k"+str(counter)] = r_reg.e_train 
        r_reg.test_square_error["k"+str(counter)] = r_reg.e_test
        
        
        
        pcr.train(X_train,y_train,X_test,y_test)
        pcr.test()
        print("*****************************************")
        print("PCR Train Set Squared Error " +str(counter)+ " : " + str(pcr.e_train))
        print("PCR Test Set Squared Error "+str(counter)+ " : " + str(pcr.e_test))
        plots.plotting(y_train,pcr.y_hat_train,y_test,pcr.y_hat_test,pcr.w_L,"PCR",feature_to_estimate)
        pcr.train_square_error["k"+str(counter)] = pcr.e_train 
        pcr.test_square_error["k"+str(counter)] = pcr.e_test
        
        
        counter +=1


    plots.plot_error_set(k,mse.train_square_error,mse.test_square_error,"MSE")
    
    plots.plot_error_set(k,iterative_ga.train_square_error,iterative_ga.test_square_error,"Iterative, Gradient")
    
    plots.plot_error_set(k,s_descent.train_square_error,s_descent.test_square_error,"Steepest Descent")
    
    plots.plot_error_set(k,r_reg.train_square_error,r_reg.test_square_error,"Ridge Regression")
    
    plots.plot_error_set(k,pcr.train_square_error,pcr.test_square_error,"PCR")

        