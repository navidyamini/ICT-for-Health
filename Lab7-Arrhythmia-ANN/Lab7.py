import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn import svm

class Read_data(object):
    
    def __init__(self, file_name):
        self.file_name = file_name
        

    def load_file(self):
        data = pd.read_csv(self.file_name,header=None,na_values=['?','\t?'])
        return data
    
class Binary_Neural_Network(object):
    
    def __init__(self,y,class_id):
        self.y=y
        self.class_id=class_id

        
    def specificity_sensitivity(self,estimated,real):
        
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for i in range(len(real+1)):
            if (estimated[i] == 1 and real[i] == 1):
                TP = TP + 1
            elif (estimated[i] == 0 and real[i] == 0):
                TN = TN + 1
            elif (estimated[i] == 1 and real[i] == 0):
                FP = FP + 1
            elif (estimated[i] == 0 and real[i] == 1):
                FN = FN + 1
        
        print("True positive (TP): ",TP)
        print("True negative (TN): ",TN)
        print("False positive (FP): ",FP)
        print("False negative (FN): ",FN)
        #sensitivity, recall, hit rate, or true positive rate (TPR)     
        #TPR=TP/P=TP/(TP+FN)=1-FNR
        TPR=TP/(TP+FN)
        print("True positive rate (TPR): ",TPR)
        #specificity, selectivity or true negative rate (TNR)
        #TNR=TN/N=TN/(TN+FP)=1-FPR
        TNR=TN/(TN+FP)
        print("True negative rate (TNR): ",TNR)
        #precision or positive predictive value (PPV)
        PPV=TP/(TP+FP)
        print("Positive predictive value (PPV): ",PPV)
        #negative predictive value (NPV)
        NPV=TN/(TN+FN)
        print("Negative predictive value (NPV): ",NPV)
        #miss rate or false negative rate (FNR)
        #FNR=FN/P=FN/(FN+TP)=1-TPR
        FNR=FN/(FN+TP)
        print("False negative rate (FNR): ",FNR)
        #fall-out or false positive rate (FPR)
        #FPR=FP/N=FP/(FP+TN)=1-TNR
        FPR=FP/(FP+TN)
        print("False positive rate (FPR): ",FPR)
        #false discovery rate (FDR)
        #FDR=FP/(FP+TP)=1-PPV
        FDR=FP/(FP+TP)
        print("False discovery rate (FDR): ",FDR)
        #false omission rate (FOR)
        #FOR=FN/(FN+TN)=1-NPV
        FOR=FN/(FN+TN)
        print("False omission rate (FOR): ",FOR)
        #accuracy (ACC)
        #ACC=(TP+TN)/(P+N)=(TP+TN)/(TP+TN+FP+FN)
        ACC=(TP+TN)/(TP+TN+FP+FN)
        print("Accuracy (ACC): ",ACC)
        
    def run(self):
        # Neural Network
        tf.set_random_seed(1234)
    
        NPatient = len(self.y)
        NFeature = self.y.shape[1]
    
        input_nodes = NFeature
        hidden_layer_node = math.floor(NFeature/2)
    
    
        #Being N the total number of patients, use N/2 patients for training
#       and the remaining patients for testing.
    
        data_train = (self.y[0:int(NPatient/2)]).as_matrix()
        data_test = (self.y[int(NPatient/2):NPatient]).as_matrix()
    
        #data_train = data_train.as_matrix()
    
        NRows = len(data_train)
    
        class_id_train = class_id[0:int(NPatient/2)].values.reshape((226, 1))
        class_id_test = class_id [int(NPatient/2):NPatient].values.reshape((226, 1))
        
        x=tf.placeholder(tf.float32,[NRows,input_nodes]) #input
        t=tf.placeholder(tf.float32,[NRows,1]) #desired output
    
        #neural netw structure:
    
        #with F input nodes, F/2 nodes in the hidden layer, one output node
    
        # matrix W(1) has F rows (as many as the features/columns of X) and
        # M1 columns (as many as the nodes in layer 1);
        w1=tf.Variable(tf.random_normal(shape=[input_nodes,
                    hidden_layer_node], mean=0.0, stddev=1.0, dtype=tf.float32))
    
        # matrix B(1) has N rows (all equal, as many as the patients/rows in
        # matrix (X) and M1 columns (as many as the nodes in layer 1)
        b1=tf.Variable(tf.random_normal(shape=[1,hidden_layer_node],
                    mean=0.0, stddev=1.0, dtype=tf.float32))
    
        a1=tf.matmul(x,w1)+b1
        z1=tf.nn.sigmoid(a1)
    
        # matrixW(2) has M1 rows (as many as the nodes in layer 1) and M2
        # columns (as many as the nodes in layer 2);
        w2=tf.Variable(tf.random_normal(shape=[hidden_layer_node,1], 
                                        mean=0.0, stddev=1.0, dtype=tf.float32))
        b2=tf.Variable(tf.random_normal(shape=[1,1], mean=0.0,
                                        stddev=1.0, dtype=tf.float32))
    
        # neural network output
        #a2=tf.matmul(z1,w2)+b2
        #y=tf.nn.sigmoid(a2)
        y2 = tf.nn.sigmoid(tf.matmul(z1, w2) + b2)

        cost=tf.nn.sigmoid_cross_entropy_with_logits(labels = t, logits = y2) #objective function
        optim=tf.train.GradientDescentOptimizer(0.0005005) # use gradient descent in the training phase
        optim_op = optim.minimize(cost, var_list=[w1,b1,w2,b2]) # minimize the objective function changing w1,b1,w2,b2
    
        #--- define the session
        sess = tf.Session()
    
        #--- initialize
        tf.global_variables_initializer().run(session = sess)
        tf.local_variables_initializer().run(session = sess)
    
        #--- run the learning machine
        for i in range(200000):
            # train
            train_data={x: data_train, t: class_id_train}
        
            sess.run([optim_op], feed_dict=train_data) #to delete
        
            if i % 500 == 0:# print the intermediate result
                print(i,np.sum(cost.eval(feed_dict=train_data,session=sess)))
    
  
        y_hat_train = y2.eval(feed_dict={x:data_train, t: class_id_train}, session = sess)
        y_hat_train = np.round(y_hat_train)

        y_hat_test = y2.eval(feed_dict={x:data_test, t: class_id_test}, session = sess)
        y_hat_test = np.round(y_hat_test)
            
        print("\n")
        print("********************* Binary Neural Network *********************")
        print("---Train Set---")
        self.specificity_sensitivity(y_hat_train,class_id_train)
            
        print("\n")
        print("---Test Set---")
        self.specificity_sensitivity(y_hat_test,class_id_test)

class Non_Binary_Neural_Network(object):

    def __init__(self,y,class_id):
        self.y = y
        self.class_id = class_id


    def Confusion_Matrix(self,data,title,size):
        sumation = np.sum(data, 1)
        normalized_cm = np.zeros([size, size])

        for i in range(size):
            for j in range(size):
                if (data[i, j]) != 0:
                    normalized_cm[i, j] = data[i, j] / sumation[i]

        plt.clf()
        plt.imshow(normalized_cm, interpolation='nearest', cmap=plt.cm.Blues)
        classNames = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.colorbar(use_gridspec=True)  # use_gridspec = True

        tick_marks = np.arange(len(classNames))
        plt.xticks(tick_marks, classNames)
        plt.yticks(tick_marks, classNames)
        thresh = normalized_cm.max() / 2.
        for i in range(len(classNames)):
            for j in range(len(classNames)):
                plt.text(j, i, format(normalized_cm[i, j], '.2f'),
                         horizontalalignment="center",
                         color="white" if normalized_cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.show()

    def run(self):

        NPatient = len(self.y)
        NFeature = self.y.shape[1]

        input_nodes = NFeature
        hidden_layer_node = math.floor(NFeature / 2)

        max_id = np.amax(self.class_id)

        # 2d_matrix_*16
        matrix = np.zeros((NPatient, max_id))

        for i in range(NPatient):
            matrix[i][self.class_id[i] - 1] = 1

        data_train = (y[0:int(NPatient / 2)]).as_matrix()
        data_test = (y[int(NPatient / 2):NPatient]).as_matrix()

        class_id_train = matrix[0:int(NPatient / 2)]
        class_id_test = matrix[int(NPatient / 2):NPatient]

        NRows = len(data_train)

        # Neural Network
        tf.set_random_seed(1234)

        x = tf.placeholder(tf.float32, [NRows, input_nodes])  # input
        t = tf.placeholder(tf.float32, [NRows, max_id])  # desired output

        # tf.random_normal or tf.truncated_normal
        w1 = tf.Variable(tf.truncated_normal(shape=[input_nodes,
                                                    hidden_layer_node], mean=0.0, stddev=1.0, dtype=tf.float32))

        b1 = tf.Variable(tf.truncated_normal(shape=[1, hidden_layer_node],
                                             mean=0.0, stddev=1.0, dtype=tf.float32))

        a1 = tf.matmul(x, w1) + b1
        z1 = tf.nn.sigmoid(a1)

        # # second layer
        w2 = tf.Variable(
            tf.truncated_normal(shape=[hidden_layer_node, hidden_layer_node], mean=0.0, stddev=1.0, dtype=tf.float32,
                                name="weights2"))
        b2 = tf.Variable(
            tf.truncated_normal(shape=[1, hidden_layer_node], mean=0.0, stddev=1.0, dtype=tf.float32, name="biases2"))
        a2 = tf.matmul(z1, w2) + b2
        z2 = tf.nn.sigmoid(a2)

        w3 = tf.Variable(tf.truncated_normal(shape=[hidden_layer_node, max_id],
                                             mean=0.0, stddev=1.0, dtype=tf.float32))
        b3 = tf.Variable(tf.truncated_normal(shape=[1, max_id], mean=0.0,
                                             stddev=1.0, dtype=tf.float32))

        # neural network output
        # a2=tf.matmul(z1,w2)+b2
        # y=tf.nn.sigmoid(a2)
        y3 = tf.nn.softmax(tf.matmul(z2, w3) + b3)

        cost = tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=y3)  # objective function
        optim = tf.train.GradientDescentOptimizer(0.0005)  # use gradient descent in the training phase
        optim_op = optim.minimize(cost, var_list=[w1, b1, w2, b2, w3,
                                                  b3])  # minimize the objective function changing w1,b1,w2,b2

        # --- define the session
        sess = tf.Session()

        # --- initialize
        tf.global_variables_initializer().run(session=sess)
        tf.local_variables_initializer().run(session=sess)

        # --- run the learning machine
        for i in range(700000):
            # train
            train_data = {x: data_train, t: class_id_train}

            sess.run([optim_op], feed_dict=train_data)  # to delete

            if i % 1000 == 0:  # print the intermediate result
                print(i, np.sum(cost.eval(feed_dict=train_data, session=sess)))

        label = tf.argmax(t, axis=1)
        prediction = tf.argmax(tf.round(y3), axis=1)

        cm = tf.contrib.metrics.confusion_matrix(labels=label, predictions=prediction, num_classes=max_id,
                                                 dtype=tf.float32)

        result_train = cm.eval(feed_dict=train_data, session=sess)
        result_test = cm.eval(feed_dict={x:data_test, t: class_id_test}, session = sess)

        self.Confusion_Matrix(result_train,'Train Set Confusion Matrix (Neural_Network)',max_id)
        self.Confusion_Matrix(result_test,'Test Set Confusion Matrix (Neural_Network)',max_id)
        
class SVM_Class(object):
    def __init__(self,original_data):
        #Remove columns all zero
        self.data = original_data.loc[:, (original_data != 0).any(axis=0)]
    
        #Drop columns which contain missing value
        self.data = data.dropna(axis=1)

        # step1
        # class 0 for healthy patients and class 1 for patients with arrhythmia
        self.data[279] = np.where(data[279] < 2, -1, 1)

        # step2
        # Define vector class id as the last column of matrix arrhythmia, define
        # matrix y as the other columns
        self.class_id = data.loc[:, 279]
        self.y = self.data.drop(279, axis=1)

        
    def largest_eigenvalues(self,eigenvalues):
        total = 0
        F = 0
        for i in range (len(eigenvalues)):
            total = total + eigenvalues[i]
            if total >= 0.9995 * sum(abs( eigenvalues)):
                F = i
                break
        return F
    
    def specificity_sensitivity(self,estimated,real):
        
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for i in range(len(real+1)):
            if (estimated[i] == 1 and real[i] == 1):
                TP = TP + 1
            elif (estimated[i] == 0 and real[i] == 0):
                TN = TN + 1
            elif (estimated[i] == 1 and real[i] == 0):
                FP = FP + 1
            elif (estimated[i] == 0 and real[i] == 1):
                FN = FN + 1
        
        print("True positive (TP): ",TP)
        print("True negative (TN): ",TN)
        print("False positive (FP): ",FP)
        print("False negative (FN): ",FN)
        #sensitivity, recall, hit rate, or true positive rate (TPR)     
        #TPR=TP/P=TP/(TP+FN)=1-FNR
        TPR=TP/(TP+FN)
        print("True positive rate (TPR): ",TPR)
        #specificity, selectivity or true negative rate (TNR)
        #TNR=TN/N=TN/(TN+FP)=1-FPR
        TNR=TN/(TN+FP)
        print("True negative rate (TNR): ",TNR)
        #precision or positive predictive value (PPV)
        PPV=TP/(TP+FP)
        print("Positive predictive value (PPV): ",PPV)
        #negative predictive value (NPV)
        NPV=TN/(TN+FN)
        print("Negative predictive value (NPV): ",NPV)
        #miss rate or false negative rate (FNR)
        #FNR=FN/P=FN/(FN+TP)=1-TPR
        FNR=FN/(FN+TP)
        print("False negative rate (FNR): ",FNR)
        #fall-out or false positive rate (FPR)
        #FPR=FP/N=FP/(FP+TN)=1-TNR
        FPR=FP/(FP+TN)
        print("False positive rate (FPR): ",FPR)
        #false discovery rate (FDR)
        #FDR=FP/(FP+TP)=1-PPV
        FDR=FP/(FP+TP)
        print("False discovery rate (FDR): ",FDR)
        #false omission rate (FOR)
        #FOR=FN/(FN+TN)=1-NPV
        FOR=FN/(FN+TN)
        print("False omission rate (FOR): ",FOR)
        #accuracy (ACC)
        #ACC=(TP+TN)/(P+N)=(TP+TN)/(TP+TN+FP+FN)
        ACC=(TP+TN)/(TP+TN+FP+FN)
        print("Accuracy (ACC): ",ACC)
        
    def run(self):
        R = np.cov(self.y.T)
        eigenvalues,eigenvector = np.linalg.eigh(R)
        lamda = np.eye(np.size(eigenvalues))*eigenvalues
        R_1 = np.dot(np.dot(eigenvector,lamda),eigenvector.T)
        
        F = self.largest_eigenvalues(eigenvalues)
        UF = eigenvector[: , : F]
        z = np.dot(y, UF)
        
        NPatient = len(z)
        data_train = (z[0:int(NPatient/2)])
        data_test = (z[int(NPatient/2):NPatient])
        
        class_id_train = (self.class_id[0:int(NPatient / 2)]).as_matrix()
        class_id_test = (self.class_id[int(NPatient / 2):NPatient]).as_matrix()
        
        clf = svm.SVC(C=0.0005, kernel = 'linear')
        clf.fit(data_train, class_id_train)
        
        y_hat_train = clf.predict(data_train)
        y_hat_test = clf.predict(data_test)
    
        print("\n")
        print("********************* SVM *********************")
        print("---Train Set---")
        self.specificity_sensitivity(y_hat_train,class_id_train)
        
        print("\n")
        print("---Test Set---")
        self.specificity_sensitivity(y_hat_test,class_id_test)

if __name__ == "__main__":
    
    file_name = "arrhythmia.data.csv"
    read_data = Read_data(file_name)
    original_data = read_data.load_file()
    original_data.info()
    print(original_data.describe())
    np.random.seed(1234)
    #Remove columns all zero
    data = original_data.loc[:, (original_data != 0).any(axis=0)]
    
    #Drop columns which contain missing value
    data = data.dropna(axis=1)

    # step1
    # class 0 for healthy patients and class 1 for patients with arrhythmia
    class_id_total = data.loc[:, 279]
    data[279] = np.where(data[279] < 2, 0, 1)

    # step2
    # Define vector class id as the last column of matrix arrhythmia, define
    # matrix y as the other columns
    class_id = data.loc[:, 279]
    y = data.drop(279, axis=1)
    
    #bnn = Binary_Neural_Network(y,class_id)
    #bnn.run()

    #nbnn = Non_Binary_Neural_Network(y,class_id_total)
    #nbnn.run()
    
    svmc =SVM_Class(original_data)
    svmc.run()