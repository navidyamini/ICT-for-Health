import pandas as pd
import numpy as np
from sklearn import tree
import graphviz as viz
#Perform hierarchical/agglomerative clustering.
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


class Read_data(object):
    
    def __init__(self, file_name):
        self.file_name = file_name
        self.features=['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc','htn','dm','cad','appet','pe','ane','class']
   
    def load_file(self):
        data = pd.read_csv(self.file_name,sep=',',skiprows=145,names=self.features,skipinitialspace=True,header=None,na_values=['?','\t?'])
        return data
    
class Pre_process_data(object):
    
    def __init__(self, data):
        self.data = data
        self.key_list=['normal','abnormal','present','notpresent','yes','no','good','poor','ckd','notckd']
        self.key_value=[0,1,0,1,0,1,0,1,1,0]
    
    def data_cleaning(self):
        self.data = self.data.replace({"\tyes": "yes"})
        self.data = self.data.replace({"\tno": "no"})
        self.data = self.data.replace({"ckd\t": "ckd"})
        self.data = self.data.replace(self.key_list,self.key_value)
        new_data = self.data.dropna()
        return new_data
    
    def data_cleaning_replace_nan(self):
        new_data = self.data.fillna(-1)
        return new_data
        
    
if __name__ == "__main__":
    
    np.random.seed(0)
    
    file_name = "chronic_kidney_disease_full.arff"
    read_data = Read_data(file_name)
    
    original_data = read_data.load_file()
    original_data.info()
    print(original_data.describe())
    
    pre_process= Pre_process_data(original_data)
    working_data=pre_process.data_cleaning()
    
    working_data.info()
    print(working_data.describe().T)    
    
    target_feature = working_data.loc[:,'class']
    target_class=['ckd','notckd']
    
    final_data=working_data.drop('class',axis=1)
    
    clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
    clf = clf.fit(final_data, target_feature)
    importance1 = clf.feature_importances_
    plt.figure(figsize=(5, 5))
    plt.stem(importance1)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Importance of features, null values are removed')
    plt.grid()
    
    dot_data = tree.export_graphviz(clf,out_file="Tree.dot",feature_names=read_data.features[0:24],class_names=target_class,filled=True,rounded=True,special_characters=True)
    
    #CREATE RESULTS.pdf FILE WITH TREE SCHEMA IN THE SAME ROOT, RESULTS.pdf 
    dot_data = tree.export_graphviz(clf, out_file=None,feature_names=read_data.features[0:24],class_names=target_class,filled=True,rounded=True,special_characters=True) 
    graph = viz.Source(dot_data) 
    graph.render("Results")
    
    
    working_data_replaced_nan=pre_process.data_cleaning_replace_nan()
    
    working_data_replaced_nan.info()
    print(working_data_replaced_nan.describe().T)    
    
    target_feature = working_data_replaced_nan.loc[:,'class']
    target_class=['ckd','notckd']
    
    final_data=working_data_replaced_nan.drop('class',axis=1)
    
    clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
    clf = clf.fit(final_data, target_feature)
    importance2 = clf.feature_importances_
    plt.figure(figsize=(5, 5))
    plt.stem(importance2)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Importance of features,null values are replaced with -1')
    plt.grid()

    
    dot_data = tree.export_graphviz(clf,out_file="Tree_nan.dot",feature_names=read_data.features[0:24],class_names=target_class,filled=True,rounded=True,special_characters=True)
    
    #CREATE RESULTS.pdf FILE WITH TREE SCHEMA IN THE SAME ROOT, RESULTS.pdf 
    dot_data = tree.export_graphviz(clf, out_file=None,feature_names=read_data.features[0:24],class_names=target_class,filled=True,rounded=True,special_characters=True) 
    graph = viz.Source(dot_data) 
    graph.render("Results_nan")
    

#Use the agglomerative clustering of SciPy on the data of Chronic kidney disease
    
    Z = linkage(working_data,'single') #single,complete,average,weighted,centroid,median,ward

    plt.figure(figsize=(5, 5))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    dendrogram(Z,orientation='top',distance_sort='descending',show_leaf_counts=True)
    plt.show()
    
    
  