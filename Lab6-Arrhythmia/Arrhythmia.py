import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Read_data(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def load_file(self):
        data = pd.read_csv(self.file_name, header=None, na_values=['?', '\t?'])
        data.info()
        print(data.describe())
        return data


class Pre_Process(object):
    def __init__(self, data):
        self.data = data

    def cleaning(self):
        # Remove columns all zero
        self.data = self.data.loc[:, (self.data != 0).any(axis=0)]
        # Drop columns which contain missing value
        self.data = self.data.dropna(axis=1)
        # Change in the matrix last column values larger than 1 with value 2
        self.data[279] = np.where(self.data[279] > 1, 2, 1)
        return self.data


class Minimum_Distance_Criterion(object):
    def __init__(self, x1, x2, y, class_id):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.class_id = class_id

    def run(self):
        # Rj ={u:||u - Xj||^2 <= ||u - Xk||^2, k!=j}

        y_hat = []
        for i in range(len(y)):
            d1 = np.linalg.norm(self.y.iloc[i, :] - self.x1) ** 2
            d2 = np.linalg.norm(self.y.iloc[i, :] - self.x2) ** 2
            if d1 <= d2:
                y_hat.append(1)
            else:
                y_hat.append(2)
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for i in range(len(y_hat)):
            if (y_hat[i] == 2 and self.class_id[i] == 2):
                TP = TP + 1
            elif (y_hat[i] == 1 and self.class_id[i] == 1):
                TN = TN + 1
            elif (y_hat[i] == 2 and self.class_id[i] == 1):
                FP = FP + 1
            elif (y_hat[i] == 1 and self.class_id[i] == 2):
                FN = FN + 1

        print("\n")
        print("********************* Minimum Distance Criterion *********************")
        print("True positive (TP): ", TP)
        print("True negative (TN): ", TN)
        print("False positive (FP): ", FP)
        print("False negative (FN): ", FN)
        # sensitivity, recall, hit rate, or true positive rate (TPR)
        # TPR=TP/P=TP/(TP+FN)=1-FNR
        TPR = TP / (TP + FN)
        print("True positive rate (TPR): ", TPR)
        # specificity, selectivity or true negative rate (TNR)
        # TNR=TN/N=TN/(TN+FP)=1-FPR
        TNR = TN / (TN + FP)
        print("True negative rate (TNR): ", TNR)
        # precision or positive predictive value (PPV)
        PPV = TP / (TP + FP)
        print("Positive predictive value (PPV): ", PPV)
        # negative predictive value (NPV)
        NPV = TN / (TN + FN)
        print("Negative predictive value (NPV): ", NPV)
        # miss rate or false negative rate (FNR)
        # FNR=FN/P=FN/(FN+TP)=1-TPR
        FNR = FN / (FN + TP)
        print("False negative rate (FNR): ", FNR)
        # fall-out or false positive rate (FPR)
        # FPR=FP/N=FP/(FP+TN)=1-TNR
        FPR = FP / (FP + TN)
        print("False positive rate (FPR): ", FPR)
        # false discovery rate (FDR)
        # FDR=FP/(FP+TP)=1-PPV
        FDR = FP / (FP + TP)
        print("False discovery rate (FDR): ", FDR)
        # false omission rate (FOR)
        # FOR=FN/(FN+TN)=1-NPV
        FOR = FN / (FN + TN)
        print("False omission rate (FOR): ", FOR)
        # accuracy (ACC)
        # ACC=(TP+TN)/(P+N)=(TP+TN)/(TP+TN+FP+FN)
        ACC = (TP + TN) / (TP + TN + FP + FN)
        print("Accuracy (ACC): ", ACC)


class Bayes_Criterion(object):
    def __init__(self, y1, y2, y, class_id):
        self.y1 = y1
        self.y2 = y2
        self.y = y
        self.class_id = class_id

    def largest_eigenvalues(self, eigenvalues):
        total = 0
        F = 0
        for i in range(len(eigenvalues)):
            total = total + eigenvalues[i]
            if total >= 0.99995 * sum(abs(eigenvalues)):
                F = i
                break
        return F

    def PDF(self, P, F, R, S, W, y):
        part_1 = P / (np.sqrt((pow(2 * np.pi, F)) * np.linalg.det(R)))
        p = []
        for i in range(y):
            part_2 = (-0.5) * np.dot(np.dot((S[i, :] - W).T, np.linalg.inv(R)), (S[i, :] - W))
            p.append(part_1 * np.exp(part_2))
        return p

    def run(self):
        # step8

        P_1 = np.size(self.y1) / np.size(self.y)
        P_2 = np.size(self.y2) / np.size(self.y)

        # step 9 & 10
        R_1 = np.cov(self.y1.T)
        R_2 = np.cov(self.y2.T)

        eigenvalues_1, eigenvector_1 = np.linalg.eig(R_1)
        eigenvalues_2, eigenvector_2 = np.linalg.eig(R_2)

        lamda_1 = np.eye(np.size(eigenvalues_1)) * eigenvalues_1
        lamda_2 = np.eye(np.size(eigenvalues_2)) * eigenvalues_2

        R_11 = np.dot(np.dot(eigenvector_1, lamda_1), eigenvector_1.T)
        R_22 = np.dot(np.dot(eigenvector_2, lamda_2), eigenvector_2.T)

        # the eigenvectors with the lowest eigenvalues bear the least information
        # about the distribution of the data,
        # and those are the ones we want to drop.

        # step 12 getting UF1 and UF2

        F_1 = self.largest_eigenvalues(eigenvalues_1)
        F_2 = self.largest_eigenvalues(eigenvalues_2)

        UF1 = eigenvector_1[:, : F_1]
        UF2 = eigenvector_2[:, : F_2]

        # step 13
        # project y1 (F columns) onto UF1 and get the new matrix z1
        # project y2 (F columns) onto UF2 and get the new matrix z2
        z1 = np.dot(self.y1, UF1)
        z2 = np.dot(self.y2, UF2)

        z1_cov = np.cov(z1.T)
        z2_cov = np.cov(z2.T)

        # step 14
        # find the means of z1 and z2  and call them w1 and w2
        w1 = np.mean(z1, axis=0)
        w2 = np.mean(z2, axis=0)

        # step 16
        # project matrix y onto UF1 to get matrix s1 and onto UF2 to get matrix s2
        s1 = np.dot(self.y, UF1)
        s2 = np.dot(self.y, UF2)

        P1 = self.PDF(P_1, F_1, z1_cov, s1, w1, len(self.y))
        P2 = self.PDF(P_2, F_2, z2_cov, s2, w2, len(self.y))

        y_hat = []

        for i in range(len(self.y)):

            if P1[i] >= P2[i]:
                y_hat.append(1)
            else:
                y_hat.append(2)

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for i in range(len(y_hat)):
            if (y_hat[i] == 2 and self.class_id[i] == 2):
                TP = TP + 1
            elif (y_hat[i] == 1 and self.class_id[i] == 1):
                TN = TN + 1
            elif (y_hat[i] == 2 and self.class_id[i] == 1):
                FP = FP + 1
            elif (y_hat[i] == 1 and self.class_id[i] == 2):
                FN = FN + 1
        print("\n")
        print("********************* Bayes Criterion *********************")
        print("True positive (TP): ", TP)
        print("True negative (TN): ", TN)
        print("False positive (FP): ", FP)
        print("False negative (FN): ", FN)
        # sensitivity, recall, hit rate, or true positive rate (TPR)
        # TPR=TP/P=TP/(TP+FN)=1-FNR
        TPR = TP / (TP + FN)
        print("True positive rate (TPR): ", TPR)
        # specificity, selectivity or true negative rate (TNR)
        # TNR=TN/N=TN/(TN+FP)=1-FPR
        TNR = TN / (TN + FP)
        print("True negative rate (TNR): ", TNR)
        # precision or positive predictive value (PPV)
        PPV = TP / (TP + FP)
        print("Positive predictive value (PPV): ", PPV)
        # negative predictive value (NPV)
        NPV = TN / (TN + FN)
        print("Negative predictive value (NPV): ", NPV)
        # miss rate or false negative rate (FNR)
        # FNR=FN/P=FN/(FN+TP)=1-TPR
        FNR = FN / (FN + TP)
        print("False negative rate (FNR): ", FNR)
        # fall-out or false positive rate (FPR)
        # FPR=FP/N=FP/(FP+TN)=1-TNR
        FPR = FP / (FP + TN)
        print("False positive rate (FPR): ", FPR)
        # false discovery rate (FDR)
        # FDR=FP/(FP+TP)=1-PPV
        FDR = FP / (FP + TP)
        print("False discovery rate (FDR): ", FDR)
        # false omission rate (FOR)
        # FOR=FN/(FN+TN)=1-NPV
        FOR = FN / (FN + TN)
        print("False omission rate (FOR): ", FOR)
        # accuracy (ACC)
        # ACC=(TP+TN)/(P+N)=(TP+TN)/(TP+TN+FP+FN)
        ACC = (TP + TN) / (TP + TN + FP + FN)
        print("Accuracy (ACC): ", ACC)


class Minimum_Distance_Criterion_Non_Binary(object):
    def __init__(self, data, y):

        self.data = data.loc[:, (original_data != 0).any(axis=0)]

        # Drop columns which contain missing value
        self.data = self.data.dropna(axis=1)

        # Define vector class id as the last column of matrix arrhythmia, define
        # matrix y as the other columns
        self.class_id = self.data.loc[:, 279]

        self.y = y

    def run(self):
        patient_class = []
        for i in range(16):
            patient_class.append(self.class_id == i + 1)

        patient_data = []
        for i in range(16):
            patient_data.append(self.y[patient_class[i]])

        x_mean = []
        for i in range(16):
            x_mean.append(np.mean(patient_data[i], axis=0))

        for i in range(10, 13):
            x_mean[i] = np.inf

        distances = np.zeros([16])
        # y_hat3 = np.empty(y.shape[0])
        y_hat = []
        for i in range(len(self.y)):
            for j in range(16):
                distances[j] = (pow(np.linalg.norm(self.y.iloc[i, :] - x_mean[j], ord=2), 2))
            # y_hat3[i] = np.argmin(distances)+1
            y_hat.append(np.argmin(distances) + 1)
            # Confusion Matrix
        cm = np.zeros([16, 16])
        for b in range(len(self.y)):
            cm[int(self.class_id[b]) - 1, int(y_hat[b]) - 1] += 1

        normalized_cm = np.zeros([16, 16])
        sumation = np.sum(cm, 1)
        for i in range(16):
            for j in range(16):
                if (cm[i, j]) != 0:
                    normalized_cm[i, j] = cm[i, j] / sumation[i]
        print(normalized_cm)

        plt.clf()
        plt.imshow(normalized_cm, interpolation='nearest', cmap=plt.cm.Blues)
        classNames = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        plt.title('Confusion Matrix (Minimum Distance Criterion)')
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


if __name__ == "__main__":
    file_name = "arrhythmia.data.csv"
    read_data = Read_data(file_name)
    original_data = read_data.load_file()

    pre_process = Pre_Process(original_data)
    data = pre_process.cleaning()

    # Define vector class id as the last column of matrix arrhythmia, define
    # matrix y as the other columns
    class_id = data.loc[:, 279]
    y = data.drop(279, axis=1)

    # boolean variable with True or False
    is_one = (class_id == 1)
    is_two = (class_id == 2)
    # filter rows using  the boolean variable
    y1 = y[is_one]
    y2 = y[is_two]

    # x1, the mean of the row vectors in y1, and x2, the mean of the row vectors in y2
    x1 = np.mean(y1, axis=0)
    x2 = np.mean(y2, axis=0)

    mdc = Minimum_Distance_Criterion(x1, x2, y, class_id)
    mdc.run()

    bc = Bayes_Criterion(y1, y2, y, class_id)
    bc.run()

    mdcnb = Minimum_Distance_Criterion_Non_Binary(original_data, y)
    mdcnb.run()