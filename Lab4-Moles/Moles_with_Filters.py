from sklearn.cluster import KMeans
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import math
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
import cv2 as cv


class Moles(object):

    def __init__(self,imageFile):
        self.originalImage = mpimg.imread(imageFile)
        self.filtered_image = cv.blur(self.originalImage , (5, 5))
        self.filtered_image = cv.GaussianBlur(self.filtered_image,(5,5),0)
        self.filtered_image = cv.medianBlur(self.filtered_image, 5)
#        self.filtered_image = self.originalImage

    def showImage(self,image,titleStr):
        plt.figure()
        plt.imshow(image)
        plt.title(titleStr)
        #plt.show()

    def K_menas(self,NoClusters,seeds):
        kmeans = KMeans(n_clusters = NoClusters, random_state=seeds)

        [self.N1, self.N2, self.N3] = self.filtered_image.shape
        im_2D = self.filtered_image.reshape((self.N1 * self.N2, self.N3))

        kmeans.fit(im_2D)
        imm = im_2D.copy()
        self.centroids = kmeans.cluster_centers_.astype('uint8')
        labels = kmeans.labels_
        self.clusters = kmeans.predict(im_2D).reshape(self.N1, self.N2)

        for kc in range(NoClusters):
            ind = (kmeans.labels_ == kc)
            #print(ind)
            #print(self.centroids[kc, :])
            imm[ind, :] = self.centroids[kc, :]

        self.quantized_image = imm.reshape((self.N1, self.N2, self.N3))
        
    def crop_image(self,titleStr):
        self.filter_clusters = cv.blur(self.clusters, (5, 5))
        #self.filter_clusters = self.clusters

        # Among the centroids, find the darkest color, it corresponds to the mole
        darkest_color = np.amin(self.centroids, axis=0)
        #print("darkest color: ", darkest_color)
        #finding the darkest color index
        self.darkest_color_index = np.where(self.centroids.sum(axis=1) == np.min(self.centroids.sum(axis=1)))
        #print(self.darkest_color_index)

        # Find the median
        indexes = np.where(self.filter_clusters == self.darkest_color_index)
        x_median = np.median(indexes[1])
        y_median = np.median(indexes[0])
        y = int(self.N2/2)
        x = int(self.N1/2)
#        x = int(x_median)
#        y = int(y_median)

        plt.figure('median')
        plt.imshow(self.quantized_image)
        plt.title('Median of '+titleStr)
        plt.scatter(x, y)
        #plt.show()

        right = x
        left = x
        up = y
        down = y

        count_right = 0
        count_left = 0
        count_up = 0
        count_down = 0

        #find the rectangle to crop the image
        not_darek = 0

        while (self.filter_clusters[right][y] == self.darkest_color_index or not_darek < 11):
            right = right + 1
            count_right = count_right + 1
            if (self.filter_clusters[right][y] == self.darkest_color_index):
                not_darek = 0
            else:
                not_darek = not_darek + 1

        not_darek = 0

        while (self.filter_clusters[left][y] == self.darkest_color_index):
            left = left - 1
            count_left = count_left + 1
            if (self.filter_clusters[right][y] == self.darkest_color_index):
                not_darek = 0
            else:
                not_darek = not_darek + 1
        not_darek = 0

        while (self.filter_clusters[x][up] == self.darkest_color_index):
            up = up + 1
            count_up = count_up + 1
            if (self.filter_clusters[right][y] == self.darkest_color_index):
                not_darek = 0
            else:
                not_darek = not_darek + 1

        not_darek = 0

        while (self.filter_clusters[x][down] == self.darkest_color_index):
            down = down - 1
            count_down = count_down + 1
            if (self.filter_clusters[right][y] == self.darkest_color_index):
                not_darek = 0
            else:
                not_darek = not_darek + 1

        # calculate the crop size with 50 pixel as safe margin
        max_distance = max(count_left, count_right, count_up, count_down)
        crop_size = max_distance + 50

        self.cropped_image = self.originalImage[x - crop_size: x + crop_size, y - crop_size: y + crop_size:]
        self.cropped_clusters = self.clusters[x - crop_size:x + crop_size, y - crop_size:y + crop_size:]

    def finding_bordders(self,titleStr):
        [self.m1, self.m2] = self.cropped_clusters.shape
        self.moles = np.zeros((self.m1 + 2, self.m2 + 2))
        borders = np.zeros((self.m1, self.m2))

        for i in range(self.m1):
            for j in range(self.m2):
                if (self.cropped_clusters[i][j] == self.darkest_color_index):
                    self.moles[i + 1][j + 1] = 1

        # Consider each column of the sub-image, find the index of the first pixel
        # having the darkest color and the index of the last pixel having the darkest
        # color
        column_vector = []
        for i in range(self.m1):
            for j in range(self.m1):
                if self.moles[i][j] == 1:
                    column_vector.append(j)
            if len(column_vector) != 0:
                borders[i][column_vector[0]] = 1
                borders[i][column_vector[-1]] = 1
            column_vector = []

        # Repeat the above point, considering now the rows
        row_vector = []
        for i in range(self.m1):
            for j in range(self.m1):
                if self.moles[j][i] == 1:
                    row_vector.append(j)
            if len(row_vector) != 0:
                borders[row_vector[0]][i] = 1
                borders[row_vector[-1]][i] = 1
            row_vector = []

        borders_index = np.where(borders == 1)
        plt.figure('borders')
        plt.imshow(self.cropped_image)
        plt.scatter(borders_index[1], borders_index[0], s=1, c="r")
        plt.title('Borders of '+titleStr)
        #plt.show()

        perimeter = borders.sum()
        area = self.moles.sum()

        radius = np.sqrt(area / np.pi)
        perimeter_circle = 2 * np.pi * radius

        ratio = perimeter / perimeter_circle
        return ratio

    def improvment(self,titleStr):
        processed_moles = self.moles
        processed_borders = np.zeros((self.m1 + 2, self.m2 + 2))

        moles_index = np.where(self.moles == 1)

        for i in range(len(moles_index[0])):

            n = 0
            s = 0
            e = 0
            w = 0

            ne = 0
            se = 0
            nw = 0
            sw = 0

            x_index = moles_index[0][i]
            y_index = moles_index[1][i]

            cell = self.moles[x_index][y_index]

            if (cell == 1):
                n = self.moles[x_index - 1][y_index]
                s = self.moles[x_index + 1][y_index]
                e = self.moles[x_index][y_index + 1]
                w = self.moles[x_index][y_index - 1]

                ne = self.moles[x_index - 1][y_index + 1]
                se = self.moles[x_index + 1][y_index + 1]
                nw = self.moles[x_index - 1][y_index - 1]
                sw = self.moles[x_index + 1][y_index - 1]

            neighbors = n + s + e + w + ne + se + nw + sw

            if (neighbors != 0 and neighbors != 8):
                processed_borders[x_index][y_index] = 1

            if (neighbors == 0):
                processed_moles[x_index][y_index] = 0

        processed_borders_index = np.where(processed_borders == 1)
        plt.figure('Finale Borders')
        plt.imshow(self.cropped_image)
        plt.scatter(processed_borders_index[1], processed_borders_index[0], s=1, c="r")
        plt.title('Finale Calculate borders of '+ titleStr)
        plt.show()

        perimeter = processed_borders.sum()
        area = processed_moles.sum()

        radius = np.sqrt(area / np.pi)
        perimeter_circle = 2 * np.pi * radius

        ratio = perimeter / perimeter_circle
        return ratio


if __name__ == "__main__":

    num_of_clusters = 3
    seeds = 0

    low_risk_ratio = []
    final_low_risk_ratio = []
    #
    try:
        for i in range(11):
            name = "low_risk_" + str(i + 1)
            filein = "images/" + name + ".jpg"
            moles = Moles(filein)

            moles.showImage(moles.originalImage, "Original Image of " + name)
            moles.showImage(moles.filtered_image, "First filter on " + name)

            moles.K_menas(num_of_clusters, seeds)
            moles.showImage(moles.quantized_image, "Quantized Image of " + name)
            moles.showImage(moles.clusters, "K-means Cluster of " + name)
            # moles.showImage(moles.filter_clusters, "Filter the Clusters of " + name)
            moles.crop_image(name)
            moles.showImage(moles.cropped_image, "Cropped Image of " + name)
            moles.showImage(moles.cropped_clusters, "Cropped Clusters of " + name)
            ratio = moles.finding_bordders(name)
            if not math.isnan(ratio):
                low_risk_ratio.append(ratio)
            better_ratio = moles.improvment(name)
            if not math.isnan(ratio):
                final_low_risk_ratio.append(better_ratio)
            print("First Low risk  mean ratio calculated: ", np.mean(low_risk_ratio))
            print("Final Low risk  mean ratio calculated: ", np.mean(final_low_risk_ratio))
    except:
        pass
    #
    medium_risk_ratio = []
    final_medium_risk_ratio = []

    for i in range(16):
        name = "medium_risk_" + str(i + 1)
        filein = "images/" + name + ".jpg"
        moles = Moles(filein)

        moles.showImage(moles.originalImage, "Original Image of " + name)
        moles.showImage(moles.filtered_image, "First filter on " + name)

        moles.K_menas(num_of_clusters, seeds)
        moles.showImage(moles.quantized_image, "Quantized Image of " + name)
        moles.showImage(moles.clusters, "K-means Cluster of " + name)
        #   moles.showImage(moles.filter_clusters, "Filter the Clusters of " + name)
        moles.crop_image(name)
        moles.showImage(moles.cropped_image, "Cropped Image of " + name)
        moles.showImage(moles.cropped_clusters, "Cropped Clusters of " + name)
        ratio = moles.finding_bordders(name)
        if not math.isnan(ratio):
            medium_risk_ratio.append(ratio)
        better_ratio = moles.improvment(name)
        if not math.isnan(better_ratio):
            final_medium_risk_ratio.append(better_ratio)
        print("First Medium risk  mean ratio calculated: ", np.mean(medium_risk_ratio))
        print("Final Medium risk  mean ratio calculated: ", np.mean(final_medium_risk_ratio))

    melanoma_risk_ratio = []
    final_melanoma_risk_ratio = []

    for i in range(27):
        name = "melanoma_" + str(i + 1)
        filein = "images/" + name + ".jpg"
        moles = Moles(filein)

        moles.showImage(moles.originalImage, "Original Image of " + name)
        moles.showImage(moles.filtered_image, "First filter on " + name)

        moles.K_menas(num_of_clusters, seeds)
        moles.showImage(moles.quantized_image, "Quantized Image of " + name)
        moles.showImage(moles.clusters, "K-means Cluster of " + name)
        # moles.showImage(moles.filter_clusters, "Filter the Clusters of " + name)
        moles.crop_image(name)
        moles.showImage(moles.cropped_image, "Cropped Image of " + name)
        moles.showImage(moles.cropped_clusters, "Cropped Clusters of " + name)
        ratio = moles.finding_bordders(name)
        if not math.isnan(ratio):
            melanoma_risk_ratio.append(ratio)
        better_ratio = moles.improvment(name)
        if not math.isnan(better_ratio):
            final_melanoma_risk_ratio.append(better_ratio)
        print("First Melanoma mean ratio calculated: ", np.mean(melanoma_risk_ratio))
        print("Final Melanoma  mean ratio calculated: ", np.mean(final_melanoma_risk_ratio))

    print("First Low risk  mean ratio calculated: ", np.mean(low_risk_ratio))
    print("Final Low risk  mean ratio calculated: ", np.mean(final_low_risk_ratio))

    print("First Medium risk  mean ratio calculated: ", np.mean(medium_risk_ratio))
    print("Final Medium risk  mean ratio calculated: ", np.mean(final_medium_risk_ratio))

    print("First Melanoma mean ratio calculated: ", np.mean(melanoma_risk_ratio))
    print("Final Melanoma  mean ratio calculated: ", np.mean(final_melanoma_risk_ratio))
        