import numpy as np
import skimage
import skimage.feature
import skimage.color
import cv2

class Image:
    def __init__(self, image, name): 
        self.image = self.setImage(image)
        self.name = self.setName(name)
        self.numberOfGrid = self.setNumberOfGrid(name)
        self.numberOfImageBelonging = self.setNumberOfImageBelonging(name)
        self.datasetBelonging = self.setDatasetBelonging(name)
        self.classBelonging = self.setClassBelonging(name)
        self.size = self.setSize(image)
        
        self.colorChannelsRGB = self.extractColorChannelsRGB()
        self.RGBMean = self.calculateRGBMean()
        self.RGBMode = self.calculateRGBMode()
        self.RGBVariance = self.calculateRGBVariance()
        self.RGBStandardDeviation = self.calculateRGBStandardDeviation()
        self.colorHistogram = self.calculateColorHistogram()

        self.grayLevelCooccurrenceMatrixProperties = self.calculateGrayLevelCooccurrenceMatrixProperties()
        self.histogramOfOrientedGradients = self.calculateHistogramOfOrientedGradients()
        self.peakLocalMax = self.calculatePeakLocalMax()

    def setImage(self, image):
        return image
    
    def setName(self, name):
        return name
    
    def setNumberOfGrid(self, name):
        return name.split("_")[3]
    
    def setNumberOfImageBelonging(self, name):
        return name.split("_")[2]
    
    def setDatasetBelonging(self, name):
        return name.split("_")[1]
    
    def setClassBelonging(self, name):
        return name.split("_")[4]
    
    def setSize(self, image):
        return image.shape
     
    def extractColorChannelsRGB(self):
        redChannel = self.image[:,:,0]
        greenChannel = self.image[:,:,1]
        blueChannel = self.image[:,:,2]

        return [redChannel, greenChannel, blueChannel]
    
    def calculateRGBMean(self):
        redChannel = self.image[:,:,0]
        greenChannel = self.image[:,:,1]
        blueChannel = self.image[:,:,2]
        redMean = np.mean(redChannel)
        greenMean = np.mean(greenChannel)
        blueMean = np.mean(blueChannel) 

        return [redMean, greenMean, blueMean]
    
    def calculateRGBMode(self):
        redChannel = self.image[:,:,0]
        greenChannel = self.image[:,:,1]
        blueChannel = self.image[:,:,2]
        redMode = skimage.exposure.histogram(redChannel)[1].argmax()
        greenMode = skimage.exposure.histogram(greenChannel)[1].argmax()
        blueMode = skimage.exposure.histogram(blueChannel)[1].argmax()

        return [redMode, greenMode, blueMode]
    
    def calculateRGBVariance(self):
        redChannel = self.image[:,:,0]
        greenChannel = self.image[:,:,1]
        blueChannel = self.image[:,:,2]
        redVariance = np.var(redChannel.flatten())
        greenVariance = np.var(greenChannel.flatten())
        blueVariance = np.var(blueChannel.flatten())

        return [redVariance, greenVariance, blueVariance]
    
    def calculateRGBStandardDeviation(self):
        redChannel = self.image[:,:,0]
        greenChannel = self.image[:,:,1]
        blueChannel = self.image[:,:,2]
        redStandardDeviation = np.std(redChannel)
        greenStandardDeviation = np.std(greenChannel)
        blueStandardDeviation = np.std(blueChannel)

        return [redStandardDeviation, greenStandardDeviation, blueStandardDeviation]
    
    def calculateColorHistogram(self):
        image = self.image
        bins = 256
        # Calcula el histograma para cada canal de color
        histogram = [cv2.calcHist([image], [i], None, [bins], [0, 256]) for i in range(3)]
        # Normaliza y aplana el histograma para convertirlo en un vector
        histogram = [cv2.normalize(hist, hist).flatten() for hist in histogram]

        return histogram

    def calculateGrayLevelCooccurrenceMatrixProperties(self):
        image_gray = skimage.color.rgb2gray(self.image)
        image_gray_u8 = (image_gray * 255).astype(np.uint8)
        glcm = skimage.feature.graycomatrix(image_gray_u8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = skimage.feature.graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = skimage.feature.graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = skimage.feature.graycoprops(glcm, 'homogeneity')[0, 0]
        energy = skimage.feature.graycoprops(glcm, 'energy')[0, 0]
        correlation = skimage.feature.graycoprops(glcm, 'correlation')[0, 0]

        return [contrast, dissimilarity, homogeneity, energy, correlation]

    def calculateHistogramOfOrientedGradients(self):
        image_gray = skimage.color.rgb2gray(self.image)

        return skimage.feature.hog(image_gray, pixels_per_cell=(16, 16), cells_per_block=(1, 1), orientations=9, visualize=False)

    def calculatePeakLocalMax(self):
        image_gray = skimage.color.rgb2gray(self.image)

        return skimage.feature.peak_local_max(image_gray, min_distance=1, threshold_abs=0.1, num_peaks=10)
    
    
    
    def generateFeatureVector(self):
        featureVector = np.array([])

        featureVector = np.append(featureVector, np.concatenate([ channel.flatten() for channel in self.colorChannelsRGB ]))
        featureVector = np.append(featureVector, self.RGBMean)
        featureVector = np.append(featureVector, self.RGBMode)
        featureVector = np.append(featureVector, self.RGBVariance)
        featureVector = np.append(featureVector, self.RGBStandardDeviation)
        featureVector = np.append(featureVector, np.concatenate([ histogram.flatten() for histogram in self.colorHistogram ]))
        featureVector = np.append(featureVector, self.grayLevelCooccurrenceMatrixProperties)
        featureVector = np.append(featureVector, self.histogramOfOrientedGradients)
        featureVector = np.append(featureVector, self.peakLocalMax)

        return featureVector
    