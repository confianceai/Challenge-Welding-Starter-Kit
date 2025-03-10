# This class satisfy the AI component interface required for your solution be accepted as challenge candidate.
# In contains as requested a class named "myAicomponent" containing the 2 requested methods (load_model(), and predict()).
# Of course you are free to add any other methods you may need. This is the case here .

from abc import ABC, abstractmethod
from challenge_solution.absAIComponent import AbstractAIComponent
import numpy as np

# New version
import os
import cv2
import tensorflow as tf
import pickle
import keras
import sklearn
from keras import Sequential, layers, ops, Model, activations
from keras.layers import Conv2D,BatchNormalization,Activation,MaxPooling2D,Flatten,Dense,ZeroPadding2D,Dropout,Rescaling,TimeDistributed,Resizing
from keras.regularizers import l2

from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from scipy.optimize import minimize
from scipy.spatial.distance import mahalanobis
from scipy.optimize import minimize
from numpy import cov
from numpy.linalg import inv

class MyAIComponent(AbstractAIComponent):
    def __init__(self,img_shape=(360, 640, 3), filter_size=64, dense_size=64, n_classes=2, l2_reg=0.0001,conf_threshold=0.7,name='MyAIComponent'):
        """
         Init a ML-component for detect non-conform weld from image.
        """                      
        self.name=name
        #Model parameters
        self.filter_size = filter_size
        self.dense_size = dense_size
        self.img_shape = img_shape
        self.n_classes = n_classes
        self.l2_reg = l2_reg

        # Conf & OOD parameters
        self.conf_threshold = conf_threshold
        self.temperature = tf.Variable(1.0, dtype=tf.float32, trainable=True)
        self.resolution = (img_shape[0],img_shape[1],img_shape[2])

        #Parameters to estimate for models
        self.initialized = False
        self.model = None
        self.full_model = None
         
        #Parameters to estimate for trust
        self.initialized = False
        self.threshold = None
        self.temperature = None
        self.mean = None
        self.inv_cov = None
        
    def _init_model(self,params=None):
        model_params = {'img_shape':self.img_shape,'filter_size':self.filter_size,'dense_size':self.dense_size,'n_classes':self.n_classes,'l2_reg':self.l2_reg}
        self.model = alexnet_model(**model_params)
        self.model.build((None, self.resolution[0], self.resolution[1], self.resolution[2]))
        self.initialized = True
        
    def preprocess(self,list_X,list_y):
        list_resolution = [i.shape for i in list_X]
        for n,resolution in enumerate((list_resolution)):
            if ((resolution[0] != self.resolution[0]) | (resolution[1] != self.resolution[1])):
                list_X[n] = cv2.resize(list_X[n], dsize=(self.resolution[0], self.resolution[1]), interpolation=cv2.INTER_CUBIC)

        list_X = np.concatenate([i[None] for i in list_X])

        if(list_y is not None):
            if(type(list_y) is list):
                list_y = np.array(list_y)
        
        return(list_X,list_y)
            
    def set_full_model(self):
        self.full_model = Model(inputs=self.model.get_layer("input").input, outputs=[self.model.layers[-1].output,self.model.get_layer("latent_space").output])
        
    def fit(self,X,y=None,validation_data=None,epochs=50,class_weight=None,callback='default',learning_rate=0.001):
        """
        fit model and trust module
        """

        if(type(X) is list):
            X,y = self.preprocess(X,y)
        

        if(not(self.initialized)):
            self._init_model()
        
        callbacks = None
        if(callback=='default'):
            call0=tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience=30, verbose=0, mode='max',restore_best_weights=True)
            call1=tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', min_delta=0.001, factor=0.2, patience=10, verbose=0, mode='min', cooldown=0, min_lr=0.00005)
            callbacks=[call0,call1]
            
        opt=tf.optimizers.Adam(learning_rate)
        self.model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=[tf.metrics.SparseCategoricalAccuracy()])
        hist=self.model.fit(x=X,y=y,validation_data=validation_data,class_weight=class_weight,epochs=epochs,shuffle=True,callbacks=callbacks,verbose=False)
        self.model.save_weights(self.name+'.weights.h5')

        # Set model that provide penultimate logits and embedding
        self.set_full_model()
        
        # Compute logit 
        if(validation_data is None):
            validation_data = X
            
        logits, features = self.full_model.predict(validation_data)

        #
        self.fit_ood_params(features,y)
        
        #
        self.fit_uncertainty_params(logits,y)
        return(hist)
    def fit_uncertainty_params(self,logits,y=None):
        """
        Fit uncertainty parameters related to probabilties calibration
        """
        def loss_fn(T):
            """
            estimation of temperature parameters on validation set
            """
            T = np.maximum(T, 1e-6)  # Ensure temperature is positive
            scaled_logits = logits / T
            loss = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)(y, scaled_logits)
            return loss.numpy()
        
        result = minimize(loss_fn, x0=np.array([1]), method='Powell', bounds=[(1e-6, 100)],options={"maxiter":100})
        self.temperature = result.x[0]
        
    def fit_ood_params(self,features,y=None):
        """
        Fit ood parameters related to ood detection
        """
        # Latent space monitoring
        self.PCA = sklearn.decomposition.PCA(n_components=10)
        features = self.PCA.fit_transform(features)
        self.mean = np.mean(features, axis=0)
        covariance_matrix = cov(features, rowvar=False)
        self.inv_cov = inv(covariance_matrix + np.eye(covariance_matrix.shape[0]) * 1e-6)  # Regularization
        distances = np.array([mahalanobis(f, self.mean, self.inv_cov) for f in features])
        self.threshold = distances.max() * 1.1
        
    
    def predict(self,input_images, images_meta_informations):
        """
        Perform a prediction using the appropriate model.
        Parameters:
            input_image: The input image as a NumPy array.
            image_meta_information: Metadata dictionary for the image.
        Returns:
            A string prediction from ["OK", "KO", "UNKNOWN"].
        """

        if(type(input_images) is list):
            X, _ = self.preprocess(input_images,None)

        else:
            X = input_images

        # Model inference
        logits,features = self.full_model.predict(X,verbose=False)
        # Uncertainty estimation
        probabilities = self.predict_uncertainty(logits)
        # Ood score computation
        ood_scores = self.predict_ood_score(features)

        list_pred,list_prob,list_score = self.postprocessing(probabilities,ood_scores)
        
        # Create final output dict
        
        return {"predictions" : list_pred, "probabilities": list_prob, "OOD_scores": list_score}

    def predict_uncertainty(self,logits):
        """ Calibrate logits to obtain uncertainty score"""
        scaled_logits = logits/self.temperature
        probabilities = tf.nn.softmax(scaled_logits).numpy()
        return probabilities

    def predict_ood_score(self, features):
        """ Compute ood-score """
        features = self.PCA.transform(features)
        distances = [mahalanobis(f, self.mean, self.inv_cov) for f in features]
        return np.array(distances) / self.threshold  # Returns True for OOD samples

    
    def postprocessing(self,probabilities,ood_scores):
        """ Post_process ood-score """
        list_pred = []
        list_prob = []
        list_score= []
        for probability,ood_score in zip(probabilities,ood_scores):
            pred,prob = self.uncertainty_mitigation(probability,ood_score)
            if(pred == -1):
                list_pred.append('UNKNOWN')
            if(pred == 0):
                list_pred.append('KO')
            if(pred == 1):
                list_pred.append('OK')
            list_prob.append(prob)
            list_score.append(ood_score)
        return(list_pred,list_prob,list_score)
        
    def uncertainty_mitigation(self,probability,ood_score):
        """
        uncertainty_mitigation determine if unknow from conf_threeshold or ood_threeshold
        """
        new_probabilities = np.zeros(3)
        if((max(probability)< self.conf_threshold) or ood_score>1):
            prediction = -1
            new_probabilities[:2] = probability/2
            new_probabilities[2] = 0.5
        else:
            prediction = np.argmax(probability)
            new_probabilities[:2] = probability
            new_probabilities[2] = 0
        return(prediction,new_probabilities)


    def save_model(self, config={'path':'','name':'ML_Component'}):
        """
        Save model on a model.p containing parameters and a model.weights.h5 containing model weights
        """
        # Save model_weights
        save_path = os.path.join(config['path'],config['name']+'.weights.h5')
        self.model.save_weights(save_path)

        model_tmp = self.model
        # Save config
        self.model = True
        dict_parameters = self.__dict__

        save_path = os.path.join(config['path'],config['name']+'.p')
        pickle.dump(dict_parameters, open(save_path, "wb"))
        self.model = model_tmp

    def load_model(self):
        """
        load model from a model.p containing parameters and a model.weights.h5 containing model weights
        """
        
        # suppose here that python current dir is component path
        config={'path':'','name':'model_test'}

        # Read model_config
        load_path = os.path.join(config['path'],config['name']+'.p')
        dict_parameters = pickle.load(open(load_path, "rb"))

        for attributes, values in dict_parameters.items():
            self.__setattr__(attributes, values)
            
        # Load mode weigth
        self._init_model()
        load_path = os.path.join(config['path'],config['name']+'.weights.h5')
        self.model.load_weights(load_path)
        self.set_full_model()

def alexnet_model(img_shape=(360, 640, 3), filter_size=256, dense_size=512, n_classes=2, l2_reg=0.0001):
    """
    Generate a keras tensorflow alexnet model for classification
    """
    # Initialize model
    alexnet = Sequential()

    # Layer 1
    alexnet.add(Resizing(img_shape[0], img_shape[1],interpolation="bilinear",crop_to_aspect_ratio=False,pad_to_aspect_ratio=False,fill_mode="constant", fill_value=0.0,name='input'))
    alexnet.add(Rescaling(1./255,input_shape=img_shape))

    alexnet.add(Conv2D(filter_size,(7, 7),strides=(5,5), padding='same', kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(4, 4)))

    # Layer 2
    alexnet.add(Conv2D(filter_size, (3,3),padding='same', kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    
    alexnet.add(Conv2D(filter_size, (5, 5), padding='same', kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 4
    alexnet.add(Conv2D(filter_size, (3, 3), padding='same', kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 5
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(filter_size, (2, 2), padding='same', kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Layer 6
    alexnet.add(ZeroPadding2D((1, 1)))
    alexnet.add(Conv2D(filter_size, (1, 1), padding='same', kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))
    alexnet.add(Dropout(0.5))

    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(dense_size, kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 7
    alexnet.add(Dense(int(dense_size/2),name="latent_space", kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 8
    alexnet.add(Dense(n_classes,name='logits'))
    alexnet.add(Activation('softmax'))
    return alexnet

    