from keras import backend as K 
from matplotlib import pyplot 
from numpy import *
from attention import AttentionLayer
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd 
from pathlib import Path
import re
import keras
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings

class TextSummarizationModel:
    def __init__(self, xTrain, yTrain, xVal, yVal, xVocabSize, yVocabSize, maxTextLen, tokenizerX, tokenizerY):
        self.model = self.getModel(xTrain, yTrain, xVal, yVal, xVocabSize, yVocabSize, maxTextLen)
        self.buildDictinaryToConvertIndexToWord(tokenizerX, tokenizerY)
        self.buildInferenceForEncoderDecoder()
        
    def drawModelFromTraining(self):
        pyplot.plot(self.history.history['loss'], label='train') 
        pyplot.plot(self.history.history['val_loss'], label='test') 
        pyplot.legend() 
        pyplot.show()

    def getModel(self, xTrain, yTrain, xVal, yVal, xVocabSize, yVocabSize, maxTextLen):
        my_file = Path("./textSumamrizationModel.h5")
        if my_file.is_file(): 
            self.model = keras.models.load_model('./textSumamrizationModel.h5', custom_objects={'AttentionLayer': AttentionLayer})

        else:
            self.encoderInput = Input(shape=(maxTextLen,))
            embL = Embedding(xVocabSize, 200,trainable=True)(self.encoderInput)
            encoderLSTM = LSTM(300, return_state=True, return_sequences=True,dropout=0.4,recurrent_dropout=0.4)
            self.encoderOutput, self.stateH, self.stateC= encoderLSTM(embL)

            # Decoder
            self.decoderInput = Input(shape=(None,))
            self.decL = Embedding(yVocabSize, 200,trainable=True)
            decEmb = self.decL(self.decoderInput)

            self.decoderLstm = LSTM(300, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
            decoderOutputs,decoderFwdState, decoderBackState = self.decoderLstm(decEmb,initial_state=[self.stateH, self.stateC])

            #Attention layer
            self.attnL = AttentionLayer(name='attention_layer')
            attnO, attnS = self.attnL([self.encoderOutput, decoderOutputs])
            decoderCInput = Concatenate(axis=-1, name='concat_layer')([decoderOutputs, attnO])
            #dense layer
            self.decoderDense =  TimeDistributed(Dense(yVocabSize, activation='softmax'))
            decoderOutputs = self.decoderDense(decoderCInput)

            # Define the model 
            self.model = Model([self.encoderInput, self.decoderInput], decoderOutputs)
            self.model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
            self.model.summary()
            es = EarlyStopping(monitor='val_loss', mode='min', verbose = 1, patience = 2)

            self.history =  self.model.fit([xTrain,yTrain[:,:-1]], yTrain.reshape(yTrain.shape[0],yTrain.shape[1], 1)[:,1:] ,epochs= 5,callbacks=[es],batch_size=128, validation_data=([xVal,yVal[:,:-1]], yVal.reshape(yVal.shape[0],yVal.shape[1], 1)[:,1:]))
            
            self.model.save('textSumamrizationModel.h5')
            self.drawModelFromTraining()
        
    def buildDictinaryToConvertIndexToWord(self, tokenizerX, tokenizerY):
        self.revTargetWordIndex = tokenizerY.index_word
        self.revSourcetWordIndex = tokenizerX.index_word
        self.targetWord = tokenizerY.word_index

    def buildInferenceForEncoderDecoder(self):
        
        #################################
        self.encoderModel = Model(inputs=self.encoderInput,outputs=[self.encoderOutput, self.stateH, self.stateC])

        # Decoder setup
        # Below tensors will hold the states of the previous time step
        decoderStateInputH = Input(shape=(300,))
        decoderStateInputC = Input(shape=(300,))
        decoderHiddenStateInput = Input(shape=(400,300))

        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        decoderOutputs2 , s, c = self.decoderLstm(self.decL(self.decoderInput), initial_state=[decoderStateInputH, decoderStateInputC])

        #attention inference
        aInf, aSInf = self.attnL([decoderHiddenStateInput, decoderOutputs2])
        dInfC = Concatenate(axis=-1, name='concat')([decoderOutputs2, aInf])

        # A dense softmax layer to generate prob dist. over the target vocabulary
        decoderOutputs2 = self.decoderDense(dInfC) 

        # Final decoder model
        self.decoderModel = Model(
            [self.decoderInput] + [decoderHiddenStateInput,decoderStateInputH, decoderStateInputC],
            [decoderOutputs2] + [s, c])
    
    def decodeSeq(self, seq):
#         self.revTargetWordIndex = tokenizerY.index_word
#         self.revSourcetWordIndex = tokenizerX.index_word
#         self.targetWord = tokenizerY.word_index
    # Encode the input as state vectors.
        eo, eh, ec = self.encoderModel.predict(seq)

        # Generate empty target sequence of length 1.
        tseq = np.zeros((1,1))

        # Populate the first word of target sequence with the start word.
        tseq[0, 0] = self.targetWord['beginmush']

        stopCondition = False
        decodedsent = ''
        while not stopCondition:

            output_tokens, h, c = self.encoderModel.predict([tseq] + [eo, eh, ec])

            # Sample a token
            sti = np.argmax(output_tokens[0, -1, :])
            sampleTok = self.revTargetWordIndex[sti]

            if(sampleTok!='endmush'):
                decodedsent += ' '+sampleTok

            # Exit condition: either hit max length or find stop word.
            if (sampleTok == 'endmush'  or len(decodedsent.split()) >= (max_summary_len-1)):
                stopCondition = True

            # Update the target sequence (of length 1).
            tseq = np.zeros((1,1))
            tseq[0, 0] = sti

            # Update internal states
            eh, ec = h, c

        return decodedsent       

    def seq2summary(self, seq):
        newString=''
        for i in seq:
            if((i!=0 and i!=self.targetWord['beginmush']) and i!=self.targetWord['endmush']):
                newString += self.revTargetWordIndex[i]+' '
        return newString

    def seq2text(self, seq):
        newString=''
        for i in seq:
            if(i!=0):
                newString=newString+self.revSourcetWordIndex[i]+' '
        return newString
    



        
