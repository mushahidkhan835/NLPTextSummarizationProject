from keras import backend as K 
from matplotlib import pyplot 
from numpy import *
from attention import AttentionLayer
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd 
from pathlib import Path
import re
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings

class TextSummarizationModel:
    def __init__(self, xTrain, yTrain, xVal, yVal, xVocabSize, yVocabSize, maxTextLen, tokenizerX, tokenizerY):
        self.model = self.getModel(xTrain, yTrain, xVal, yVal, xVocabSize, yVocabSize, maxTextLen)
        self.buildDictinaryToConvertIndexToWord(tokenizerX, tokenizerY)
        self.buildInferenceForEncoderDecoder()
        for i in range(0, 10):
                print("Text:",self.seq2text(xTrain[i]))
                print("Original summary:",self.seq2summary(yTrain[i]))
                print("Predicted summary:",self.decodeSeq(xTrain[i].reshape(1,maxTextLen)))
                print("\n")

        
    def drawModelFromTraining(self):
        pyplot.plot(self.history.history['loss'], label='train') 
        pyplot.plot(self.history.history['val_loss'], label='test') 
        pyplot.legend() 
        pyplot.show()

    def getModel(self, xTrain, yTrain, xVal, yVal, xVocabSize, yVocabSize, maxTextLen):
        my_file = Path("/textSumamrizationModel.h5")
        if my_file.is_file(): 
            self.model = keras.models.load_model('/textSumamrizationModel.h5')
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
            decoder_dense =  TimeDistributed(Dense(yVocabSize, activation='softmax'))
            decoderOutputs = decoder_dense(decoderCInput)

            # Define the model 
            self.model = Model([self.encoderInput, self.decoderInput], decoderOutputs)
            self.model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
            self.model.summary()
            es = EarlyStopping(monitor='val_loss', mode='min', verbose = 1, patience = 2)

            self.history =  self.model.fit([xTrain,yTrain[:,:-1]], yTrain.reshape(yTrain.shape[0],yTrain.shape[1], 1)[:,1:] ,epochs= 30,callbacks=[es],batch_size=128, validation_data=([xVal,yVal[:,:-1]], yVal.reshape(yVal.shape[0],yVal.shape[1], 1)[:,1:]))
            
            self.model.save('textSumamrizationModel.h5')
            self.drawModelFromTraining()
        
    def buildDictinaryToConvertIndexToWord(self, tokenizerX, tokenizerY):
        self.revTargetWordIndex = tokenizerY.index_word
        self.revSourcetWordIndex = tokenizerX.index_word
        self.targetWord = tokenizerY.word_index

    def buildInferenceForEncoderDecoder(self):
        self.encoderModel = Model(inputs=self.encoderInput,outputs=[self.encoderOutput, self.stateH, self.stateC])

        # Decoder setup
        # Below tensors will hold the states of the previous time step
        decoderStateInputH = Input(shape=(latent_dim,))
        decoderStateInputC = Input(shape=(latent_dim,))
        decoderHiddenStateInput = Input(shape=(max_text_len,latent_dim))

        # To predict the next word in the sequence, set the initial states to the states from the previous time step
        decoderOutputs2 , s, c = self.decoderLSTM(self.decL(self.decoderInput), initial_state=[decoderStateInputH, decoderStateInputC])

        #attention inference
        aInf, aSInf = self.attnL([decoderHiddenStateInput, decoderOutputs2])
        dInfC = Concatenate(axis=-1, name='concat')([decoderOutputs2, aInf])

        # A dense softmax layer to generate prob dist. over the target vocabulary
        decoderOutputs2 = decoder_dense(dInfC) 

        # Final decoder model
        self.decoderModel = Model(
            [self.decoderInput] + [decoderHiddenStateInput,decoderStateInputH, decoderStateInputC],
            [decoderOutputs2] + [s, c])
    
    def decodeSeq(self, seq):
#         self.revTargetWordIndex = tokenizerY.index_word
#         self.revSourcetWordIndex = tokenizerX.index_word
#         self.targetWord = tokenizerY.word_index
    # Encode the input as state vectors.
        eo, eh, ec = encoder_model.predict(seq)

        # Generate empty target sequence of length 1.
        tseq = np.zeros((1,1))

        # Populate the first word of target sequence with the start word.
        tseq[0, 0] = self.targetWord['_START_']

        stopCondition = False
        decodedsent = ''
        while not stopCondition:

            output_tokens, h, c = self.predict([tseq] + [eo, eh, ec])

            # Sample a token
            sti = np.argmax(output_tokens[0, -1, :])
            sampleTok = self.revTargetWordIndex[sti]

            if(sampleTok!='_END_'):
                decodedsent += ' '+sampleTok

            # Exit condition: either hit max length or find stop word.
            if (sampleTok == '_END_'  or len(decodedsent.split()) >= (max_summary_len-1)):
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
            if((i!=0 and i!=self.targetWord['_START_']) and i!=self.targetWord['_END_']):
                newString += self.revTargetWordIndex[i]+' '
        return newString

    def seq2text(seq):
        newString=''
        for i in seq:
            if(i!=0):
                newString=newString+self.revSourcetWordIndex[i]+' '
        return newString
    



        
