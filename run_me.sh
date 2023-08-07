VISIBLE_CUDA_DEVICES=1

# Train the agents
python train_MB.py -learningRate 0.005 -mHiddenSize 50 -cHiddenSize 50 -batchSize 2000 \
                -imgFeatSize 20 -embedSize 20\
                -numEpochs 1000\
		-decay "log"\
                -mOutVocab 3 -mActFreq 2 -cOutVocab 3 -numRounds 12 -numVitalParams 6 -ix 0 #-useGPU
