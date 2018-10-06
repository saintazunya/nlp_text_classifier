import numpy as np
def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print( "Done.",len(model)," words loaded!")
    return model

if __name__=='__main__':
    loadGloveModel('/home/zcf/Downloads/glove.840B.300d.txt')

