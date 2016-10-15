'''
Created on Oct 12, 2016

@author: george
'''
import pickle
import numpy as np

if __name__ == '__main__':

        
    with open('ends1.pickle','rb') as f3:  # Python 3: open(..., 'rb')
        dict1 = pickle.load(f3)
        
    with open('ends2.pickle','rb') as f4:  # Python 3: open(..., 'rb')
        dict2 = pickle.load(f4)
    
    with open('bnorm1.pickle','rb') as f:  # Python 3: open(..., 'rb')
        m1,v1 = pickle.load(f)
        
    with open('bnorm2.pickle','rb') as f:  # Python 3: open(..., 'rb')
        m2,v2 = pickle.load(f)
        
    
    print (np.amax(m1-m2)) 
    print (np.amax(v1-v2))     
    print(dict1.keys())
    a1 = dict1['resnet_v1_50/conv1']
    a2 = dict2['resnet_v1_50/conv1']
    
    b=a1-a2
    mb = np.amax(b)
    
    for key in dict1.keys():
        obj1 = dict1[key]
        obj2 = dict2[key]
        diff = obj1 - obj2
        if np.amax(diff) !=0:
            i=i+1
            print (np.amax(diff))
            print (key)
            
    
    print ('Total = {}'.format(i))
    
    