'''
Created on Dec 13, 2016

@author: george
'''
from multiprocessing import Process, Queue

class QueuedFetcher(Process):
    '''
    classdocs
    '''


    def __init__(self, data_provider,queue_size = 30):
        '''
        Constructor
        '''
        super(QueuedFetcher, self).__init__()
        self.data_provider = data_provider
        self.queue = Queue(queue_size)
        
        
    def start(self):
        super(QueuedFetcher,self).start()   
        def cleanup():
                print ('Terminating Queued Fetcher')
                super().terminate()
                super().join()
                
        import atexit
        atexit.register(cleanup) 
        
    def run(self):
        print ('Queued Fetcher started')
        while True:
            data_batch = self.data_provider.get_next_minibatch()
            self.queue.put(data_batch)    