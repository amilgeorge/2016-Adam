'''
Created on Dec 13, 2016

@author: george
'''
from multiprocessing import Process, Queue
import multiprocessing

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
        self.exit = multiprocessing.Event()
        
        
    def start(self):
        super(QueuedFetcher,self).start()   
        def cleanup():
                print ('Terminating Queued Fetcher')
                super().terminate()
                super().join()
                
        import atexit
        atexit.register(cleanup) 

    def get_next(self):
        return self.queue.get(True)

    def shutdown(self):
        print("Shutdown initiated")
        self.exit.set()

    def run(self):
        print ('Queued Fetcher started')
        while not self.exit.is_set():
            #print('Queue size :{}'.format(self.queue.qsize()))
            data_batch = self.data_provider.next_mini_batch_sync()
            self.queue.put(data_batch)

        print('Queued Fetcher shutdown')