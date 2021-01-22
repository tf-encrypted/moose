import numpy

from moose.storage.base import DataStore


class MemoryDataStore(DataStore):
    def __init__(self, initial_store={}):
        self.store = initial_store

    async def load(self, session_id, key):
        x = numpy.loadtxt(key)
        print('SHAPE!!', x.shape)
        x = x.reshape(10,1)
        return x
        # return self.store[key]

    async def save(self, session_id, key, value):
        print('SAVE!!', key, value, value.shape)
        numpy.savetxt(f'{session_id}-{key}', value)
        # # with open(key, 'w') as f:
        # #     f.write(value)
        #
        # self.store[key] = value
