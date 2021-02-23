import abc


class DataStore:
    @abc.abstractmethod
    async def load(self, session_id, key, optional_arguments):
        pass

    @abc.abstractmethod
    async def save(self, session_id, key, value):
        pass
