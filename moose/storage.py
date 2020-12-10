import asyncio

from moose.logger import get_logger


class AsyncStore:
    def __init__(self, initial_values={}, loop=None):
        self.loop = loop or asyncio.get_event_loop()
        self.key_to_future = dict()
        self.future_to_key = dict()
        for key, value in initial_values.items():
            self.get_future(key).set_result(value)

    async def put(self, key, value):
        if key not in self.key_to_future:
            self._create_future(key)
        return self.key_to_future[key].set_result(value)

    async def get(self, key):
        if key not in self.key_to_future:
            self._create_future(key)
        return await self.key_to_future[key]

    def get_future(self, key):
        if key not in self.key_to_future:
            self._create_future(key)
        return self.key_to_future[key]

    def _create_future(self, key):
        future = self.loop.create_future()
        get_logger().debug(f"Future created: id:{id(future)}, key:{key}")
        self.key_to_future[key] = future
        self.future_to_key[future] = key
        future.add_done_callback(self._future_done_callback)

    def _future_done_callback(self, future):
        key = self.future_to_key[future]
        get_logger().debug(f"Future done: id:{id(future)}, key:{key}")
        # def self.key_to_future[key]
        del self.future_to_key[future]
