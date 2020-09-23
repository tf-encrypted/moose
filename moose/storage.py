import asyncio
from collections import defaultdict


class AsyncStore:
  def __init__(self, loop=None):
    loop = loop or asyncio.get_event_loop()
    self.values = defaultdict(loop.create_future)

  async def put(self, key, value):
    return self.values[key].set_result(value)

  async def get(self, key):
    return await self.values[key]

  def get_future(self, key):
    return self.values[key]
