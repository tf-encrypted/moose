import gmqtt

from moose.logger import get_logger


class ChannelManager:
    def __init__(self, broker_host, ident):
        self.receive_buffer = AsyncStore()
        self.ident = ident
        self.broker_host = broker_host
        self.client = None

    async def setup_client(self):

        def on_connect(client, flags, rc, properties):
            get_logger().debug("Connected")
            client.subscribe(f"{self.ident}/#", qos=1)

        async def on_message(client, topic, payload, qos, properties):
            get_logger().debug(f"Message received, topic:{topic}")
            ident, session_id, rendezvous_key = topic.split("/")
            assert ident == self.ident, ident
            key = (session_id, rendezvous_key)
            await self.receive_buffer.put(key, payload)
            return gmqtt.mqtt.constants.PubRecReasonCode.SUCCESS

        client = gmqtt.MQTTClient(self.ident)
        client.on_message = on_message
        client.on_connect = on_connect
        await client.connect(broker_host, port=1883, ssl=False, keepalive=60)
        return client

    def get_hostname(self, player_name):
        raise NotImplementedError()

    async def get_value(self, rendezvous_key, session_id):
        raise NotImplementedError()

    async def receive(self, op, session_id):
        if not self.client:
            # TODO potential race condition in multi-threaded scenario
            self.client = await self.setup_client()
        key = (session_id, op.rendezvous_key)
        await self.receive_buffer.get(key)

    async def send(self, value, op, session_id):
        if not self.client:
            # TODO potential race condition in multi-threaded scenario
            self.client = await self.setup_client()
        topic = f"{self.ident}/{session_id}/{op.rendezvous_key}"
        self.client.publish(topic, payload=value, qos=1, retain=True)
