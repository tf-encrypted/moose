# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import channels.secure_channel_pb2 as secure__channel__pb2


class SecureChannelStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetValue = channel.unary_unary(
                '/securechannel.SecureChannel/GetValue',
                request_serializer=secure__channel__pb2.KeyValue.SerializeToString,
                response_deserializer=secure__channel__pb2.Value.FromString,
                )
        self.AddValueToBuffer = channel.unary_unary(
                '/securechannel.SecureChannel/AddValueToBuffer',
                request_serializer=secure__channel__pb2.RemoteValue.SerializeToString,
                response_deserializer=secure__channel__pb2.Empty.FromString,
                )


class SecureChannelServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetValue(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def AddValueToBuffer(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SecureChannelServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetValue': grpc.unary_unary_rpc_method_handler(
                    servicer.GetValue,
                    request_deserializer=secure__channel__pb2.KeyValue.FromString,
                    response_serializer=secure__channel__pb2.Value.SerializeToString,
            ),
            'AddValueToBuffer': grpc.unary_unary_rpc_method_handler(
                    servicer.AddValueToBuffer,
                    request_deserializer=secure__channel__pb2.RemoteValue.FromString,
                    response_serializer=secure__channel__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'securechannel.SecureChannel', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class SecureChannel(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetValue(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/securechannel.SecureChannel/GetValue',
            secure__channel__pb2.KeyValue.SerializeToString,
            secure__channel__pb2.Value.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def AddValueToBuffer(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/securechannel.SecureChannel/AddValueToBuffer',
            secure__channel__pb2.RemoteValue.SerializeToString,
            secure__channel__pb2.Empty.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)
