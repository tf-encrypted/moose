# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: secure_channel.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='secure_channel.proto',
  package='securechannel',
  syntax='proto3',
  serialized_options=b'\n\017io.grpc.channelB\022SecureChannelProtoP\001\242\002\003RTG',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x14secure_channel.proto\x12\rsecurechannel\"H\n\x0bRemoteValue\x12\r\n\x05value\x18\x01 \x01(\x02\x12\x16\n\x0erendezvous_key\x18\x02 \x01(\t\x12\x12\n\nsession_id\x18\x03 \x01(\x03\"6\n\x08KeyValue\x12\x16\n\x0erendezvous_key\x18\x01 \x01(\t\x12\x12\n\nsession_id\x18\x02 \x01(\x03\"\x16\n\x05Value\x12\r\n\x05value\x18\x01 \x01(\x02\"\x07\n\x05\x45mpty2\x94\x01\n\rSecureChannel\x12;\n\x08GetValue\x12\x17.securechannel.KeyValue\x1a\x14.securechannel.Value\"\x00\x12\x46\n\x10\x41\x64\x64ValueToBuffer\x12\x1a.securechannel.RemoteValue\x1a\x14.securechannel.Empty\"\x00\x42-\n\x0fio.grpc.channelB\x12SecureChannelProtoP\x01\xa2\x02\x03RTGb\x06proto3'
)




_REMOTEVALUE = _descriptor.Descriptor(
  name='RemoteValue',
  full_name='securechannel.RemoteValue',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='securechannel.RemoteValue.value', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='rendezvous_key', full_name='securechannel.RemoteValue.rendezvous_key', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='session_id', full_name='securechannel.RemoteValue.session_id', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=39,
  serialized_end=111,
)


_KEYVALUE = _descriptor.Descriptor(
  name='KeyValue',
  full_name='securechannel.KeyValue',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='rendezvous_key', full_name='securechannel.KeyValue.rendezvous_key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='session_id', full_name='securechannel.KeyValue.session_id', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=113,
  serialized_end=167,
)


_VALUE = _descriptor.Descriptor(
  name='Value',
  full_name='securechannel.Value',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='securechannel.Value.value', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=169,
  serialized_end=191,
)


_EMPTY = _descriptor.Descriptor(
  name='Empty',
  full_name='securechannel.Empty',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=193,
  serialized_end=200,
)

DESCRIPTOR.message_types_by_name['RemoteValue'] = _REMOTEVALUE
DESCRIPTOR.message_types_by_name['KeyValue'] = _KEYVALUE
DESCRIPTOR.message_types_by_name['Value'] = _VALUE
DESCRIPTOR.message_types_by_name['Empty'] = _EMPTY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RemoteValue = _reflection.GeneratedProtocolMessageType('RemoteValue', (_message.Message,), {
  'DESCRIPTOR' : _REMOTEVALUE,
  '__module__' : 'secure_channel_pb2'
  # @@protoc_insertion_point(class_scope:securechannel.RemoteValue)
  })
_sym_db.RegisterMessage(RemoteValue)

KeyValue = _reflection.GeneratedProtocolMessageType('KeyValue', (_message.Message,), {
  'DESCRIPTOR' : _KEYVALUE,
  '__module__' : 'secure_channel_pb2'
  # @@protoc_insertion_point(class_scope:securechannel.KeyValue)
  })
_sym_db.RegisterMessage(KeyValue)

Value = _reflection.GeneratedProtocolMessageType('Value', (_message.Message,), {
  'DESCRIPTOR' : _VALUE,
  '__module__' : 'secure_channel_pb2'
  # @@protoc_insertion_point(class_scope:securechannel.Value)
  })
_sym_db.RegisterMessage(Value)

Empty = _reflection.GeneratedProtocolMessageType('Empty', (_message.Message,), {
  'DESCRIPTOR' : _EMPTY,
  '__module__' : 'secure_channel_pb2'
  # @@protoc_insertion_point(class_scope:securechannel.Empty)
  })
_sym_db.RegisterMessage(Empty)


DESCRIPTOR._options = None

_SECURECHANNEL = _descriptor.ServiceDescriptor(
  name='SecureChannel',
  full_name='securechannel.SecureChannel',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=203,
  serialized_end=351,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetValue',
    full_name='securechannel.SecureChannel.GetValue',
    index=0,
    containing_service=None,
    input_type=_KEYVALUE,
    output_type=_VALUE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='AddValueToBuffer',
    full_name='securechannel.SecureChannel.AddValueToBuffer',
    index=1,
    containing_service=None,
    input_type=_REMOTEVALUE,
    output_type=_EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_SECURECHANNEL)

DESCRIPTOR.services_by_name['SecureChannel'] = _SECURECHANNEL

# @@protoc_insertion_point(module_scope)
