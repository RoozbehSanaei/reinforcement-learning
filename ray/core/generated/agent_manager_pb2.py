# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/ray/protobuf/agent_manager.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='src/ray/protobuf/agent_manager.proto',
  package='ray.rpc',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n$src/ray/protobuf/agent_manager.proto\x12\x07ray.rpc\"|\n\x14RegisterAgentRequest\x12\x1b\n\tagent_pid\x18\x01 \x01(\x05R\x08\x61gentPid\x12\x1d\n\nagent_port\x18\x02 \x01(\x05R\tagentPort\x12(\n\x10\x61gent_ip_address\x18\x03 \x01(\tR\x0e\x61gentIpAddress\"E\n\x12RegisterAgentReply\x12/\n\x06status\x18\x01 \x01(\x0e\x32\x17.ray.rpc.AgentRpcStatusR\x06status*F\n\x0e\x41gentRpcStatus\x12\x17\n\x13\x41GENT_RPC_STATUS_OK\x10\x00\x12\x1b\n\x17\x41GENT_RPC_STATUS_FAILED\x10\x01\x32\x62\n\x13\x41gentManagerService\x12K\n\rRegisterAgent\x12\x1d.ray.rpc.RegisterAgentRequest\x1a\x1b.ray.rpc.RegisterAgentReplyb\x06proto3'
)

_AGENTRPCSTATUS = _descriptor.EnumDescriptor(
  name='AgentRpcStatus',
  full_name='ray.rpc.AgentRpcStatus',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='AGENT_RPC_STATUS_OK', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='AGENT_RPC_STATUS_FAILED', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=246,
  serialized_end=316,
)
_sym_db.RegisterEnumDescriptor(_AGENTRPCSTATUS)

AgentRpcStatus = enum_type_wrapper.EnumTypeWrapper(_AGENTRPCSTATUS)
AGENT_RPC_STATUS_OK = 0
AGENT_RPC_STATUS_FAILED = 1



_REGISTERAGENTREQUEST = _descriptor.Descriptor(
  name='RegisterAgentRequest',
  full_name='ray.rpc.RegisterAgentRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='agent_pid', full_name='ray.rpc.RegisterAgentRequest.agent_pid', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, json_name='agentPid', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='agent_port', full_name='ray.rpc.RegisterAgentRequest.agent_port', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, json_name='agentPort', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='agent_ip_address', full_name='ray.rpc.RegisterAgentRequest.agent_ip_address', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, json_name='agentIpAddress', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=49,
  serialized_end=173,
)


_REGISTERAGENTREPLY = _descriptor.Descriptor(
  name='RegisterAgentReply',
  full_name='ray.rpc.RegisterAgentReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='ray.rpc.RegisterAgentReply.status', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, json_name='status', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=175,
  serialized_end=244,
)

_REGISTERAGENTREPLY.fields_by_name['status'].enum_type = _AGENTRPCSTATUS
DESCRIPTOR.message_types_by_name['RegisterAgentRequest'] = _REGISTERAGENTREQUEST
DESCRIPTOR.message_types_by_name['RegisterAgentReply'] = _REGISTERAGENTREPLY
DESCRIPTOR.enum_types_by_name['AgentRpcStatus'] = _AGENTRPCSTATUS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RegisterAgentRequest = _reflection.GeneratedProtocolMessageType('RegisterAgentRequest', (_message.Message,), {
  'DESCRIPTOR' : _REGISTERAGENTREQUEST,
  '__module__' : 'src.ray.protobuf.agent_manager_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.RegisterAgentRequest)
  })
_sym_db.RegisterMessage(RegisterAgentRequest)

RegisterAgentReply = _reflection.GeneratedProtocolMessageType('RegisterAgentReply', (_message.Message,), {
  'DESCRIPTOR' : _REGISTERAGENTREPLY,
  '__module__' : 'src.ray.protobuf.agent_manager_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.RegisterAgentReply)
  })
_sym_db.RegisterMessage(RegisterAgentReply)



_AGENTMANAGERSERVICE = _descriptor.ServiceDescriptor(
  name='AgentManagerService',
  full_name='ray.rpc.AgentManagerService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=318,
  serialized_end=416,
  methods=[
  _descriptor.MethodDescriptor(
    name='RegisterAgent',
    full_name='ray.rpc.AgentManagerService.RegisterAgent',
    index=0,
    containing_service=None,
    input_type=_REGISTERAGENTREQUEST,
    output_type=_REGISTERAGENTREPLY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_AGENTMANAGERSERVICE)

DESCRIPTOR.services_by_name['AgentManagerService'] = _AGENTMANAGERSERVICE

# @@protoc_insertion_point(module_scope)