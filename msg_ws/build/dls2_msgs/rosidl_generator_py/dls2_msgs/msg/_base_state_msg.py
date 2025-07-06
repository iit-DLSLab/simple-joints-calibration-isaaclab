# generated from rosidl_generator_py/resource/_idl.py.em
# with input from dls2_msgs:msg/BaseStateMsg.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

# Member 'position'
# Member 'orientation'
# Member 'linear_velocity'
# Member 'angular_velocity'
# Member 'linear_acceleration'
# Member 'angular_acceleration'
# Member 'stance_status'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_BaseStateMsg(type):
    """Metaclass of message 'BaseStateMsg'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('dls2_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'dls2_msgs.msg.BaseStateMsg')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__base_state_msg
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__base_state_msg
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__base_state_msg
            cls._TYPE_SUPPORT = module.type_support_msg__msg__base_state_msg
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__base_state_msg

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class BaseStateMsg(metaclass=Metaclass_BaseStateMsg):
    """Message class 'BaseStateMsg'."""

    __slots__ = [
        '_frame_id',
        '_sequence_id',
        '_timestamp',
        '_robot_name',
        '_position',
        '_orientation',
        '_linear_velocity',
        '_angular_velocity',
        '_linear_acceleration',
        '_angular_acceleration',
        '_stance_status',
    ]

    _fields_and_field_types = {
        'frame_id': 'string',
        'sequence_id': 'uint32',
        'timestamp': 'double',
        'robot_name': 'string',
        'position': 'double[3]',
        'orientation': 'double[4]',
        'linear_velocity': 'double[3]',
        'angular_velocity': 'double[3]',
        'linear_acceleration': 'double[3]',
        'angular_acceleration': 'double[3]',
        'stance_status': 'double[4]',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('uint32'),  # noqa: E501
        rosidl_parser.definition.BasicType('double'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 3),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 4),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 3),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 3),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 3),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 3),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('double'), 4),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.frame_id = kwargs.get('frame_id', str())
        self.sequence_id = kwargs.get('sequence_id', int())
        self.timestamp = kwargs.get('timestamp', float())
        self.robot_name = kwargs.get('robot_name', str())
        if 'position' not in kwargs:
            self.position = numpy.zeros(3, dtype=numpy.float64)
        else:
            self.position = numpy.array(kwargs.get('position'), dtype=numpy.float64)
            assert self.position.shape == (3, )
        if 'orientation' not in kwargs:
            self.orientation = numpy.zeros(4, dtype=numpy.float64)
        else:
            self.orientation = numpy.array(kwargs.get('orientation'), dtype=numpy.float64)
            assert self.orientation.shape == (4, )
        if 'linear_velocity' not in kwargs:
            self.linear_velocity = numpy.zeros(3, dtype=numpy.float64)
        else:
            self.linear_velocity = numpy.array(kwargs.get('linear_velocity'), dtype=numpy.float64)
            assert self.linear_velocity.shape == (3, )
        if 'angular_velocity' not in kwargs:
            self.angular_velocity = numpy.zeros(3, dtype=numpy.float64)
        else:
            self.angular_velocity = numpy.array(kwargs.get('angular_velocity'), dtype=numpy.float64)
            assert self.angular_velocity.shape == (3, )
        if 'linear_acceleration' not in kwargs:
            self.linear_acceleration = numpy.zeros(3, dtype=numpy.float64)
        else:
            self.linear_acceleration = numpy.array(kwargs.get('linear_acceleration'), dtype=numpy.float64)
            assert self.linear_acceleration.shape == (3, )
        if 'angular_acceleration' not in kwargs:
            self.angular_acceleration = numpy.zeros(3, dtype=numpy.float64)
        else:
            self.angular_acceleration = numpy.array(kwargs.get('angular_acceleration'), dtype=numpy.float64)
            assert self.angular_acceleration.shape == (3, )
        if 'stance_status' not in kwargs:
            self.stance_status = numpy.zeros(4, dtype=numpy.float64)
        else:
            self.stance_status = numpy.array(kwargs.get('stance_status'), dtype=numpy.float64)
            assert self.stance_status.shape == (4, )

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.frame_id != other.frame_id:
            return False
        if self.sequence_id != other.sequence_id:
            return False
        if self.timestamp != other.timestamp:
            return False
        if self.robot_name != other.robot_name:
            return False
        if all(self.position != other.position):
            return False
        if all(self.orientation != other.orientation):
            return False
        if all(self.linear_velocity != other.linear_velocity):
            return False
        if all(self.angular_velocity != other.angular_velocity):
            return False
        if all(self.linear_acceleration != other.linear_acceleration):
            return False
        if all(self.angular_acceleration != other.angular_acceleration):
            return False
        if all(self.stance_status != other.stance_status):
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def frame_id(self):
        """Message field 'frame_id'."""
        return self._frame_id

    @frame_id.setter
    def frame_id(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'frame_id' field must be of type 'str'"
        self._frame_id = value

    @builtins.property
    def sequence_id(self):
        """Message field 'sequence_id'."""
        return self._sequence_id

    @sequence_id.setter
    def sequence_id(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'sequence_id' field must be of type 'int'"
            assert value >= 0 and value < 4294967296, \
                "The 'sequence_id' field must be an unsigned integer in [0, 4294967295]"
        self._sequence_id = value

    @builtins.property
    def timestamp(self):
        """Message field 'timestamp'."""
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'timestamp' field must be of type 'float'"
            assert not (value < -1.7976931348623157e+308 or value > 1.7976931348623157e+308) or math.isinf(value), \
                "The 'timestamp' field must be a double in [-1.7976931348623157e+308, 1.7976931348623157e+308]"
        self._timestamp = value

    @builtins.property
    def robot_name(self):
        """Message field 'robot_name'."""
        return self._robot_name

    @robot_name.setter
    def robot_name(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'robot_name' field must be of type 'str'"
        self._robot_name = value

    @builtins.property
    def position(self):
        """Message field 'position'."""
        return self._position

    @position.setter
    def position(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'position' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 3, \
                "The 'position' numpy.ndarray() must have a size of 3"
            self._position = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 3 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'position' field must be a set or sequence with length 3 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._position = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def orientation(self):
        """Message field 'orientation'."""
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'orientation' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 4, \
                "The 'orientation' numpy.ndarray() must have a size of 4"
            self._orientation = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 4 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'orientation' field must be a set or sequence with length 4 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._orientation = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def linear_velocity(self):
        """Message field 'linear_velocity'."""
        return self._linear_velocity

    @linear_velocity.setter
    def linear_velocity(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'linear_velocity' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 3, \
                "The 'linear_velocity' numpy.ndarray() must have a size of 3"
            self._linear_velocity = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 3 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'linear_velocity' field must be a set or sequence with length 3 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._linear_velocity = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def angular_velocity(self):
        """Message field 'angular_velocity'."""
        return self._angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'angular_velocity' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 3, \
                "The 'angular_velocity' numpy.ndarray() must have a size of 3"
            self._angular_velocity = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 3 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'angular_velocity' field must be a set or sequence with length 3 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._angular_velocity = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def linear_acceleration(self):
        """Message field 'linear_acceleration'."""
        return self._linear_acceleration

    @linear_acceleration.setter
    def linear_acceleration(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'linear_acceleration' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 3, \
                "The 'linear_acceleration' numpy.ndarray() must have a size of 3"
            self._linear_acceleration = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 3 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'linear_acceleration' field must be a set or sequence with length 3 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._linear_acceleration = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def angular_acceleration(self):
        """Message field 'angular_acceleration'."""
        return self._angular_acceleration

    @angular_acceleration.setter
    def angular_acceleration(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'angular_acceleration' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 3, \
                "The 'angular_acceleration' numpy.ndarray() must have a size of 3"
            self._angular_acceleration = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 3 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'angular_acceleration' field must be a set or sequence with length 3 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._angular_acceleration = numpy.array(value, dtype=numpy.float64)

    @builtins.property
    def stance_status(self):
        """Message field 'stance_status'."""
        return self._stance_status

    @stance_status.setter
    def stance_status(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float64, \
                "The 'stance_status' numpy.ndarray() must have the dtype of 'numpy.float64'"
            assert value.size == 4, \
                "The 'stance_status' numpy.ndarray() must have a size of 4"
            self._stance_status = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 4 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -1.7976931348623157e+308 or val > 1.7976931348623157e+308) or math.isinf(val) for val in value)), \
                "The 'stance_status' field must be a set or sequence with length 4 and each value of type 'float' and each double in [-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000, 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000]"
        self._stance_status = numpy.array(value, dtype=numpy.float64)
