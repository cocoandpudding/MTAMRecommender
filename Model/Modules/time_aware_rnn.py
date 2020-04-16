from keras import activations, initializers
import tensorflow as tf
from keras.activations import sigmoid
#from tensorflow.python.context.engine import input_spec
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell, _check_supported_dtypes,_hasattr
from tensorflow.python.eager import context
from tensorflow.python.ops import math_ops, init_ops, variable_scope, array_ops, nn_ops
from tensorflow.python.ops.rnn_cell_impl import RNNCell, GRUCell,_zero_state_tensors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import ops


_BIAS_VARIABLE_NAME = 'bias'
_WEIGHTS_VARIABLE_NAME = 'kernel'


class TimeAwareGRUCell_sigmoid(GRUCell):
    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initialier=None,
                 bias_initializer=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(GRUCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        _check_supported_dtypes(self.dtype)

        if context.executing_eagerly() and context.num_gpus()> 0:
            logging.warn("This is not optimized for performance.")
        self.input_spec = input_spec.InputSpec(ndim = 2)
        self._num_units = num_units
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh
        self._kernel_initializer = initializers.get(kernel_initialier)
        self._bias_initializer = initializers.get(bias_initializer)


    def build(self, inputs_shape):
            if inputs_shape[-1] is None:
                raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                                 str(inputs_shape))
            _check_supported_dtypes(self.dtype)
            input_depth = inputs_shape[-1]-2
            self._gate_kernel = self.add_variable(
                "gates/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth + self._num_units, 2 * self._num_units],
                initializer=self._kernel_initializer)
            self._gate_bias = self.add_variable(
                "gates/%s" % _BIAS_VARIABLE_NAME,
                shape=[2 * self._num_units],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.constant_initializer(1.0, dtype=self.dtype)))
            self._candidate_kernel = self.add_variable(
                "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth + self._num_units, self._num_units],
                initializer=self._kernel_initializer)
            self._candidate_bias = self.add_variable(
                "candidate/%s" % _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.zeros_initializer(dtype=self.dtype)))

    def call(self, inputs, state):
        dtype = inputs.dtype
        time_now_score = tf.expand_dims(inputs[:, -1], -1)
        time_last_score = tf.expand_dims(inputs[:, -2], -1)
        inputs = inputs[:, :-2]
        input_size = inputs.get_shape().with_rank(2)[1]
        # decay gates
        scope = variable_scope.get_variable_scope()
        with variable_scope.variable_scope(scope) as unit_scope:
            with variable_scope.variable_scope(unit_scope):
                self._time_input_w1 = variable_scope.get_variable(
                    "_time_input_w1", shape=[self._num_units], dtype=dtype)
                self._time_input_bias1 = variable_scope.get_variable(
                    "_time_input_bias1", shape=[self._num_units], dtype=dtype)
                self._time_input_w2 = variable_scope.get_variable(
                    "_time_input_w2", shape=[self._num_units], dtype=dtype)
                self._time_input_bias2 = variable_scope.get_variable(
                    "_time_input_bias2", shape=[self._num_units], dtype=dtype)
                self._time_kernel_w1 = variable_scope.get_variable(
                    "_time_kernel_w1", shape=[input_size, self._num_units], dtype=dtype)
                self._time_kernel_t1 = variable_scope.get_variable(
                    "_time_kernel_t1", shape=[self._num_units, self._num_units], dtype=dtype)
                self._time_bias1 = variable_scope.get_variable(
                    "_time_bias1", shape=[self._num_units], dtype=dtype)
                self._time_kernel_w2 = variable_scope.get_variable(
                    "_time_kernel_w2", shape=[input_size, self._num_units], dtype=dtype)
                self._time_kernel_t2 = variable_scope.get_variable(
                    "_time_kernel_t2", shape=[self._num_units, self._num_units], dtype=dtype)
                self._time_bias2 = variable_scope.get_variable(
                    "_time_bias2", shape=[self._num_units], dtype=dtype)
                #self._o_kernel_t1 = variable_scope.get_variable(
                    #"_o_kernel_t1", shape=[self._num_units, self._num_units], dtype=dtype)
                #self._o_kernel_t2 = variable_scope.get_variable(
                    #"_o_kernel_t2", shape=[self._num_units, self._num_units], dtype=dtype)
        #time_now_input = tf.nn.tanh(tf.log(1+time_now_score) * self._time_input_w1 + self._time_input_bias1)
        #time_last_input = tf.nn.tanh(tf.log(1+time_last_score) * self._time_input_w2 + self._time_input_bias2)
        time_now_input = tf.nn.tanh(time_now_score * self._time_input_w1 + self._time_input_bias1)
        time_last_input = tf.nn.tanh(time_last_score * self._time_input_w2 + self._time_input_bias2)

        time_now_state = math_ops.matmul(inputs, self._time_kernel_w1) + \
                         math_ops.matmul(time_now_input,self._time_kernel_t1) + self._time_bias1
        time_last_state = math_ops.matmul(inputs, self._time_kernel_w2) + \
                          math_ops.matmul(time_last_input,self._time_kernel_t2) + self._time_bias2

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        #new_h = u * state * sigmoid(time_last_state) + (1 - u) * c * sigmoid(time_now_state)
        new_h = u * state * sigmoid(time_now_state) + (1 - u) * c * sigmoid(time_last_state)
        return new_h, new_h



class TimeAwareGRUCell_decay_new(GRUCell):
    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initialier=None,
                 bias_initializer=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(GRUCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)
        _check_supported_dtypes(self.dtype)

        if context.executing_eagerly() and context.num_gpus()> 0:
            logging.warn("This is not optimized for performance.")
        self.input_spec = input_spec.InputSpec(ndim = 2)
        self._num_units = num_units
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh
        self._kernel_initializer = initializers.get(kernel_initialier)
        self._bias_initializer = initializers.get(bias_initializer)


    def build(self, inputs_shape):
            if inputs_shape[-1] is None:
                raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" %
                                 str(inputs_shape))
            _check_supported_dtypes(self.dtype)
            input_depth = inputs_shape[-1]-2
            self._gate_kernel = self.add_variable(
                "gates/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth + self._num_units, 2 * self._num_units],
                initializer=self._kernel_initializer)
            self._gate_bias = self.add_variable(
                "gates/%s" % _BIAS_VARIABLE_NAME,
                shape=[2 * self._num_units],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.constant_initializer(1.0, dtype=self.dtype)))
            self._candidate_kernel = self.add_variable(
                "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
                shape=[input_depth + self._num_units, self._num_units],
                initializer=self._kernel_initializer)
            self._candidate_bias = self.add_variable(
                "candidate/%s" % _BIAS_VARIABLE_NAME,
                shape=[self._num_units],
                initializer=(self._bias_initializer
                             if self._bias_initializer is not None else
                             init_ops.zeros_initializer(dtype=self.dtype)))
    def call(self, inputs, state):
        dtype = inputs.dtype
        time_now_score = tf.expand_dims(inputs[:, -1], -1)
        time_last_score = tf.expand_dims(inputs[:, -2], -1)
        inputs = inputs[:, :-2]
        input_size = inputs.get_shape().with_rank(2)[1]
        # decay gates
        scope = variable_scope.get_variable_scope()
        with variable_scope.variable_scope(scope) as unit_scope:
            with variable_scope.variable_scope(unit_scope):
                #weights for time now
                self._time_kernel_w1 = variable_scope.get_variable(
                    "_time_kernel_w1", shape=[self._num_units], dtype=dtype)
                self._time_kernel_b1 = variable_scope.get_variable(
                    "_time_kernel_b1", shape=[self._num_units], dtype=dtype)
                self._time_history_w1 =variable_scope.get_variable(
                    "_time_history_w1", shape=[self._num_units], dtype=dtype)
                self._time_history_b1 =variable_scope.get_variable(
                    "_time_history_b1", shape=[self._num_units], dtype=dtype)
                self._time_w1 = variable_scope.get_variable(
                    "_time_w1", shape=[self._num_units], dtype=dtype)
                self._time_w12 = variable_scope.get_variable(
                    "_time_w12", shape=[self._num_units], dtype=dtype)
                self._time_b1 = variable_scope.get_variable(
                    "_time_b1", shape=[self._num_units], dtype=dtype)
                self._time_b12 = variable_scope.get_variable(
                    "_time_b12", shape=[self._num_units], dtype=dtype)
                #weight for time last
                self._time_kernel_w2 = variable_scope.get_variable(
                    "_time_kernel_w2", shape=[self._num_units], dtype=dtype)
                self._time_kernel_b2 = variable_scope.get_variable(
                    "_time_kernel_b2", shape=[self._num_units], dtype=dtype)
                self._time_history_w2 =variable_scope.get_variable(
                    "_time_history_w2", shape=[self._num_units], dtype=dtype)
                self._time_history_b2 =variable_scope.get_variable(
                    "_time_history_b2", shape=[self._num_units], dtype=dtype)
                self._time_w2 = variable_scope.get_variable(
                    "_time_w2", shape=[self._num_units], dtype=dtype)
                self._time_b2 = variable_scope.get_variable(
                    "_time_b2", shape=[self._num_units], dtype=dtype)

        #time_now_weight = tf.nn.relu( inputs * self._time_kernel_w1+self._time_kernel_b1)
        time_last_weight = tf.nn.relu(inputs * self._time_kernel_w1 + self._time_kernel_b1+state * self._time_history_w1)
        #time_now_state = tf.sigmoid( time_now_weight+ self._time_w1*tf.log(time_now_score+1)+self._time_b12)

        #time_last_weight =  tf.nn.relu(inputs* self._time_kernel_w2+self._time_kernel_b2 +state * self._time_history_w2)
        #time_last_state = tf.sigmoid( time_last_weight+ self._time_w2*tf.log(time_last_score+1)+self._time_b2)

        #version 2
        #time_last_score =  tf.nn.relu(self._time_w1 * tf.log(time_last_score + 1) + self._time_b1)
        time_last_score = tf.nn.relu(self._time_w1 * time_last_score+ self._time_b1)
        time_last_state = tf.sigmoid(self._time_kernel_w2*time_last_weight+self._time_w12*time_last_score+self._time_b12)
        #time_last_score = tf.nn.relu(self._time_w2 * tf.log(time_last_score + 1) + self._time_b2)




        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        #time_last_weight = tf.nn.relu(inputs * self._time_kernel_w2 + self._time_kernel_b2 + state * self._time_history_w2)
        #time_last_state = tf.sigmoid(time_last_weight + self._time_w2 * tf.log(time_last_score + 1) + self._time_b2)
        #new_h = u * state *  time_now_state + (1 - u) * c * time_last_state 0.0185 0.0136
        #new_h = u * state + (1 - u) * c * time_now_state 0.0237 0.013
        #new_h = u * state * time_now_state + (1 - u) * c * time_last_state #no position 0.0211 0.0137
        #new_h = u * state + (1 - u) * c * time_now_state #no position 0.0211 0.0143
        #new_h = u * state + (1 - u) * c 0.0185 0.0138
        #####
        #sli_rec no position 0.026 0.0144
        #####
        #new_h = u * state + (1 - u) * c * time_last_state #0.0237 0.0157
        new_h = u * state + (1 - u) * c * time_last_state
        return new_h, new_h









