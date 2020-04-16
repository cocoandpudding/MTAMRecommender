import keras
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn import dynamic_rnn
import tensorflow as tf

from Model.Modules.time_aware_rnn import TimeAwareGRUCell_sigmoid, TimeAwareGRUCell_decay_new


class GRU():

    def build_single_cell(self, hidden_units):
        ''' Create a single-layer RNNCell.

            Args:
              hidden_units: Units of RNNCell.

            Returns:
             An example of the single layer RNNCell
            '''
        #cell_type = TimeAwareGRUCell_sigmoid
        cell_type=GRUCell
        #cell_type = TimeAwareGRUCell_decay_new
        cell = cell_type(hidden_units)
        return cell

    def build_cell(self, hidden_units,
                   depth=1):
        '''Create forward and reverse RNNCell networks.

            Args:
              hidden_units: Units of RNNCell.
              depth: The number of RNNCell layers.

            Returns:
              An example of RNNCell
            '''
        cell_lists = [self.build_single_cell(hidden_units) for i in range(depth)]
        return MultiRNNCell(cell_lists)

    def bidirectional_gru_net(self, hidden_units,
                              input_data,
                              input_length):
        '''Create GRU net.

            Args:
              hidden_units: Units of RNN.
              input_data: Input of RNN.
              input_length: The length of input_data.

            Returns:
              Outputs, which are (output_fw, output_bw), are composed of tensor
              which outputs outward to cell and outward to cell
            '''
        cell_fw = self.build_cell(hidden_units)
        cell_bw = self.build_cell(hidden_units)
        outputs = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_data, input_length,
                                                  dtype=tf.float32)
        return outputs
    def gru_net(self,hidden_units,input_data,input_length):


        cell = self.build_cell(hidden_units)
        input_length = tf.reshape(input_length,[-1])
        outputs, _ = tf.nn.dynamic_rnn(cell,inputs=input_data,sequence_length=input_length,dtype=tf.float32)

        return outputs

    def time_aware_gru_net(self, hidden_units, input_data, input_length,type):
        if type =='T-SeqRec':
            cell = self.build_time_aware_gru_cell_sigmoid(hidden_units)
        elif type == 'new':
            cell = self.build_time_aware_gru_cell_new(hidden_units)
        #cell = self.build_cell(hidden_units)
        self.input_length = tf.reshape(input_length, [-1])
        outputs, _ = dynamic_rnn(cell, inputs=input_data, sequence_length=self.input_length, dtype=tf.float32)
        return outputs

    def build_time_aware_gru_cell_sigmoid(self, hidden_units):
        cell = TimeAwareGRUCell_sigmoid(hidden_units)
        return MultiRNNCell([cell])


    def build_time_aware_gru_cell_new(self, hidden_units):
        cell = TimeAwareGRUCell_decay_new(hidden_units)
        return MultiRNNCell([cell])


    def build_time_aware_gru_cell(self, hidden_units):
        ''' Create a single-layer RNNCell.

            Args:
              hidden_units: Units of RNNCell.

            Returns:
             An example of the single layer RNNCell
            '''
        cell_type = TimeAwareGRUCell_sigmoid
        cell = cell_type(hidden_units)
        return cell

    def gru_net_initial_time(self, hidden_units, input_data, initial_state, input_length, depth=1):
        multi_cell = self.build_time_aware_gru_cell_sigmoid(hidden_units)
        input_length = tf.reshape(input_length, [-1])
        output, state = tf.nn.dynamic_rnn(multi_cell, input_data, sequence_length=input_length,
                                          initial_state=initial_state, dtype=tf.float32)
        return output

    def gru_net_initial(self, hidden_units, input_data, initial_state, input_length, depth=1):
        cell_lists = [self.build_single_cell(hidden_units) for i in range(depth)]
        multi_cell = MultiRNNCell(cell_lists, state_is_tuple=False)
        input_length = tf.reshape(input_length, [-1])
        output, state = tf.nn.dynamic_rnn(multi_cell, input_data, sequence_length=input_length,
                                          initial_state=initial_state, dtype=tf.float32)
        return output
