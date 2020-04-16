import tensorflow as tf
from tensorflow.python.ops import variable_scope


class Time_Aware_Attention():

    def normalize(self,inputs,
                  epsilon=1e-8,
                  scope="ln",
                  reuse=None):
        '''Applies layer normalization.

        Args:
          inputs: A tensor with 2 or more dimensions, where the first dimension has
          `batch_size`.
          epsilon: A floating number. A very small number for preventing ZeroDivision Error.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
          by the same name.

        Returns:
          A tensor with the same shape and data dtype as `inputs`.
        '''
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs


    def feedforward(self,inputs,
                    num_units=[2048, 512],
                    scope="feedforward",
                    reuse=None):
        '''Point-wise feed forward net.

        Args:
          inputs: A 3d tensor with shape of [N, T, C].
          num_units: A list of two integers.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
          by the same name.

        Returns:
          A 3d tensor with the same shape and dtype as inputs
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)

            # Residual connection
            outputs += inputs

            # Normalize
            outputs = self.normalize(outputs)

        return outputs


    def TiSAS_multihead_attention(self,queries,
                            keys,
                            key_length,
                            query_length,
                            t_querys,
                            t_keys,
                            t_querys_length,
                            t_keys_length,
                            num_units=None,
                            num_heads=8,
                            dropout_rate=0,
                            is_training=True,
                            scope="multihead_attention",
                            reuse=None,
                            ):
        '''Applies multihead attention.

        Args:
          queries: A 3d tensor with shape of [N, T_q, C_q].
          queries_length: A 1d tensor with shape of [N].
          keys: A 3d tensor with shape of [N, T_k, C_k].
          keys_length:  A 1d tensor with shape of [N].
          num_units: A scalar. Attention size.
          dropout_rate: A floating point number.
          is_training: Boolean. Controller of mechanism for dropout.
          num_heads: An int. Number of heads.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
          by the same name.

        Returns
          A 3d tensor with shape of (N, T_q, C)
        '''
        t_querys = tf.stack([t_querys] * t_keys_length, axis=2)
        t_keys = tf.stack([t_keys] * t_querys_length, axis=1)

        # decay = tf.relu(time_w * tf.log((t_querys - tf.transpose(t_keys))+1)+time_b)
        interval = tf.log(tf.add(tf.abs(tf.subtract(t_querys, t_keys)), 1))

        # Linear projections, C = # dim or column, T_x = # vectors or actions
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        #Q = tf.layers.dropout(Q, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        #K = tf.layers.dropout(K, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        #V = tf.layers.dropout(V, rate=dropout_rate, training=tf.convert_to_tensor(is_training))


        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            interval_ =  tf.concat([interval]*num_heads, axis=0)  # (h*N, T_k, C/h)
            #decay_gate_ = tf.layers.dropout(decay_gate_, rate=dropout_rate,
                                      #training=tf.convert_to_tensor(is_training))

            # Multiplication
            # query-key score matrix
            # each big score matrix is then split into h score matrix with same size
            # w.r.t. different part of the feature
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]),  name ='3')  # (h*N, T_q, T_k)
            outputs += interval_

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            #key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))# (N, T_k)
            key_masks = tf.sequence_mask(key_length, tf.shape(keys)[1])  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            #outputs = tf.where(tf.equal(key_masks, 0), outputs, paddings)  # (h*N, T_q, T_k)
            outputs = tf.where(key_masks, outputs, paddings)  # (h*N, T_q, T_k)

            # Causality = Future blinding: No use, removed

            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
            '''
            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)

            # Attention vector
            att_vec = outputs

            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

            # Weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            outputs += queries

            # Normalize
            outputs = self.normalize(outputs)  # (N, T_q, C)
            '''

            # Query Masking
            #query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.sequence_mask(query_length, tf.shape(queries)[1], dtype=tf.float32)  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)
            print(outputs.shape.as_list())
            print(query_masks.shape.as_list())

            # Attention vector
            #########Tom Sun
            att_vec = outputs

            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

            # Weighted sum
            outputs = tf.matmul(outputs, V_, name = '4')# ( h*N, T_q, C/h)


            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
            outputs = outputs

            # Residual connection
            outputs += queries

            # Normalize
            outputs = self.normalize(outputs)  # (N, T_q, C)

        return outputs, att_vec
    def time_aware_multihead_attention(self,queries,
                            keys,
                            key_length,
                            query_length,
                            t_querys,
                            t_keys,
                            t_querys_length,
                            t_keys_length,
                            num_units=None,
                            num_heads=8,
                            dropout_rate=0,
                            is_training=True,
                            scope="multihead_attention",
                            reuse=None,
                            ):
        '''Applies multihead attention.

        Args:
          queries: A 3d tensor with shape of [N, T_q, C_q].
          queries_length: A 1d tensor with shape of [N].
          keys: A 3d tensor with shape of [N, T_k, C_k].
          keys_length:  A 1d tensor with shape of [N].
          num_units: A scalar. Attention size.
          dropout_rate: A floating point number.
          is_training: Boolean. Controller of mechanism for dropout.
          num_heads: An int. Number of heads.
          scope: Optional scope for `variable_scope`.
          reuse: Boolean, whether to reuse the weights of a previous layer
          by the same name.

        Returns
          A 3d tensor with shape of (N, T_q, C)
        '''
        # Linear projections, C = # dim or column, T_x = # vectors or actions
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        #Q = tf.layers.dropout(Q, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        #K = tf.layers.dropout(K, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        #V = tf.layers.dropout(V, rate=dropout_rate, training=tf.convert_to_tensor(is_training))


        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]
            #list = t_querys.get_shape().as_list()
            #query_len = queries.get_shape().as_list()[-2]
            #key_len = queries.get_shape().as_list()[-2]

            # time decay gate
            scope = variable_scope.get_variable_scope()
            with variable_scope.variable_scope(scope,reuse=None) as unit_scope:
                with variable_scope.variable_scope(unit_scope):
                    time_input_w = variable_scope.get_variable("_time_input_w",
                                                               shape=[num_units,num_units],
                                                               dtype=queries.dtype)
                    '''
                    time_input_b = variable_scope.get_variable("_time_input_b",
                                                                shape=[t_querys_length, t_keys_length],
                                                                dtype=queries.dtype)
                    time_input_w1 = variable_scope.get_variable("_time_input_w1",
                                                               shape=[t_querys_length, t_keys_length],
                                                               dtype=queries.dtype)
                    time_input_b1 = variable_scope.get_variable("_time_input_b1",
                                                                shape=[t_querys_length, t_keys_length],
                                                                dtype=queries.dtype)
                    time_output_w1 = variable_scope.get_variable("time_output_w1",
                                                               shape=[t_querys_length, t_keys_length],
                                                               dtype=queries.dtype)
                    time_output_w2 = variable_scope.get_variable("time_output_w2",
                                                                 shape=[t_querys_length, t_keys_length],
                                                                 dtype=queries.dtype)
                    time_output_b = variable_scope.get_variable("time_output_b",
                                                               shape=[t_querys_length, t_keys_length],
                                                               dtype=queries.dtype)
                    '''
                    #time_input_b = variable_scope.get_variable("_time_input_b",
                                                               #shape=[t_querys_length, t_keys_length],
                                                               #dtype=queries.dtype)
                    time_input_w1 = variable_scope.get_variable("_time_input_w1",
                                                                shape=[t_querys_length, t_keys_length],
                                                                dtype=queries.dtype)
                    time_input_b1 = variable_scope.get_variable("_time_input_b1",
                                                                shape=[t_querys_length, t_keys_length],
                                                                dtype=queries.dtype)
                    time_output_w1 = variable_scope.get_variable("time_output_w1",
                                                                 shape=[t_querys_length, t_keys_length],
                                                                 dtype=queries.dtype)
                    time_output_w2 = variable_scope.get_variable("time_output_w2",
                                                                 shape=[t_querys_length, t_keys_length],
                                                                 dtype=queries.dtype)
                    time_output_w3 = variable_scope.get_variable("time_output_w3",
                                                                 shape=[t_querys_length, t_keys_length],
                                                                 dtype=queries.dtype)
                    time_output_b = variable_scope.get_variable("time_output_b",
                                                                shape=[t_querys_length, t_keys_length],
                                                                dtype=queries.dtype)
                    #time_w = variable_scope.get_variable(
                        #"_time_w", shape=[query_len, key_len], dtype=queries.dtype)
                    #time_b = variable_scope.get_variable(
                        #"_time_b", shape=[query_len, key_len], dtype=queries.dtype)
                    #time_b2 = variable_scope.get_variable(
                       # "_time_b2", shape=[query_len, key_len], dtype=queries.dtype)

            time_query_key = tf.matmul(queries,time_input_w, name ='1')
            time_query_key = tf.matmul(time_query_key,keys, transpose_b=True ,name ='2')
            #time_query_key = tf.nn.tanh(time_query_key+time_input_b)
            time_query_key = tf.nn.tanh(time_query_key)
            #time_query_key = tf.layers.dropout(time_query_key, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

            '''
            t_querys = tf.expand_dims(t_querys,2 )
            t_querys = tf.concat([t_querys] * t_keys_length, axis=2)
            '''
            t_querys = tf.stack([t_querys] * t_keys_length, axis=2)

            '''
            t_keys = tf.expand_dims(t_keys, 1)
            t_keys = tf.concat([t_keys] * t_querys_length, axis=1)
            '''
            t_keys = tf.stack([t_keys] * t_querys_length, axis=1)

            #decay = tf.relu(time_w * tf.log((t_querys - tf.transpose(t_keys))+1)+time_b)
            decay = tf.log(tf.add(tf.abs(tf.subtract(t_querys,t_keys)),1))
            #decay_mean = tf.reduce_sum(decay)/(t_keys_length*t_querys_length)
            #decay = decay/(decay_mean+1)
            #decay = self.normalize(decay)
            decay = tf.nn.tanh(decay * time_input_w1 + time_input_b1)
            #decay = tf.nn.tanh(decay * time_input_w1)


            #decay_gate = time_output_w1 * decay * time_query_key + time_output_b 1
            #decay_gate = time_output_w1 * decay + time_output_b 1
            # 3
            decay_gate = time_output_w1 * decay + time_output_w2 * time_query_key + time_output_b

            #decay_gate = tf.sigmoid(time_output_w1*decay*time_query_key+time_output_b)
            #decay_gate = tf.exp(-time_query_key * decay)
            #sigmoid -> exp decay 0.145 0.067
            #relu sigmoid 0.150 0.729
            #relu ->exp decay 0.1423 0.0676
            #relu-> sigmoid + 0.156
            #relu-> sigmoid + split
            #relu sigmoid time_output_w1*decay+time_output_w2*time_query_key+time_output_b
            #0.50 0.68







            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            decay_gate_ =  tf.concat([decay_gate]*num_heads, axis=0)  # (h*N, T_k, C/h)
            #decay_gate_ = tf.layers.dropout(decay_gate_, rate=dropout_rate,
                                      #training=tf.convert_to_tensor(is_training))

            # Multiplication
            # query-key score matrix
            # each big score matrix is then split into h score matrix with same size
            # w.r.t. different part of the feature
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]),  name ='3')  # (h*N, T_q, T_k)
            outputs *= tf.nn.sigmoid(decay_gate_)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            #key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))# (N, T_k)
            key_masks = tf.sequence_mask(key_length, tf.shape(keys)[1])  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            #outputs = tf.where(tf.equal(key_masks, 0), outputs, paddings)  # (h*N, T_q, T_k)
            outputs = tf.where(key_masks, outputs, paddings)  # (h*N, T_q, T_k)

            # Causality = Future blinding: No use, removed

            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
            '''
            # Query Masking
            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)

            # Attention vector
            att_vec = outputs

            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

            # Weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            outputs += queries

            # Normalize
            outputs = self.normalize(outputs)  # (N, T_q, C)
            '''

            # Query Masking
            #query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
            query_masks = tf.sequence_mask(query_length, tf.shape(queries)[1], dtype=tf.float32)  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)
            print(outputs.shape.as_list())
            print(query_masks.shape.as_list())

            # Attention vector
            #########Tom Sun
            att_vec = outputs

            # Dropouts
            #outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

            # Weighted sum
            outputs = tf.matmul(outputs, V_, name = '4')# ( h*N, T_q, C/h)


            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
            outputs = outputs

            # Residual connection
            outputs += queries

            # Normalize
            outputs = self.normalize(outputs)  # (N, T_q, C)

        return outputs, att_vec

    #encoder-decoder construct
    def self_attention(self,enc, num_units, num_heads, num_blocks, dropout_rate, is_training, reuse, key_length, query_length,
                       t_querys, t_keys, t_querys_length,t_keys_length):
        with tf.variable_scope("encoder"):
            for i in range(num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    enc, stt_vec = self.time_aware_multihead_attention(queries=enc,
                                                       keys=enc,
                                                       num_units=num_units,
                                                       num_heads=num_heads,
                                                       dropout_rate=dropout_rate,
                                                       is_training=is_training,
                                                       scope="self_attention",
                                                       key_length = key_length,
                                                       query_length = query_length,
                                                       t_querys= t_querys,
                                                       t_keys = t_keys,
                                                       t_querys_length= t_querys_length,
                                                       t_keys_length= t_keys_length

                                                       )

                    ### Feed Forward
                    #enc = self.feedforward(enc,
                                      #num_units=[num_units // 4, num_units],
                                      #scope="feed_forward", reuse=reuse)


                    self.self_attention_att_vec = stt_vec


            return enc
    def Tiself_attention(self,enc, num_units, num_heads, num_blocks, dropout_rate, is_training, reuse, key_length, query_length,
                       t_querys, t_keys, t_querys_length,t_keys_length):
        with tf.variable_scope("encoder"):
            for i in range(num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    enc, stt_vec = self.TiSAS_multihead_attention(queries=enc,
                                                       keys=enc,
                                                       num_units=num_units,
                                                       num_heads=num_heads,
                                                       dropout_rate=dropout_rate,
                                                       is_training=is_training,
                                                       scope="self_attention",
                                                       key_length = key_length,
                                                       query_length = query_length,
                                                       t_querys= t_querys,
                                                       t_keys = t_keys,
                                                       t_querys_length= t_querys_length,
                                                       t_keys_length= t_keys_length

                                                       )

                    ### Feed Forward
                    #enc = self.feedforward(enc,
                                      #num_units=[num_units // 4, num_units],
                                      #scope="feed_forward", reuse=reuse)


                    self.self_attention_att_vec = stt_vec


            return enc

    def vanilla_attention(self,enc,dec,num_units, num_heads, num_blocks, dropout_rate, is_training, reuse, key_length, query_length,  t_querys, t_keys,t_querys_length, t_keys_length):

        #dec = tf.expand_dims(dec, 1)#在1的位置上增加1维
        with tf.variable_scope("decoder"):
            for i in range(num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ## Multihead Attention ( vanilla attention)
                    dec, att_vec = self.time_aware_multihead_attention(queries=dec,
                                                       keys=enc,
                                                       num_units=num_units,
                                                       num_heads=num_heads,
                                                       dropout_rate=dropout_rate,
                                                       is_training=is_training,
                                                       scope="vanilla_attention",
                                                       key_length = key_length,
                                                       query_length = query_length,
                                                       t_querys =t_querys,
                                                       t_keys = t_keys,
                                                       t_querys_length=t_querys_length,
                                                       t_keys_length=t_keys_length
                                                       )

                    ## Feed Forward
                    #dec = self.feedforward(dec,num_units=[num_units // 4, num_units],
                                      #scope="feed_forward", reuse=reuse)


                    self.vanilla_attention_att_vec = att_vec


        # 此处怀疑有错误，非常重要
        dec = tf.reshape(dec, [-1, num_units])
        return dec

    # def vanilla_attention(self, queries,
    #                       keys,
    #                       keys_length):
    #     '''
    #       queries:     [B, H]
    #       keys:        [B, T, H]
    #       keys_length: [B]
    #     '''
    #     queries = tf.tile(queries, [1, 2])
    #     queries = tf.expand_dims(queries, 1)  # [B, 1, H]
    #     # Multiplication
    #     outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1]))  # [B, 1, T]
    #
    #     # Mask
    #     key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # [B, T]
    #     key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
    #     paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    #     outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]
    #
    #     # Scale
    #     outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)
    #
    #     # Activation
    #     outputs = tf.nn.softmax(outputs)  # [B, 1, T]
    #
    #     # Weighted sum
    #     outputs = tf.matmul(outputs, keys)  # [B, 1, H]
    #
    #     return outputs

