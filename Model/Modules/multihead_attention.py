import tensorflow as tf

class Attention():

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


    def multihead_attention(self,queries,
                            keys,
                            key_length,
                            query_length,
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
        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

            # Linear projections, C = # dim or column, T_x = # vectors or actions
            Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)


            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            # Multiplication
            # query-key score matrix
            # each big score matrix is then split into h score matrix with same size
            # w.r.t. different part of the feature
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

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
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            outputs += queries

            # Normalize
            outputs = self.normalize(outputs)  # (N, T_q, C)

        return outputs, att_vec

    #encoder-decoder construct
    def self_attention(self,enc, num_units, num_heads, num_blocks, dropout_rate, is_training, reuse, key_length, query_length):
        with tf.variable_scope("encoder"):
            for i in range(num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    enc, stt_vec = self.multihead_attention(queries=enc,
                                                       keys=enc,
                                                       num_units=num_units,
                                                       num_heads=num_heads,
                                                       dropout_rate=dropout_rate,
                                                       is_training=is_training,
                                                       scope="self_attention",
                                                       key_length = key_length,
                                                       query_length = query_length
                                                       )

                    ### Feed Forward
                    #enc = self.feedforward(enc,
                                      #num_units=[num_units // 4, num_units],
                                      #scope="feed_forward", reuse=reuse)


                    self.self_attention_att_vec = stt_vec


            return enc


    def vanilla_attention(self,enc,dec,num_units, num_heads, num_blocks, dropout_rate, is_training, reuse, key_length, query_length):

        #dec = tf.expand_dims(dec, 1)#在1的位置上增加1维
        with tf.variable_scope("decoder"):
            for i in range(num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ## Multihead Attention ( vanilla attention)
                    dec, att_vec = self.multihead_attention(queries=dec,
                                                       keys=enc,
                                                       num_units=num_units,
                                                       num_heads=num_heads,
                                                       dropout_rate=dropout_rate,
                                                       is_training=is_training,
                                                       scope="vanilla_attention",
                                                       key_length = key_length,
                                                       query_length = query_length)

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

