import os
import tensorflow as tf
import pickle
import numpy as np
import re

os.chdir("/user_home/hajung/vae_model")

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string(
    "data_dir", "/user_home/hajung/vae_data/",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "output_dir", './model11',
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_integer("batch_count", 32, "Total batch size for training.")

flags.DEFINE_float("learning_rate", 0.001, "The initial learning rate for Adam.")

flags.DEFINE_integer("epoch_count", 1,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_rate", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_integer("train_count", 8595, None)
flags.DEFINE_integer("test_count", 2149, None)
flags.DEFINE_bool("do_train", True, None)
flags.DEFINE_bool("do_eval", False, None)
flags.DEFINE_bool("do_test", True, None)

class Console:

    def __init__(self):
        print("Console Init!")


    def main(self):
        epoch_train_steps = int(FLAGS.train_count / FLAGS.batch_count)
        num_train_steps = epoch_train_steps * float(FLAGS.epoch_count)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_rate)
        print("Epoch train steps %d" % epoch_train_steps)
        print("Total train steps %d" % num_train_steps)

        model_fn = build_model_fn(num_train_steps, num_warmup_steps)
        estimator = tf.estimator.Estimator(model_fn, model_dir=FLAGS.output_dir)

        if FLAGS.do_train:
            mode = tf.estimator.ModeKeys.TRAIN
            train_input_fn = build_input_fn(mode, "Train", "Train")
            estimator.train(train_input_fn)

        if FLAGS.do_eval:
            mode = tf.estimator.ModeKeys.EVAL
            test_input_fn = build_input_fn(mode, "Test", "Test")
            estimator.evaluate(test_input_fn)

        if FLAGS.do_predict:
            mode = tf.estimator.ModeKeys.PREDICT
            test_input_fn = build_input_fn(mode, "Test", "Test")
            predictions = estimator.predict(test_input_fn)
            cnt =0
            for item in predictions:
                if cnt > 3:
                    break
                print(item)
                cnt +=1


def build_input_fn(mode, X_file, Y_file):
    def input_fn(params):

        if mode == tf.estimator.ModeKeys.TRAIN:
            with open(FLAGS.data_dir + X_file, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(FLAGS.data_dir + X_file, 'rb') as f:
                data = pickle.load(f)

                
        ds = tf.data.Dataset.from_tensor_slices((data, data))
        if mode == tf.estimator.ModeKeys.TRAIN:
            batch = ds.repeat(FLAGS.epoch_count).apply(tf.contrib.data.batch_and_drop_remainder(FLAGS.batch_count))
        else:
            batch = ds.batch(FLAGS.batch_count)
        return batch


    return input_fn


def build_model_fn( num_train_steps, num_warmup_steps):
    def model_fn(features, labels,mode, params):
        input_ids = features
        labels = features

        kl_weight = 1.0
        LD = 100
        DD = 100
        LAD = 100
        LEN = 60
        BS = FLAGS.batch_count

        with open(FLAGS.data_dir + "embedding", 'rb') as f:
            embedding_value = pickle.load(f)


        embedding = tf.get_variable("embedding", dtype=tf.float32, initializer=tf.cast(embedding_value, tf.float32))
        e_l = tf.nn.embedding_lookup(embedding, input_ids)

        with tf.variable_scope("encoder"):
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(LD, state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(LD, state_is_tuple=True)
            _initial_state_fw = cell_fw.zero_state(BS, tf.float32)
            _initial_state_bw = cell_bw.zero_state(BS, tf.float32)            
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, e_l, dtype=tf.float32)
        fw_output = outputs[0][:,:,-1]
        bw_output = outputs[1][:,:,-1]
        cat_output = tf.concat([fw_output, bw_output], axis=1)
        z_output_mean = tf.keras.layers.Dense(DD)(cat_output)
        z_output_sd = tf.keras.layers.Dense(DD)(cat_output)
        epsilon = tf.random_normal(shape=(BS, DD), mean=0., stddev=1.0)
        z = z_output_mean + tf.multiply(tf.exp(z_output_sd / 2) , epsilon)
        
        with tf.variable_scope("decoder"):
            repeat_z = tf.tile(tf.expand_dims(z, 1), [1,LEN,1])
            cell = tf.nn.rnn_cell.BasicLSTMCell(LD)
            decode, states = tf.nn.dynamic_rnn(cell, repeat_z, dtype=tf.float32)
            decode_mean = tf.keras.layers.Dense(embedding.shape[0])(decode)

        with tf.variable_scope("loss"):
            labels = tf.cast(labels, tf.int32)
            target_weights = tf.constant(np.ones((BS, LEN)), tf.float32)

            xent_loss = tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(decode_mean, labels,
                                                                   weights=target_weights,
                                                                   average_across_timesteps=False,
                                                                   average_across_batch=False), axis=-1)
            logits = decode_mean
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            kl_loss = - 0.5 * tf.reduce_sum(1 + z_output_sd - tf.square(z_output_mean) - tf.exp(z_output_mean), axis=-1)
            xent_loss = tf.reduce_mean(xent_loss)
            kl_loss = tf.reduce_mean(kl_loss)
            loss = xent_loss + (kl_weight * kl_loss)
            tf.summary.scalar("total_loss",loss)

        
        output_spec = None

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(1e-3).minimize(loss,global_step=tf.train.get_global_step())
#            train_op = create_optimizer(
#                loss, FLAGS.learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
            output_spec = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(loss, labels, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(labels, predictions)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn, [loss, labels, logits])
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metrics)
        else:
            output_spec = tf.estimator.EstimatorSpec(mode=mode,
                                                     predictions={"prediction":predictions})
        return output_spec
    return model_fn


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

    if use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)

            m = tf.get_variable(
                name=param_name + "/adam_m",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())
            v = tf.get_variable(
                name=param_name + "/adam_v",
                shape=param.shape.as_list(),
                dtype=tf.float32,
                trainable=False,
                initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                    tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                    tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                              tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = self.learning_rate * update

            next_param = param - update_with_lr

            assignments.extend(
                [param.assign(next_param),
                 m.assign(next_m),
                 v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


Console().main()
