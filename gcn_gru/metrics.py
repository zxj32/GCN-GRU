import tensorflow as tf


def masked_mse_square(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    # loss = tf.square(preds - labels)
    loss = tf.square(preds - labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def masked_mse_abs(preds, labels, mask):
    """ MSE with masking."""
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    # loss = tf.square(preds - labels)
    # preds = tf.reduce_mean(preds, axis=1)
    loss = tf.abs(preds - labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_decode(preds, mask):
    """MSE decode with masking."""
    mask = tf.cast(mask, dtype=tf.float32)
    loss = mask - preds
    mask /= tf.reduce_mean(mask)
    loss = loss * mask
    return tf.reduce_mean(loss)


def masked_decode_sparse(pred, adj):
    logits = tf.convert_to_tensor(pred, name="logits")
    targets = tf.convert_to_tensor(adj, name="targets")
    # targets = tf.reshape(targets, [70317, 70317])
    try:
        targets.get_shape().merge_with(logits.get_shape())
    except ValueError:
        raise ValueError(
            "logits and targets must have the same shape (%s vs %s)" %
            (logits.get_shape(), targets.get_shape()))
    loss = targets - logits
    targets /= tf.reduce_mean(targets)
    loss = loss * targets
    return tf.reduce_mean(loss)


def masked_mae_rnn(preds, labels, mask):
    loss = tf.abs(preds - labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_mse_rnn(preds, labels, mask):
    loss = tf.square(preds - labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)