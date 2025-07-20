import tensorflow as tf

def compute_residual(model, res, a=2.0, nu=0.05):
    x, y, t = tf.split(res, 3, axis=1)

    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([x, y, t])
        with tf.GradientTape(persistent=True) as tape0:
            tape0.watch([x, y, t])
            u = model(tf.concat([x, y, t], axis=1))
        u_x = tape0.gradient(u, x)
        u_y = tape0.gradient(u, y)
        u_t = tape0.gradient(u, t)
    u_xx = tape1.gradient(u_x, x)
    u_yy = tape1.gradient(u_y, y)

    del tape0, tape1

    residual = u_t + a * (u_x + u_y) - nu * (u_xx + u_yy)
    return residual


def compute_loss(model, res, in_i, out_i, in_b, out_b):
    out_pred_i = model(in_i)
    out_pred_b = model(in_b)
    residual = compute_residual(model, res)

    loss_i = tf.reduce_mean(tf.square(out_i - out_pred_i))
    loss_b = tf.reduce_mean(tf.square(out_b - out_pred_b))
    loss_r = tf.reduce_mean(tf.square(residual))

    total_loss = loss_i + loss_b + loss_r

    return total_loss, loss_i, loss_b, loss_r


@tf.function
def train_step(model, optimizer, res, in_i, out_i, in_b, out_b):
    with tf.GradientTape() as tape:
        total_loss, loss_r, loss_b, loss_i = compute_loss(
            model, res, in_i, out_i, in_b, out_b
        )
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss, loss_r, loss_b, loss_i