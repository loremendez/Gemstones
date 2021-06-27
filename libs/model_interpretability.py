import tensorflow as tf
import numpy as np
import copy

def interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x +  alphas_x * delta

    return images

def compute_gradients(images, model):
    with tf.GradientTape() as tape:
        tape.watch(images)
        prediction = model(images)

    return tape.gradient(prediction, images)

def get_integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)

    return integrated_gradients

@tf.function
def get_integrated_gradients(baseline, image, model, m_steps=50, batch_size=32):
    # 1. Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

    # Initialize TensorArray outside loop to collect gradients.
    gradient_batches = tf.TensorArray(tf.float32, size=m_steps+1)

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        # 2. Generate interpolated inputs between baseline and input.
        interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                           image=image,
                                                           alphas=alpha_batch)

        # 3. Compute gradients between model outputs and interpolated inputs.
        gradient_batch = compute_gradients(images=interpolated_path_input_batch,
                                           model=model)

        # Write batch indices and gradients to extend TensorArray.
        gradient_batches = gradient_batches.scatter(tf.range(from_, to), gradient_batch)

    # Stack path gradients together row-wise into single tensor.
    total_gradients = gradient_batches.stack()

    # 4. Integral approximation through averaging gradients.
    avg_gradients = get_integral_approximation(gradients=total_gradients)

    # 5. Scale integrated gradients with respect to input.
    integrated_gradients = (image - baseline) * avg_gradients

    return integrated_gradients


#@tf.function
def get_VarGrad(img=None, baseline='black', model=None, n_images=15, IG_m_steps=30):

    if baseline == 'black':
        bl = tf.zeros(shape=img.shape)
    elif baseline == 'noise':
        bl = tf.random.normal(shape=img.shape, mean=0, stddev=1)
    else:
        raise NotImplementedError('{} baseline method not implemented'.format(baseline))

    for i in range(n_images):
        rnorm_img = tf.random.normal(shape=img.shape, mean=0, stddev=1)

        ig_temp = get_integrated_gradients(baseline=bl,
                                           image=img+rnorm_img,
                                           model=model,
                                           m_steps=IG_m_steps
                                           )

        ig_temp = tf.expand_dims(ig_temp, axis=0)

        if i == 0:
            ig_sample = ig_temp
        else:
            ig_sample = tf.concat((ig_sample, ig_temp), axis=0)

    return tf.math.reduce_std(ig_sample, axis=0)
