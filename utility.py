import tensorflow as tf
import time

def _parse_data_infer(image_paths):
    image_content = tf.read_file(image_paths)
    images = tf.image.decode_png(image_content, channels=0, dtype=tf.uint16)

    return images


def _normalize_data_infer(image):
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, tf.reduce_max(image))

    return image

def _rotate_data_infer_16(image):
    image = tf.contrib.image.rotate(image, 20, 'BILINEAR')
    return image

def _rotate_data_infer_32(image):
    image = tf.contrib.image.rotate(image, 40, 'BILINEAR')
    return image

def _brightness_infer(image):
    max_delta=20
    return tf.image.random_brightness(image, max_delta)


def _gaussian_noise_infer(image):
    noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=0.15, dtype=tf.float32)
    return image + noise
def _resize_data_infer(image):
    image = tf.cast(tf.image.resize_images(tf.cast(image, tf.int16), [256, 256]), tf.uint16)
    return image


def _normalize_data(image, mask):
    """Normalizes data in between 0-1"""
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, tf.reduce_max(image))

    mask = tf.cast(mask, tf.float32)
    mask = tf.divide(mask, 255.0)

    return image, mask


def _resize_data(image, mask):
    """Resizes images to smaller dimensions."""
    image = tf.cast(tf.image.resize_images(tf.cast(image, tf.int16), [256, 256]), tf.uint16)
    mask = tf.image.resize_images(mask, [256, 256])

    return image, mask

def _rotate_data(image, mask, angle):
    
    image = tf.contrib.image.rotate(image, angle, 'BILINEAR')
    mask = tf.contrib.image.rotate(tf.cast(mask, tf.float32), angle, 'BILINEAR')
    return image,mask

def _flip_data(image, mask):
    image = tf.image.flip_left_right(image)
    mask = tf.image.flip_left_right(mask)
    return image, mask

def _brightness(image, mask):
    max_delta=0.1
    return tf.image.random_brightness(image, max_delta), tf.image.random_brightness(mask, max_delta)

def _gaussian_noise(image, mask):
    noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=0.15, dtype=tf.float32)
    return image + noise, mask + noise

def _parse_data(image_paths, mask_paths):
    """Reads image and mask files"""
    tf.logging.debug("parsing %s, %s" % (image_paths, mask_paths))
    image_content = tf.read_file(image_paths)
    images = tf.image.decode_png(image_content, channels=0, dtype=tf.uint16)

    mask_content = tf.read_file(mask_paths)
    masks = tf.image.decode_png(mask_content, channels=0)

    return images, masks


def data_batch(image_paths, mask_paths, batch_size=4, augment=True, num_threads=72, shuffle=True):
    PF = 14000
    """Reads data, normalizes it, shuffles it, then batches it, returns a
       the next element in dataset op and the dataset initializer op.
       Inputs:
        image_paths: A list of paths to individual images
        mask_paths: A list of paths to individual mask images
        batch_size: Number of images/masks in each batch returned
        num_threads: Number of parallel calls to make
       Returns:
        next_element: A tensor with shape [2], where next_element[0]
                      is image batch, next_element[1] is the corresponding
                      mask batch
        init_op: Data initializer op, needs to be executed in a session
                 for the data queue to be filled up and the next_element op
                 to yield batches"""

    # Convert lists of paths to tensors for tensorflow
    images_name_tensor = tf.constant(image_paths)
    
    if mask_paths:
        mask_name_tensor = tf.constant(mask_paths)
        data = tf.data.Dataset.from_tensor_slices(
            (images_name_tensor, mask_name_tensor))

        s = time.perf_counter()
        
        data = data.map(_parse_data, num_parallel_calls=num_threads).prefetch(PF)
        e = time.perf_counter() - s
        print("Parsing Data : %.3f" % (e))

        s = time.perf_counter()
        data = data.map(_resize_data, num_parallel_calls=num_threads).prefetch(PF)
        #data = data.map(lambda img, msk: tf.py_func(_resize_data, [img, msk], [tf.float32, tf.float32]), num_parallel_calls=num_threads).prefetch(PF)
        e = time.perf_counter() - s
        print("Resizing Data : %.3f" % (e))

        s = time.perf_counter()
        data = data.map(_normalize_data, num_parallel_calls=num_threads).prefetch(PF)

        #data = data.map(lambda img, msk: tf.py_func(_normalize_data, [img, msk], [tf.float32, tf.float32]), num_parallel_calls=num_threads).prefetch(PF)
                
        e = time.perf_counter() - s
        print("Normalizing Data : %.3f" % (e))

        tmp = []

        for i in range(10, 180, 15):

            s = time.perf_counter()
            #rot = data.map(lambda img, msk: tf.py_func(_rotate_data, [img, msk, i], [tf.float32, tf.float32]), num_parallel_calls=num_threads).prefetch(PF)
            rot = data.map(lambda img, msk: _rotate_data(img, msk, i), num_parallel_calls=num_threads).prefetch(PF)
            e = time.perf_counter() - s
            print("Rotating Data %d : %.3f" % (i, e))

            s = time.perf_counter()
            #flp = data.map(lambda img, msk: tf.py_func(_flip_data, [img, msk, i], [tf.float32, tf.float32]), num_parallel_calls=num_threads).prefetch(PF)
            flp = data.map(_flip_data, num_parallel_calls=num_threads).prefetch(PF)

            e = time.perf_counter() - s
            print("Flipping Data %d : %.3f" % (i, e))
            tmp.append(rot)
            #tmp.append(flp)

        for t in tmp:
            data = data.concatenate(t)

        #data = data.map(_normalize_data,
        #                num_parallel_calls=num_threads).prefetch(30)
    else:
        data = tf.data.Dataset.from_tensor_slices((images_name_tensor))
        data = data.map(_parse_data_infer,
                        num_parallel_calls=num_threads).prefetch(PF)
        data = data.map(_resize_data_infer,
                        num_parallel_calls=num_threads).prefetch(PF)
        data = data.map(_normalize_data_infer,
                        num_parallel_calls=num_threads).prefetch(PF)
        """
        data = data.concatenate(data.map(_rotate_data_infer_16,
                        num_parallel_calls=num_threads).prefetch(30))
        data = data.concatenate(data.map(_rotate_data_infer_32,
                        num_parallel_calls=num_threads).prefetch(30))
        data = data.concatenate(data.map(_brightness_infer,
                        num_parallel_calls=num_threads).prefetch(30))
        data = data.concatenate(data.map(_gaussian_noise_infer,
                        num_parallel_calls=num_threads).prefetch(30))
        """
        #data = data.map(_normalize_data_infer,
        #                num_parallel_calls=num_threads).prefetch(30)
    # Batch the data

    print("Batching")
    data = data.batch(batch_size=batch_size)
    data = data.prefetch(buffer_size=PF)
   
    """
    s = time.perf_counter()
    data = data.shuffle(14000 * 7)
    e = time.perf_counter() - s
    print("Shuffling Data %d : %.3f" % (e))
    """
    # Create iterator
    print("Creating")
    s = time.perf_counter()

    iterator = tf.data.Iterator.from_structure(
        data.output_types, data.output_shapes)
    e = time.perf_counter() - s
    print("Creating iterator : %.3f" % (e))

    # Next element Op
    next_element = iterator.get_next()
    print("Next")
    # Data set init. op
    s = time.perf_counter()
    init_op = iterator.make_initializer(data)
    e = time.perf_counter() - s
    print("Initializing iterator : %.3f" % (e))
    return next_element, init_op
