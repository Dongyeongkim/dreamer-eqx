import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
import tensorflow.compat.v1 as tf1


class Logger:
    def __init__(self, path, videos=True, fps=20):
        self._fps = fps
        self._videos = videos
        self._writer = tf.summary.create_file_writer(path)
        self._writer.set_as_default()

    def _write(self, summaries, step):
        for name, value in summaries.items():
            try:
                if isinstance(value, str):
                    tf.summary.text(name, value, step)
                elif len(value.shape) == 0:
                    tf.summary.scalar(name, value, step)
                elif len(value.shape) == 1:
                    if len(value) > 1024:
                        value = value.copy()
                        np.random.shuffle(value)
                        value = value[:1024]
                    tf.summary.histogram(name, value, step)
                elif len(value.shape) == 2:
                    tf.summary.image(name, value[None, ..., None], step)
                elif len(value.shape) == 3:
                    tf.summary.image(name, value[None], step)
                elif len(value.shape) == 4 and self._videos:
                    self._video_summary(name, value, step)
            except Exception:
                print("Error writing summary:", name)
                raise

            self._writer.flush()

    def _video_summary(self, name, video, step):
        name = name if isinstance(name, str) else name.decode("utf-8")
        assert video.dtype in (np.float32, np.uint8), (video.shape, video.dtype)
        if np.issubdtype(video.dtype, np.floating):
            video = np.clip(255 * video, 0, 255).astype(np.uint8)
        try:
            T, H, W, C = video.shape
            summary = tf1.Summary()
            image = tf1.Summary.Image(height=H, width=W, colorspace=C)
            image.encoded_image_string = _encode_gif(video, self._fps)
            summary.value.add(tag=name, image=image)
            tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
        except (IOError, OSError) as e:
            print("GIF summaries require ffmpeg in $PATH.", e)
            tf.summary.image(name, video, step)


def _encode_gif(frames, fps):
    from subprocess import Popen, PIPE

    h, w, c = frames[0].shape
    pxfmt = {1: "gray", 3: "rgb24"}[c]
    cmd = " ".join(
        [
            "ffmpeg -y -f rawvideo -vcodec rawvideo",
            f"-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex",
            "[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse",
            f"-r {fps:.02f} -f gif -",
        ]
    )
    proc = Popen(cmd.split(" "), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tobytes())
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError("\n".join([" ".join(cmd), err.decode("utf8")]))
    del proc
    return out
