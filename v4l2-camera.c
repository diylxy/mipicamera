/*
2021-05-18 11:02:09

This program demo demonstrates how to call MIPI camera in C and C++, and use OpenCV for image display.


firefly
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <getopt.h> /* getopt_long() */
#include <fcntl.h> /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <dlfcn.h>

#include <linux/videodev2.h>

#include "v4l2-camera.h"

#define BUFFER_COUNT 1
#define FMT_NUM_PLANES 1
#define CLEAR(x) memset(&(x), 0, sizeof(x))

#define DBG(...) do { if(!silent) printf(__VA_ARGS__); } while(0)
#define ERR(...) do { fprintf(stderr, __VA_ARGS__); } while (0)


static int fd = -1;
FILE *fp=NULL;
static unsigned int n_buffers;
struct camera_buffer *buffers;
static int silent=0;

static int width = 640;
static int height = 480;
static int format = V4L2_PIX_FMT_YUV420;
static enum v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;


static int xioctl(int fh, int request, void *arg)
{
	int r;
	do {
		r = ioctl(fh, request, arg);
	} while (-1 == r && EINTR == errno);
	return r;
}

int open_device(const char *dev_name)
{
	if (fd != -1) {
		ERR("device already opened\n");
		return 2;
	}
    fd = open(dev_name, O_RDWR /* required */ /*| O_NONBLOCK*/, 0);

    if (-1 == fd) {
        ERR("Cannot open '%s': %d, %s\n",
                    dev_name, errno, strerror(errno));
        return 1;
    }
	return 0;
}


static int init_mmap(void)
{
	struct v4l2_requestbuffers req;

	CLEAR(req);

	req.count = BUFFER_COUNT;
	req.type = buf_type;
	req.memory = V4L2_MEMORY_MMAP;

	if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req)) {
		if (EINVAL == errno) {
			ERR("device does not support "
							"memory mapping\n");
			return 1;
		} else {
			return errno;
		}
	}

	if (req.count < 1) {
		ERR("Insufficient buffer memory\n");
		return 1;
	}

	buffers = (struct camera_buffer*)calloc(req.count, sizeof(*buffers));

	if (!buffers) {
		ERR("Out of memory\n");
		return 1;
	}

	for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {
		struct v4l2_buffer buf;
		struct v4l2_plane planes[FMT_NUM_PLANES];
		CLEAR(buf);
		CLEAR(planes);

		buf.type = buf_type;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = n_buffers;

		if (V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE == buf_type) {
			buf.m.planes = planes;
			buf.length = FMT_NUM_PLANES;
		}

		if (-1 == xioctl(fd, VIDIOC_QUERYBUF, &buf))
			return errno;

		if (V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE == buf_type) {
			buffers[n_buffers].length = buf.m.planes[0].length;
			buffers[n_buffers].start =
			mmap(NULL /* start anywhere */,
					buf.m.planes[0].length,
					PROT_READ | PROT_WRITE /* required */,
					MAP_SHARED /* recommended */,
					fd, buf.m.planes[0].m.mem_offset);
		} else {
			buffers[n_buffers].length = buf.length;
			buffers[n_buffers].start =
			mmap(NULL /* start anywhere */,
					buf.length,
					PROT_READ | PROT_WRITE /* required */,
					MAP_SHARED /* recommended */,
					fd, buf.m.offset);
		}

		if (MAP_FAILED == buffers[n_buffers].start)
			return errno;
	}
	return 0;
}

void set_resolution(int _width, int _height) {
	width = _width;
	height = _height;
}

int set_format(
	int _width,
	int _height
) {
	struct v4l2_format fmt;

	CLEAR(fmt);
	fmt.type = buf_type;
	fmt.fmt.pix.width = _width;
	fmt.fmt.pix.height = _height;
	fmt.fmt.pix.pixelformat = format;
	fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;

	if (-1 == xioctl(fd, VIDIOC_S_FMT, &fmt))
		return errno;
	width = _width;
	height = _height;
	return 0;
}

int get_width(void) {
	return width;
}

int get_height(void) {
	return height;
}

int init_device(void)
{
    struct v4l2_capability cap;

    if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &cap)) {
		if (EINVAL == errno) {
			ERR("camera is no V4L2 device\n");
			return 1;
		} else {
			return errno;
		}
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) &&
            !(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE)) {
        ERR("not a video capture device, capabilities: %x\n",
                        cap.capabilities);
            return 1;
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
		ERR("device does not support streaming i/o\n");
		return 1;
    }

    if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) {
        buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	}
    else if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE) {
        buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
	}

	if (set_format(width, height)) {
		ERR("set format failed\n");
		return 1;
	}

	struct v4l2_control ctl;
	ctl.id = V4L2_CID_AUTO_WHITE_BALANCE;
	ctl.value = 1;
	ioctl(fd, VIDIOC_S_CTRL, &ctl);
	ctl.id = V4L2_CID_EXPOSURE_AUTO;
	ctl.value = V4L2_EXPOSURE_AUTO;
	ioctl(fd, VIDIOC_S_CTRL, &ctl);
	ctl.id = V4L2_CID_AUTOGAIN;
	ctl.value = 1;
	ioctl(fd, VIDIOC_S_CTRL, &ctl);

    return init_mmap();
}


int start_capturing(void)
{
	unsigned int i;
	enum v4l2_buf_type type;

	for (i = 0; i < n_buffers; ++i) {
		struct v4l2_buffer buf;

		CLEAR(buf);
		buf.type = buf_type;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = i;

		if (V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE == buf_type) {
			struct v4l2_plane planes[FMT_NUM_PLANES];

			buf.m.planes = planes;
			buf.length = FMT_NUM_PLANES;
		}
		if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
			return errno;
	}
	type = buf_type;
	if (-1 == xioctl(fd, VIDIOC_STREAMON, &type))
		return errno;
	return 0;
}


int read_frame(camera_buffer_cb processor, void *data)
{
	struct v4l2_buffer buf;
	int i, bytesused;

	if (processor == NULL) {
		ERR("processor is NULL\n");
		return 1;
	}

	CLEAR(buf);

	buf.type = buf_type;
			buf.memory = V4L2_MEMORY_MMAP;

	if (V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE == buf_type) {
			struct v4l2_plane planes[FMT_NUM_PLANES];
			buf.m.planes = planes;
			buf.length = FMT_NUM_PLANES;
	}

	if (-1 == xioctl(fd, VIDIOC_DQBUF, &buf))
		return errno;

	i = buf.index;

	if (V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE == buf_type)
			bytesused = buf.m.planes[0].bytesused;
	else
			bytesused = buf.bytesused;
	processor(&(buffers[i]), bytesused, data);
	DBG("bytesused %d\n", bytesused);

	if (-1 == xioctl(fd, VIDIOC_QBUF, &buf))
		return errno;

	return 0;
}
