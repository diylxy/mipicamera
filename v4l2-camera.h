#ifndef __V4L2_CAMERA_H__
#define __V4L2_CAMERA_H__

struct camera_buffer {
    void *start;
    size_t length;
    struct v4l2_buffer v4l2_buf;
};
typedef void (*camera_buffer_cb)(struct camera_buffer *buffer, int bytesused, void *data);

void open_device(const char *dev_name);
int set_format(
	int width,
	int height,
	unsigned int format
);
int init_device(void);
int start_capturing(void);
int read_frame(camera_buffer_cb processor, void *data);

#endif