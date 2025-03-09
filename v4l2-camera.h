#ifndef __V4L2_CAMERA_H__
#define __V4L2_CAMERA_H__

void open_device(void);
void init_device(void);
void start_capturing(void);
int read_frame();

#endif