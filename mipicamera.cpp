#include <Python.h>
#include <numpy/arrayobject.h>
#include <linux/videodev2.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
extern "C" {
#include "v4l2-camera.h"
}

bool started = false;

static PyObject *method_open(PyObject *self, PyObject *args) {
    char *filename = NULL;
    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "s", &filename)) {
        return NULL;
    }
    if (filename[0] == '\0') {
        filename = "/dev/video0";
    }
    if (open_device(filename) != 0) {
        return PyBool_FromLong(0);
    }
    return PyBool_FromLong(1);
}

static PyObject *method_setResolution(PyObject *self, PyObject *args) {
    int width, height;
    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "ii", &width, &height)) {
        return NULL;
    }

    if (started) {
        return PyBool_FromLong(0);
    }

    set_resolution(width, height);

    return PyBool_FromLong(1);
}

static PyObject *method_start(PyObject *self, PyObject *args) {
    if (started) {
        return PyBool_FromLong(1);
    }

    if (init_device() != 0) {
        return PyBool_FromLong(0);
    }

    if (start_capturing() != 0) {
        return PyBool_FromLong(0);
    }
    started = true;
    return PyBool_FromLong(1);
}

static void copy_buffer(struct camera_buffer *buffer, int bytesused, void *data) {
    int width = get_width();
    int height = get_height();
	cv::Mat yuvmat(cv::Size(width, height*3/2), CV_8UC1, buffer->start);
	cv::Mat rgbmat(cv::Size(width, height), CV_8UC3, data);
	cv::cvtColor(yuvmat, rgbmat, CV_YUV2BGR_I420);
}

static PyObject *method_read(PyObject *self, PyObject *args) {
    npy_intp dims[3] = {get_height(), get_width(), 3};
    PyObject *numpy_array;
    /* Create a new NumPy array */
    numpy_array = PyArray_SimpleNew(3, dims, NPY_UBYTE);

    if (numpy_array == NULL) {
        return NULL;
    }

    /* Fill the array with some data */
    uint8_t *data = (uint8_t *)PyArray_DATA((PyArrayObject *)numpy_array);

    read_frame(copy_buffer, data);

    return numpy_array;
}

static PyMethodDef MipicameraMethods[] = {
    {"open", method_open, METH_VARARGS, "open the camera"},
    {"setResolution", method_setResolution, METH_VARARGS, "set the resolution"},
    {"start", method_start, METH_VARARGS, "start streaming"},
    {"read", method_read, METH_VARARGS, "read one frame"},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef mipicameraModule = {
    PyModuleDef_HEAD_INIT,
    "mipicamera",
    "RK356X MIPI RGB Camera Helper",
    -1,
    MipicameraMethods
};

PyMODINIT_FUNC PyInit_mipicamera(void) {
    PyObject * ret =  PyModule_Create(&mipicameraModule);
    import_array();
    return ret;
}
