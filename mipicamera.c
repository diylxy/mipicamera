#include <Python.h>
#include <numpy/arrayobject.h>
#include "v4l2-camera.h"

static int counter = 0;

static PyObject *method_open(PyObject *self, PyObject *args) {
    char *filename = NULL;
    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "s", &filename)) {
        return NULL;
    }
    open_device();
    init_device();
    return PyBool_FromLong(1);
}

static PyObject *method_read(PyObject *self, PyObject *args) {
    char *filename = NULL;
    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "s", &filename)) {
        return NULL;
    }
    npy_intp dims[3] = {640, 480, 3}; // 1D array with 10 elements
    PyObject *numpy_array;
    /* Create a new NumPy array */
    numpy_array = PyArray_SimpleNew(3, dims, NPY_UBYTE);

    if (numpy_array == NULL) {
        return NULL;
    }

    /* Fill the array with some data */
    uint8_t *data = (uint8_t *)PyArray_DATA((PyArrayObject *)numpy_array);
    for (int i = 0; i < 640 * 480 * 3; i++) {
        *(data + i) = counter + i;
    }

    counter++;

    return numpy_array;
}

static PyMethodDef MipicameraMethods[] = {
    {"open", method_open, METH_VARARGS, "open the camera"},
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
