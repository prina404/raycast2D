#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_25_API_VERSION
#include <numpy/arrayobject.h>

static int CHECK_ARRAYS(PyArrayObject* img, PyArrayObject* rays) {
    if (!img || !rays) {
        Py_XDECREF(img);
        if (rays) {
            PyArray_DiscardWritebackIfCopy(rays);
            Py_DECREF(rays);
        }
        return 0;
    }

    if (PyArray_NDIM(img) != 2) {
        PyErr_SetString(PyExc_ValueError, "Image must be a 2D array, received shape different than (H, W)");
        Py_DECREF(img);
        PyArray_DiscardWritebackIfCopy(rays);
        Py_DECREF(rays);
        return 0;
    }

    if (PyArray_NDIM(rays) != 2 || PyArray_DIM(rays, 1) != 3) {
        PyErr_SetString(PyExc_ValueError, "Rays must be a 2D array of shape (N, 3)");
        Py_DECREF(img);
        PyArray_DiscardWritebackIfCopy(rays);
        Py_DECREF(rays);
        return 0;
    }
    return 1;
}

/**
 * Returns true if x, y are valid coordinates and the corresponding cell is not occupied
 */
static int OCCUPIED_CELL(PyArrayObject* img, int x, int y) {
    npy_intp H = PyArray_DIM(img, 0);
    npy_intp W = PyArray_DIM(img, 1);

    if (x <= 0 || x >= W - 1 || y <= 0 || y >= H - 1)
        return 1;

    npy_ubyte cell_value = *((npy_ubyte*)PyArray_GETPTR2(img, y, x));
    return cell_value == 0;
}

/**
 * Implementation taken from https://zingl.github.io/bresenham.html
 */
static void _bresenham_raycast(PyArrayObject* img, PyArrayObject* rays, int x, int y) {
    npy_intp N_RAYS = PyArray_DIM(rays, 0);

    for (size_t r = 0; r < (size_t)N_RAYS; r++) {
        unsigned int* row = (unsigned int*)PyArray_GETPTR2(rays, r, 0);

        int x0 = x, y0 = y;
        int x1 = row[0], y1 = row[1];

        int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx + dy, e2;
        for (;;) {
            if (OCCUPIED_CELL(img, x0, y0)) {  // store collision coordinates and collision flag inplace
                row[0] = x0;
                row[1] = y0;
                row[2] = 1;
                break;
            }
            if (x0 == x1 && y0 == y1) break;
            e2 = 2 * err;
            if (e2 >= dy) {
                err += dy;
                x0 += sx;
            }
            if (e2 <= dx) {
                err += dx;
                y0 += sy;
            }
        }
    }
}

static PyObject* module_raycast(PyObject* module, PyObject* args) {
    PyObject* img_obj = NULL;
    PyObject* rays_obj = NULL;
    int x, y;

    if (!PyArg_ParseTuple(args, "OOii", &img_obj, &rays_obj, &x, &y))
        return NULL;

    PyArrayObject* img = (PyArrayObject*)PyArray_FROM_OTF(
        img_obj,
        NPY_UINT8,          // expands to NPY_UBYTE enum, data type is npy_ubyte
        NPY_ARRAY_IN_ARRAY  // this requires data alignment and c-style contiguous order ([row][col] indexing)
    );

    PyArrayObject* rays = (PyArrayObject*)PyArray_FROM_OTF(
        rays_obj,
        NPY_UINT32,
        NPY_ARRAY_INOUT_ARRAY);

    if (!CHECK_ARRAYS(img, rays))
        return NULL;

    _bresenham_raycast(img, rays, x, y);

    // update array inplace
    PyArray_ResolveWritebackIfCopy(rays);
    Py_DECREF(img);
    Py_DECREF(rays);

    Py_RETURN_NONE;
}

static PyMethodDef raycaster_methods[] = {
    {"raycast", module_raycast, METH_VARARGS, "Given an image and an array of ray coordinates, return an array of ray intersections"},
    {NULL, NULL, 0, NULL}  // sentinel
};

static struct PyModuleDef raycaster_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "raycaster",
    .m_doc = "Raycaster C implementation",
    .m_size = -1,
    .m_methods = raycaster_methods,
};

PyMODINIT_FUNC PyInit_raycaster(void) {
    import_array();
    return PyModule_Create(&raycaster_module);
}
