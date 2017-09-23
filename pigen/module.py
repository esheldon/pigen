_template="""
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include "numpy/arrayobject.h" 

%(includes)s

%(all_functions)s

static PyMethodDef %(modulename)s_py_method_defs [] = {

%(py_method_defs)s

    {NULL}  /* Sentinel */
};


#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "%(modulename)s",      /* m_name */
        "Defines the funcs associated with module",  /* m_doc */
        -1,                  /* m_size */
        %(modulename)s_py_method_defs, /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_%(modulename)s(void)
#else
init%(modulename)s(void) 
#endif
{
    PyObject* m=NULL;

#if PY_MAJOR_VERSION >= 3

    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

#else

    m = Py_InitModule3("%(modulename)s", %(modulename)s_py_method_defs, "Define funcs.");
    if (m==NULL) {
        return;
    }
#endif

    // for numpy
    import_array();

#if PY_MAJOR_VERSION >= 3
    return m;
#endif

}
"""
