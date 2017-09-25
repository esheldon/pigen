from . import config
from . import funcwrap

class ModuleWrapper(dict):
    """
    module wrapper

    wrap the functions and create the python module C extension
    """
    def __init__(self, conf):
        self.conf = config.load_config(conf)

        self['modulename'] = self.conf['modulename']

        self._set_func_wraps()
        self._set_includes()
        self._set_text()

    def get_text(self):
        """
        get the full text for the C extension module
        """
        return self._text

    def _set_text(self):
        """
        Set the full text
        """
        self._text = _template % self

    def _set_includes(self):
        """
        set the includes if any
        """
        if 'includes' in self.conf:
            self['includes'] = '\n'.join(self.conf['includes'])
        else:
            self['includes'] = ''

    def _set_func_wraps(self):
        """
        get all the function wrappers and the py method
        definitions
        """
        prefix=self.conf['wrapper_prefix']
        prefix_vnames=self.conf['prefix_var_names'],
        
        funcwraps=[]
        for d in self.conf['functions']:
            fwrap = funcwrap.FuncWrapper(
                funcdef=d,
                prefix=self.conf['wrapper_prefix'],
                prefix_var_names=self.conf['prefix_var_names'],
            )
            funcwraps.append(fwrap)

        texts = [f.get_text() for f in funcwraps]
        self['all_function_texts'] = '\n\n'.join(texts)

        pydefs = ['    '+f.get_py_method_def() for f in funcwraps]
        self['py_method_defs'] = ',\n'.join(pydefs)


    def __repr__(self):
        return self.get_text()

_template="""
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include "numpy/arrayobject.h" 

%(includes)s

%(all_function_texts)s

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
