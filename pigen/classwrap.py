from __future__ import print_function, absolute_import
from .arguments import Argument, Arguments
from . import util
from . import funcwrap

class ClassWrapper(dict):
    """
    function definition wrapper

    date a C function declaration and generate the python C 
    api wrapper code for it
    """
    def __init__(self, modulename, classname, classdef, prefix, prefix_var_names=False):

        self['modulename'] = modulename
        self['classname'] = classname

        self._prefix=prefix
        self._prefix_var_names=prefix_var_names
        self._set_classdef(classdef)

        self._set_structdef()
        self._set_constructor()
        self._set_destructor()
        self._set_class_wrapper()

        self._set_defs()
        self._set_py_method_defs()
        self._set_parse_tuple_call()

        self._combine()

    def _set_classdef(self, classdef):
        self['wrapper_struct_name'] = '%s%s' % \
                (self._prefix, self['classname'])

        self['wrapper_type_name'] = '%(wrapper_struct_name)sType' % self
        self['py_methods_name'] = '%(wrapper_struct_name)s_methods' % self

        classdef['pydef_remove_prefix'] = \
                classdef.get('pydef_remove_prefix',None)
        classdef['doc'] = classdef.get('doc','No docs provided')

        self._classdef=classdef

    def _set_structdef(self):
        """
        set the basic structure definition
        """
        self._structdef = _class_struct_template % self

    def _set_constructor(self):
        """
        Use a FuncWrapper, but just pull out the parts we need, since
        init functions are special
        """

        self['init_name'] = '%(wrapper_struct_name)s_init' % self
        wrapper=funcwrap.FunctionWrapper(
            self._classdef['constructor'],
        )

        # the PyArgs_ParseTuple command, if needed
        args = wrapper.get_args()
        if args is not None:
            self['init_parse_tuple']=_init_parse_tuple_template % args
        else:
            self['init_parse_tuple'] = ""

        # calling the user's constructor
        fsplit = wrapper.get_function_call().split('=')
        if len(fsplit) < 2:
            raise ValueError("constructor does not return anything: '%s'" % \
                             self._classdef['constructor'])

        # we will set self->data = this call
        self['constructor_call'] = fsplit[1].strip()

        self._constructor_text = _class_init_template % self

    def _set_destructor(self):
        """
        Use a FuncWrapper, but just pull out the parts we need, since
        destructor functions are special
        """
        self['dealloc_name'] = '%(wrapper_struct_name)s_dealloc' % self
        wrapper=funcwrap.FunctionWrapper(
            self._classdef['destructor'],
        )

        args = wrapper.get_args()
        if args is None or len(args) != 1:
            raise ValueError("destructor should take one arg: '%s'" \
                             self._classdef['destructor'])

        self['destructor_call'] = '    %s(self->data);' % wrapper['func_name']
        self._destructor_text = _class_dealloc_template % self

    def _set_class_wrapper(self):
        """
        set the basic structure definition
        """
        self._class_wrapper = _class_wrapper_template % self


_class_struct_template="""
struct %(wrapper_struct_name)s {
    PyObject_HEAD
    void *data;
};
"""

_init_parse_tuple_template="""
    if (!PyArg_ParseTuple(args_pyobj, (char*)"%(py_formats)s", 
                          %(parse_tuple_args)s)) {
        return -1;
    }
"""

_class_init_template = """
static int
%(init_name)s(struct %(wrapper_struct_name)s* self, PyObject *args_pyobj)
{

%(init_parse_tuple)s

    self->data = %(constructor_call)s

    return 0;
}
"""

_class_dealloc_template="""
static void
%(dealloc_name)s(struct %(wrapper_struct_name)s* self)
{

%(destructor_call)s

#if PY_MAJOR_VERSION >= 3
    Py_TYPE(self)->tp_free((PyObject*)self);
#else
    self->ob_type->tp_free((PyObject*)self);
#endif

}
"""

_class_wrapper_template="""
static PyTypeObject %(wrapper_type_name)s = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "%(modulename)s.%(classname)s",            /*tp_name*/
    sizeof(struct %(wrapper_struct_name)s), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)%(dealloc_name)s,  /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "%(doc)s",             /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    %(py_methods_name)s,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)%(init_name)s,      /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,                 /* tp_new */
};
"""
