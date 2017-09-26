from __future__ import print_function, absolute_import
from .arguments import Argument, Arguments
from . import util

class FuncWrapper(dict):
    """
    function definition wrapper

    date a C function declaration and generate the python C 
    api wrapper code for it
    """
    def __init__(self, funcdef, prefix, prefix_var_names=False):

        self._set_funcdef(funcdef)
        self._prefix=prefix
        self._prefix_var_names=prefix_var_names

        self._set_defs()
        self._set_py_method_def()
        self._set_parse_tuple_call()
        self._set_function_call()

        self._combine()

    def get_text(self):
        """
        Get the text of the wrapper function definition
        """
        return self._text

    def get_py_method_def(self):
        """
        Get the text of the wrapper function definition
        """
        return self['py_method_def']


    def _set_funcdef(self, funcdef):
        if isinstance(funcdef, dict):
            self['prototype']=funcdef['def']
            self['doc'] = funcdef['doc']
        else:
            self['prototype']=funcdef
            self['doc'] = "no doc"

        self['prototype'] = self['prototype'].replace(';','')

    def _combine(self):
        """
        combine all the pieces into the wrapper function definition
        """
        args=self._args

        text_list=[
            self['wrapper_funcdef'],
            '{',
        ]

        if self['return_def'] is not None:
            text_list += ['',self['return_def']]

        if args is not None:
            if args['wrap_declarations'] is not None:
                text_list += ['',args['wrap_declarations']]
            if args['declarations'] is not None:
                text_list += ['',args['declarations']]

        if self._parse_tuple_call is not None:
            text_list += [
                '',
                self._parse_tuple_call,
            ]

        if args is not None and args['unwraps'] is not None:
            text_list += [
                '',
                args['unwraps'],
            ]
        text_list += [
            '',
            self._function_call,
            '',
            self['return_call'],
            '',
            '}',
        ]


        self._text='\n'.join(text_list)

    def _set_defs(self):
        """
        set the argument defs, return type def, function defs
        """

        # split into "type funcname" and "args)
        front,back = self['prototype'].replace(')','').split('(')

        # determine return type, and details of the overall
        # function wrapping. args are treated separately below

        func_wrap_return = FuncWrapAndReturn(front, self._prefix)
        self.update(func_wrap_return)

        # wrapping arguments to the user function
        if self._prefix_var_names:
            var_prefix=self['func_wrapper_name']
        else:
            var_prefix=None

        back=back.strip()
        if back=='void' or back=='':
            self._args=None
        else:

            arglist=[a.strip() for a in back.split(',')]
            self._args = Arguments(
                arglist,
                prefix=var_prefix,
            )

    def _set_py_method_def(self):
        """
        set the py_method def, what gets put into
        the module definition structure
        """
        if self._args is None:
            self['method_args_type'] = 'METH_NOARGS'
        else:
            self['method_args_type'] = 'METH_VARARGS'

        self['py_method_def'] = _py_method_def_template % self

    def _set_parse_tuple_call(self):
        """
        set the PyArg_ParseTuple call if needed
        """
        if self._args==None:
            self._parse_tuple_call=None
        else:
            self._parse_tuple_call = _parse_tuple_template % self._args

    def _set_function_call(self):
        """
        set the call to the wrapped function
        """
        if self._args is None:
            fargs=''
        else:
            fargs=self._args['function_args']
        fcall='%s( %s );' % (
            self['func_name'],
            fargs,
        )

        rname=self['return_var_name']
        if rname is not None:
            fcall = '%s = %s' % (rname, fcall)

        self._function_call='    %s' % fcall

           
    def __repr__(self):
        return self.get_text()

class FuncWrapAndReturn(dict):
    """
    extract information about the wrapped function
    """
    def __init__(self, front, prefix):
        if '*' in front and 'PyObject' not in front:
            raise RuntimeError("pointer return not supported, except PyObject*")

        self['return_type'], self['func_name'] = util.get_type_and_name(front)
        self['func_wrapper_name'] = '%s_%s' % (prefix, self['func_name'])
        self['wrapper_funcdef'] = _wrapper_funcdef_template % self


        self._set_return_var()
        self._set_return_call()

    def _set_return_var(self):
        """
        set up info for the return variable if it exists
        """
        if 'void' in self['return_type']:
            self['return_var_name']=None
            self['return_def']=None
        else:
            self['return_var_name']='%s_retval' % self['func_wrapper_name']
            self['return_def'] = '    %s %s;' % \
                    (self['return_type'], self['return_var_name'])

    def _set_return_call(self):
        """
        set the return call
        """
        if self['return_type'] == 'void':
            self['return_call'] = '    Py_RETURN_NONE;'
        elif 'PyObject' in self['return_type']:
            self['return_call']='    return %s;' % self['return_var_name']
        else:
            py_format = util.get_py_format_code(self['return_type'])
            self['return_call']='    return Py_BuildValue("%s", %s);' % \
                    (py_format, self['return_var_name'])
 

_wrapper_funcdef_template='PyObject* %(func_wrapper_name)s(PyObject* self, PyObject* args)'

_py_method_def_template=\
    '{"%(func_name)s",(PyCFunction)%(func_wrapper_name)s, %(method_args_type)s, "%(doc)s"}'


_parse_tuple_template="""
    if (!PyArg_ParseTuple(args, (char*)"%(py_formats)s", 
                          %(parse_tuple_args)s)) {
        return NULL;
    }"""


   
