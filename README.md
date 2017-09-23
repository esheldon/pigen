# pigen
Extremely simple python interface generator for C code

Motivation
----------

Provide a very simple way to interfact with C code.  The goal here is to
improve performance and keep this wrapper code simple and easy.  The goal is
not to wrap existing code of any kind, but rather to wrap C code written
specifically to by easy to wrap.

Principles
----------

These principles are designed to keep this code simple.

- Each function wrapper is only that: a wrapper.  Its only 
  purpose is to provide inputs to the wrapped function.
- No memory allocation is allowed in the wrapper except by the use
  of Py_BuildValue for scalar return values.
- return values can only be scalars or None (void).  If the
  return value were a pointer, it would imply some kind
  of memory allocation. If you multiple return values are needed,
  send an array(s) and fill them.
- scalars inputs are translated directly to c types.
- pointer inputs are assumed to represent numpy arrays, and the
  underlying pointer is extracted useing PyArray_DATA.  It is
  checked that the input is a numpy array.
- For array arguments, it is the responsibility of the user to provide the correct
  inputs, no array type checking is performed.
- string arrays are provided translate because they are ambiguous
  with ordinary strings.  If string arrays are needed, have your
  function take a PyObject* as an argument and do the PyArray
  calls yourself

TODO
-------
- support strings char * using const char * as the type
