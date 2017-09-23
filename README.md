# pigen
Extremely simple python interface generator for C code

Motivation
----------

Provide a very simple way to interface with C code.  The goal is to improve
performance while keeping this code simple and easy to understand.  It is not a
goal to wrap existing code of all kinds, but rather code carefully crafted to
be easy to wrap.

Principles
----------

These principles are designed to keep this code simple.  Some restrictions may
be lifted if an easy way to implement it is found.

- Each function wrapper is only that: a wrapper.  Its only 
  purpose is to provide inputs to the wrapped function.
- No memory allocation is allowed in the wrapper except by the use
  of Py_BuildValue for scalar return values.
- return values can only be scalars or None (void).  If the
  return value were a pointer, it would imply some kind
  of memory allocation. If multiple return values are needed,
  send an array(s) and fill them.
- Scalars inputs are translated directly to c types, including
  strings, but only as const char *.
- Pointer inputs are assumed to represent numpy arrays, and the
  underlying pointer is extracted using PyArray_DATA.  It is
  checked that the input is a numpy array.
- For array arguments, it is the responsibility of the user to provide the correct
  inputs, no array type checking is performed.
- String arrays are not provided translation because they are ambiguous
  with ordinary strings.  If string arrays are needed, have your
  function take a PyObject* as an argument and do the PyArray
  calls yourself

TODO
-------
- support strings char * using const char * as the type
