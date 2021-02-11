#include <ruby.h>

VALUE mMemoryViewTestHelper;
VALUE cNDArray;

typedef enum {
  ndarray_dtype_none = 0,
  ndarray_dtype_int8,
  ndarray_dtype_uint8,
  ndarray_dtype_int16,
  ndarray_dtype_uint16,
  ndarray_dtype_int32,
  ndarray_dtype_uint32,
  ndarray_dtype_int64,
  ndarray_dtype_uint64,
  ndarray_dtype_float32,
  ndarray_dtype_float64,

  ___ndarray_dtype_sentinel___
} ndarray_dtype_t;

#define NDARRAY_NUM_DTYPES ((int)___ndarray_dtype_sentinel___)

static const int ndarray_dtype_sizes[] = {
  0,
  sizeof(int8_t),
  sizeof(uint8_t),
  sizeof(int16_t),
  sizeof(uint16_t),
  sizeof(int32_t),
  sizeof(uint32_t),
  sizeof(int64_t),
  sizeof(uint64_t),
  sizeof(float),
  sizeof(double),
};

#define SIZEOF_DTYPE(type) (*(const int *)(&ndarray_dtype_sizes[type]))

static ID ndarray_dtype_ids[NDARRAY_NUM_DTYPES];

#define DTYPE_ID(type) (*(const ID *)(&ndarray_dtype_ids[type]))

static ndarray_dtype_t
ndarray_id_to_dtype_t(ID id)
{
  int i;
  for (i = 0; i < NDARRAY_NUM_DTYPES; ++i) {
    if (ndarray_dtype_ids[i] == id) {
      return (ndarray_dtype_t)i;
    }
  }
  rb_raise(rb_eArgError, "unknown dtype: %"PRIsVALUE, ID2SYM(id));
}

static ndarray_dtype_t
ndarray_sym_to_dtype_t(VALUE sym)
{
  assert(RB_TYPE_P(sym, T_SYMBOL));
  ID id = SYM2ID(sym);
  return ndarray_id_to_dtype_t(id);
}

static ndarray_dtype_t
ndarray_obj_to_dtype_t(VALUE obj)
{
  while (!RB_TYPE_P(obj, T_SYMBOL)) {
    if (RB_TYPE_P(obj, T_STRING) || rb_respond_to(obj, rb_intern("to_sym"))) {
      obj = rb_funcallv(obj, rb_intern("to_sym"), 0, NULL);
      if (!RB_TYPE_P(obj, T_SYMBOL)) {
        goto type_error;
      }
    }
    else if (!RB_TYPE_P(obj, T_STRING)) {
      if (!rb_respond_to(obj, rb_intern("to_str"))) {
        goto type_error;
      }
      else {
        obj = rb_funcallv(obj, rb_intern("to_str"), 0, NULL);
      }
    }
  }
  return ndarray_sym_to_dtype_t(obj);

type_error:
  rb_raise(rb_eTypeError, "dtype must be a symbol");
}

typedef struct {
  void *data;
  ssize_t byte_size;

  ndarray_dtype_t dtype;
  ssize_t ndim;
  ssize_t *shape;
  ssize_t *strides;
} ndarray_t;

static void ndarray_free(void *);
static size_t ndarray_memsize(const void *);

static const rb_data_type_t ndarray_data_type = {
  "memory-view-test-helper/ndarray",
  {
    0,
    ndarray_free,
    ndarray_memsize,
  },
  0, 0, RUBY_TYPED_FREE_IMMEDIATELY
};

static void
ndarray_free(void *ptr)
{
  ndarray_t *nar = (ndarray_t *)ptr;
  if (nar->data) xfree(nar->data);
  if (nar->shape) xfree(nar->shape);
  if (nar->strides) xfree(nar->strides);
  xfree(nar);
}

static size_t
ndarray_memsize(const void *ptr)
{
  ndarray_t *nar = (ndarray_t *)ptr;
  size_t size = sizeof(ndarray_t);
  if (nar->data) size += nar->byte_size;
  if (nar->shape) size += sizeof(ssize_t) * nar->ndim;
  if (nar->strides) size += sizeof(ssize_t) * nar->ndim;
  return size;
}

static VALUE
ndarray_s_allocate(VALUE klass)
{
  ndarray_t *nar;
  VALUE obj = TypedData_Make_Struct(klass, ndarray_t, &ndarray_data_type, nar);
  nar->data = NULL;
  nar->byte_size = 0;
  nar->dtype = ndarray_dtype_none;
  nar->ndim = 0;
  nar->shape = NULL;
  nar->strides = NULL;
  return obj;
}

static void
ndarray_init_row_major_strides(const ndarray_dtype_t dtype, const ssize_t ndim,
                               const ssize_t *shape, ssize_t *out_strides)
{
  const ssize_t item_size = SIZEOF_DTYPE(dtype);
  out_strides[ndim - 1] = item_size;

  int i;
  for (i = ndim - 1; i > 0; --i) {
    out_strides[i - 1] = out_strides[i] * shape[i];
  }
}

static VALUE
ndarray_initialize(VALUE obj, VALUE shape_ary, VALUE dtype_name)
{
  int i;

  Check_Type(shape_ary, T_ARRAY);

  const ssize_t ndim = (ssize_t)RARRAY_LEN(shape_ary);
  for (i = 0; i < ndim; ++i) {
    VALUE si = RARRAY_AREF(shape_ary, i);
    Check_Type(si, T_FIXNUM);
  }

  ssize_t *shape = ALLOC_N(ssize_t, ndim);
  for (i = 0; i < ndim; ++i) {
    VALUE si = RARRAY_AREF(shape_ary, i);
    shape[i] = NUM2SSIZET(si);
  }

  ndarray_dtype_t dtype = ndarray_obj_to_dtype_t(dtype_name);

  ssize_t *strides = ALLOC_N(ssize_t, ndim);
  ndarray_init_row_major_strides(dtype, ndim, shape, strides);

  ndarray_t *nar;
  TypedData_Get_Struct(obj, ndarray_t, &ndarray_data_type, nar);

  ssize_t byte_size = strides[0] * shape[0];
  nar->data = ALLOC_N(uint8_t, byte_size);
  nar->byte_size = byte_size;
  nar->dtype = dtype;
  nar->ndim = ndim;
  nar->shape = shape;
  nar->strides = strides;

  return Qnil;
}

static VALUE
ndarray_get_byte_size(VALUE obj)
{
  ndarray_t *nar;
  TypedData_Get_Struct(obj, ndarray_t, &ndarray_data_type, nar);

  return SSIZET2NUM(nar->byte_size);
}

static VALUE
ndarray_get_dtype(VALUE obj)
{
  ndarray_t *nar;
  TypedData_Get_Struct(obj, ndarray_t, &ndarray_data_type, nar);

  if (ndarray_dtype_none < nar->dtype && nar->dtype < NDARRAY_NUM_DTYPES) {
    return ID2SYM(DTYPE_ID(nar->dtype));
  }
  return Qnil;
}

static VALUE
ndarray_get_ndim(VALUE obj)
{
  ndarray_t *nar;
  TypedData_Get_Struct(obj, ndarray_t, &ndarray_data_type, nar);

  return SSIZET2NUM(nar->ndim);
}

static VALUE
ndarray_get_shape(VALUE obj)
{
  ndarray_t *nar;
  TypedData_Get_Struct(obj, ndarray_t, &ndarray_data_type, nar);

  VALUE ary = rb_ary_new_capa(nar->ndim);
  int i;
  for (i = 0; i < nar->ndim; ++i) {
    rb_ary_push(ary, SSIZET2NUM(nar->shape[i]));
  }

  return ary;
}

void
Init_memory_view_test_helper(void)
{
  mMemoryViewTestHelper = rb_define_module("MemoryViewTestHelper");
  cNDArray = rb_define_class_under(mMemoryViewTestHelper, "NDArray", rb_cObject);

  rb_define_alloc_func(cNDArray, ndarray_s_allocate);
  rb_define_method(cNDArray, "initialize", ndarray_initialize, 2);
  rb_define_method(cNDArray, "byte_size", ndarray_get_byte_size, 0);
  rb_define_method(cNDArray, "dtype", ndarray_get_dtype, 0);
  rb_define_method(cNDArray, "ndim", ndarray_get_ndim, 0);
  rb_define_method(cNDArray, "shape", ndarray_get_shape, 0);

  ndarray_dtype_ids[ndarray_dtype_int8] = rb_intern("int8");
  ndarray_dtype_ids[ndarray_dtype_uint8] = rb_intern("uint8");
  ndarray_dtype_ids[ndarray_dtype_int16] = rb_intern("int16");
  ndarray_dtype_ids[ndarray_dtype_uint16] = rb_intern("uint16");
  ndarray_dtype_ids[ndarray_dtype_int32] = rb_intern("int32");
  ndarray_dtype_ids[ndarray_dtype_uint32] = rb_intern("uint32");
  ndarray_dtype_ids[ndarray_dtype_int64] = rb_intern("int64");
  ndarray_dtype_ids[ndarray_dtype_uint64] = rb_intern("uint64");
  ndarray_dtype_ids[ndarray_dtype_float32] = rb_intern("float32");
  ndarray_dtype_ids[ndarray_dtype_float64] = rb_intern("float64");

  (void)ndarray_dtype_sizes; /* TODO: to be deleted */
}
