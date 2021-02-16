#include <ruby.h>

#include <float.h>
#include <limits.h>

#define NUM2INT8(num) num2int8(num)
#define NUM2UINT8(num) num2uint8(num)
#define NUM2INT16(num) num2int16(num)
#define NUM2UINT16(num) num2uint16(num)
#if SIZEOF_INT32_T == SIZEOF_INT
#   define NUM2INT32(num) ((int32_t)NUM2INT(num))
#   define NUM2UINT32(num) ((uint32_t)NUM2UINT(num))
#elif SIZEOF_INT32_T == SIZEOF_LONG
#   define NUM2INT32(num) ((int32_t)NUM2LONG(num))
#   define NUM2UINT32(num) ((uint32_t)NUM2LLONG(num))
#else
#   define NUM2INT32(num) num2int32(num)
#   define NUM2UINT32(num) num2uint32(num)
static int32_t
num2int32(VALUE num)
{
  return (int32_t)int_range_check(NUM2LONG(num), INT32_MIN, INT32_MAX, "int32_t");
}

static uint32_t
num2uint32(VALUE num)
{
  return (uint32_t)uint_range_check(NUM2ULONG(num), UINT32_MAX, "uint32_t");
}
#endif
#if SIZEOF_INT64_T == SIZEOF_INT
#   define NUM2INT64(num) ((int64_t)NUM2INT(num))
#   define NUM2UINT64(num) ((uint64_t)NUM2UINT(num))
#elif SIZEOF_INT64_T == SIZEOF_LONG
#   define NUM2INT64(num) ((int64_t)NUM2LONG(num))
#   define NUM2UINT64(num) ((uint64_t)NUM2ULONG(num))
#elif SIZEOF_INT64_T == SIZEOF_LONG_LONG
#   define NUM2INT64(num) ((int64_t)NUM2LL(num))
#   define NUM2UINT64(num) ((uint64_t)NUM2ULL(num))
#else
#   error Unable to define NUM2INT64 and NUM2UINT64
#endif
#define NUM2FLT(num) num2flt(num)

static long
int_range_check(long num, long min, long max, const char *type)
{
  if (min <= num && num <= max) return num;
  rb_raise(rb_eRangeError, "integer %ld too %s to convert to `%s'",
           num, num < 0 ? "small" : "big", type);
}

static unsigned long
uint_range_check(unsigned long num, unsigned long max, const char *type)
{
  if (num > max) {
    rb_raise(rb_eRangeError, "integer %lu too big to convert to `%s'", num, type);
  }
  return num;
}

static int8_t
num2int8(VALUE num)
{
  return (int8_t)int_range_check(NUM2LONG(num), INT8_MIN, INT8_MAX, "int8_t");
}

static uint8_t
num2uint8(VALUE num)
{
  return (uint8_t)uint_range_check(NUM2ULONG(num), UINT8_MAX, "uint8_t");
}

static int16_t
num2int16(VALUE num)
{
  return (int16_t)int_range_check(NUM2LONG(num), INT16_MIN, INT16_MAX, "int16_t");
}

static uint16_t
num2uint16(VALUE num)
{
  return (uint16_t)uint_range_check(NUM2ULONG(num), UINT16_MAX, "uint16_t");
}

static float
num2flt(VALUE num)
{
  double dbl = NUM2DBL(num);
  if (dbl < FLT_MIN || FLT_MAX < dbl) {
    rb_raise(rb_eRangeError, "float %lf too %s to convert to `float'",
             dbl, dbl < 0 ? "small" : "big");
  }
  return (float)dbl;
}

VALUE mMemoryViewTestHelper;
VALUE cNDArray;

#define MAX_INLINE_DIM 32

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

static VALUE
ndarray_get_value(const uint8_t *value_ptr, const ndarray_dtype_t dtype)
{
  assert(value_ptr != NULL);
  switch (dtype) {
    case ndarray_dtype_int8:
      return INT2NUM(*(int8_t *)value_ptr);
    case ndarray_dtype_uint8:
      return UINT2NUM(*(uint8_t *)value_ptr);

    case ndarray_dtype_int16:
      return INT2NUM(*(int16_t *)value_ptr);
    case ndarray_dtype_uint16:
      return UINT2NUM(*(uint16_t *)value_ptr);

    case ndarray_dtype_int32:
      return LONG2NUM(*(int32_t *)value_ptr);
    case ndarray_dtype_uint32:
      return ULONG2NUM(*(uint32_t *)value_ptr);

    case ndarray_dtype_int64:
      return LL2NUM(*(int64_t *)value_ptr);
    case ndarray_dtype_uint64:
      return ULL2NUM(*(uint64_t *)value_ptr);

    case ndarray_dtype_float32:
      return DBL2NUM(*(float *)value_ptr);
    case ndarray_dtype_float64:
      return DBL2NUM(*(double *)value_ptr);

    default:
      return Qnil;
  }
}

static VALUE
ndarray_md_aref(const ndarray_t *nar, ssize_t *indices)
{
  assert(nar != NULL);
  assert(indices != NULL);

  /* assume the size of indices equals to nar->ndim */
  const ssize_t ndim = nar->ndim;

  uint8_t *value_ptr = nar->data;
  ssize_t i;
  for (i = 0; i < ndim; ++i) {
    value_ptr += indices[i] * nar->strides[i];
  }

  return ndarray_get_value(value_ptr, nar->dtype);
}

static VALUE
ndarray_aref(int argc, VALUE *argv, VALUE obj)
{
  ndarray_t *nar;
  TypedData_Get_Struct(obj, ndarray_t, &ndarray_data_type, nar);

  if (nar->ndim != argc) {
    rb_raise(rb_eIndexError, "index dimension mismatched (%d for %"PRIdSIZE")", argc, nar->ndim);
  }

  const int item_size = SIZEOF_DTYPE(nar->dtype);

  const ssize_t ndim = nar->ndim;
  if (ndim == 1) {
    /* special case for 1-D array */
    ssize_t i = NUM2SSIZET(argv[0]);
    uint8_t *p = ((uint8_t *)nar->data) + i * item_size;
    return ndarray_get_value(p, nar->dtype);
  }

  ssize_t indices[MAX_INLINE_DIM] = { 0, };

  if (ndim > MAX_INLINE_DIM) {
    rb_raise(rb_eNotImpError, "ndim > %d is unsupported now", MAX_INLINE_DIM);
  }

  ssize_t i;
  for (i = 0; i < ndim; ++i) {
    indices[i] = NUM2SSIZET(argv[i]);
  }

  VALUE val = ndarray_md_aref(nar, indices);

  return val;
}

static VALUE
ndarray_aset(int argc, VALUE *argv, VALUE obj)
{
  ndarray_t *nar;
  TypedData_Get_Struct(obj, ndarray_t, &ndarray_data_type, nar);

  rb_check_frozen(obj);

  if (nar->ndim != argc - 1) {
    rb_raise(rb_eIndexError, "index dimension mismatched (%d for %"PRIdSIZE")", argc - 1, nar->ndim);
  }

  const VALUE val = argv[argc-1];
  const int item_size = SIZEOF_DTYPE(nar->dtype);

  if (nar->ndim == 1) {
    /* special case for 1-D array */
    ssize_t i = NUM2SSIZET(argv[0]);
    uint8_t *p = ((uint8_t *)nar->data) + i * item_size;
    switch (nar->dtype) {
      case ndarray_dtype_int8:
        *(int8_t *)p = NUM2INT8(val);
        break;
      case ndarray_dtype_uint8:
        *(uint8_t *)p = NUM2UINT8(val);
        break;

      case ndarray_dtype_int16:
        *(int16_t *)p = NUM2INT16(val);
        break;
      case ndarray_dtype_uint16:
        *(uint16_t *)p = NUM2UINT16(val);
        break;

      case ndarray_dtype_int32:
        *(int32_t *)p = NUM2INT32(val);
      case ndarray_dtype_uint32:
        *(uint32_t *)p = NUM2UINT32(val);

      case ndarray_dtype_int64:
        *(int64_t *)p = NUM2INT64(val);
        break;
      case ndarray_dtype_uint64:
        *(uint64_t *)p = NUM2UINT64(val);
        break;

      case ndarray_dtype_float32:
        *(float *)p = NUM2FLT(val);
        break;
      case ndarray_dtype_float64:
        *(double *)p = NUM2DBL(val);
        break;

      default:
        return Qnil;
    }

    return val;
  }

  rb_raise(rb_eNotImpError, "multi-dimensional aset is unsupported now");
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
  rb_define_method(cNDArray, "[]", ndarray_aref, -1);
  rb_define_method(cNDArray, "[]=", ndarray_aset, -1);

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
