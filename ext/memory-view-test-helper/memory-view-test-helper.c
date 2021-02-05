#include <ruby.h>

VALUE mMemoryViewTestHelper;
VALUE cNDArray;

void
Init_memory_view_test_helper(void)
{
  mMemoryViewTestHelper = rb_define_module("MemoryViewTestHelper");
  cNDArray = rb_define_class_under(mMemoryViewTestHelper, "NDArray", rb_cObject);
}
