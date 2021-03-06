require "memory_view_test_helper.so"
require "memory-view-test-helper/version"
require "set"

module MemoryViewTestHelper
  class NDArray
    class << self
      alias __new__ new
    end

    def self.new(shape, dtype, order: :row_major)
      __new__(shape, dtype, order)
    end

    def self.try_convert(obj, dtype: nil, order: :row_major)
      begin
        ary = obj.to_ary
      rescue TypeError
        raise ArgumentError, "the argument must be converted to an Array by to_ary (#{obj.class} given)"
      end

      dtype, shape, cache = detect_dtype_and_shape(ary, dtype)
      nar = __new__(shape, dtype, order)
      assign_cache(nar, cache)
      return nar
    end

    private_class_method def self.assign_cache(nar, cache)
      if nar.ndim == 1
        src = cache[0][:ary]
        src.each_with_index do |x, i|
          nar[i] = x
        end
      else
        assign_cache_recursive(nar, [], cache, 0)
      end
    end

    private_class_method def self.assign_cache_recursive(nar, idx, cache, k)
      if cache[k][:dim]+1 != nar.ndim
        cache[k][:ary].each_index do |i|
          k = assign_cache_recursive(nar, [*idx, i], cache, k+1)
        end
      else
        cache[k][:ary].each_with_index do |x, i|
          nar[*idx, i] = x
        end
      end
      k
    end

    private_class_method def self.detect_dtype_and_shape(ary, dtype)
      current_dim = ary.length
      shape = []
      cache = []
      _, dtype, shape, cache = detect_dtype_and_shape_recursive(ary, 0, nil, dtype, shape, cache)
      return dtype, shape, cache
    end

    private_class_method def self.detect_dtype_and_shape_recursive(obj, dim, max_dim, fixed_dtype, out_shape, conversion_cache)
      dtype = detect_dtype(obj)
      unless dtype.nil?
        # obj is scalar
        # TODO handle scalar object
        if max_dim.nil?
          max_dim = dim # update max_dim
        elsif dim != max_dim
          dim_failed = [dim, max_dim].min
          raise ArgumentError, "inhomogeneous array detected at the the #{dim_failed}#{ordinal(dim_failed)} dimension"
        end
        return max_dim, dtype, out_shape, conversion_cache
      end

      # obj is array-like
      ary = Array(obj)
      conversion_cache << {obj:obj, ary:ary, dim:dim}

      dim_size = ary.length
      if out_shape.length <= dim
        # update_shape
        out_shape[dim] = dim_size
      elsif out_shape[dim] != dim_size
        raise ArgumentError, "size mismatch at the #{dim}#{ordinal(dim)} dimension (#{dim_size} for #{out_shape[dim]})"
      end

      # recursive detection
      ary.each do |sub|
        max_dim, dtype_sub, = detect_dtype_and_shape_recursive(sub, dim + 1, max_dim, fixed_dtype, out_shape, conversion_cache)
        dtype = promote_dtype(dtype, dtype_sub) unless fixed_dtype
      end

      return max_dim, (fixed_dtype || dtype), out_shape, conversion_cache
    end

    private_class_method def self.detect_dtype(obj)
      case obj
      when Integer
        :int64
      when Float, Rational
        :float64
      when ->(x) { x.is_a?(Complex) && x.imag == 0 }
        detect_dtype(x.real)
      when Enumerable, proc { obj.respond_to?(:to_ary) }
        nil
      else
        raise TypeError, "#{obj.class} is unsupported"
      end
    end

    INTEGER_TYPES = Set[:int8, :uint8, :int16, :uint16, :int32, :uint32, :int64, :uint64].freeze

    SIZEOF_DTYPE = {
      int8:    1,  uint8:  1,
      int16:   2,  uint16: 2,
      int32:   4,  uint32: 4,
      int64:   8,  uint64: 8,
      float32: 4,
      float64: 8
    }.freeze

    private_class_method def self.promote_dtype(dtype_a, dtype_b)
      # TODO: use sizeof
      if dtype_a == dtype_b
        dtype_a
      elsif dtype_a.nil? || dtype_b.nil?
        dtype_a || dtype_b
      else
        sizeof_a = SIZEOF_DTYPE[dtype_a]
        sizeof_b = SIZEOF_DTYPE[dtype_b]

        if INTEGER_TYPES.include?(dtype_a) && INTEGER_TYPES.include?(dtype_b)
          # both are integer
          if sizeof_a > sizeof_b
            dtype_a
          elsif sizeof_b > sizeof_a
            dtype_b
          else
            raise TypeError, "auto promotion between signed and unsigned is not supported"
          end
        elsif INTEGER_TYPES.include?(dtype_a)
          # b is float
          dtype_b
        elsif INTEGER_TYPES.include?(dtype_b)
          # a is float
          dtype_a
        else
          # both are float
          sizeof_a > sizeof_b ? dtype_a : dtype_b
        end
      end
    end

    private_class_method def self.ordinal(n)
      case n % 10
      when 1
        n != 11 ? "st" : "th"
      when 2
        n != 12 ? "nd" : "th"
      when 3
        n != 13 ? "rd" : "th"
      else
        "th"
      end
    end

    def item_size
      SIZEOF_DTYPE[dtype]
    end

    def reshape(new_shape, order: :row_major)
      reshape_impl(new_shape.to_ary, order.to_sym)
    end
  end
end
