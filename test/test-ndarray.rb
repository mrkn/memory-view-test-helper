class NDArrayTest < Test::Unit::TestCase
  sub_test_case(".new") do
    test("called with shape and dtype") do
      ary = MemoryViewTestHelper::NDArray.new([2, 3], :float32)
      assert_equal({ shape: [2, 3],    ndim: 2,        dtype: :float32,  byte_size: 24 },
                   { shape: ary.shape, ndim: ary.ndim, dtype: ary.dtype, byte_size: ary.byte_size })
    end

    test("large ndim") do
      shape = [1]*100
      ary = MemoryViewTestHelper::NDArray.new(shape, :float32)
      assert_equal(shape, ary.shape)
    end
  end

  sub_test_case(".try_convert") do
    sub_test_case("called with 1-D array") do
      test("with dtype") do
        items = [1, 2, 3, 4, 5]
        ary = MemoryViewTestHelper::NDArray.try_convert(items, dtype: :float64)
        actual_items = 0.upto(4).map {|i| ary[i] }
        assert_equal({ shape: [5],       ndim: 1,        dtype: :float64,  byte_size: 40,            items: items.map(&:to_f) },
                     { shape: ary.shape, ndim: ary.ndim, dtype: ary.dtype, byte_size: ary.byte_size, items: actual_items })
      end

      test("without dtype") do
        ary1 = MemoryViewTestHelper::NDArray.try_convert([1,   2,   3])
        ary2 = MemoryViewTestHelper::NDArray.try_convert([1.0, 2,   3])
        ary3 = MemoryViewTestHelper::NDArray.try_convert([1,   2.0, 3])
        ary4 = MemoryViewTestHelper::NDArray.try_convert([1,   2,   3.0])
        assert_equal([:int64, :float64, :float64, :float64],
                     [ary1, ary2, ary3, ary4].map(&:dtype))
      end

      test("large ndim") do
        items = 99.times.inject([42, -8]) {|a, b| [a]}
        ary = MemoryViewTestHelper::NDArray.try_convert(items, dtype: :float64)
        preind = [0]*99
        assert_equal({ shape: [*[1]*99, 2], ndim: 100     , byte_size: 16           , value: [42.0, -8.0] },
                     { shape: ary.shape,    ndim: ary.ndim, byte_size: ary.byte_size, value: [ary[*preind, 0], ary[*preind, 1]] })
      end
    end

    sub_test_case("called with 2-D array and dtype") do
      def setup
        @items = [[1, 2], [3, 4], [5, 6]]
      end

      test("with order: :row_major") do
        ary = MemoryViewTestHelper::NDArray.try_convert(@items, dtype: :float64, order: :row_major)
        actual_items = 0.upto(2).map {|i| [ary[i, 0], ary[i, 1]] }
        strides = [2*ary.item_size, ary.item_size ]
        assert_equal({ shape: [3, 2],    strides: strides,     ndim: 2,        dtype: :float64,  byte_size: 48,            items: @items.map {|row| row.map(&:to_f) } },
                     { shape: ary.shape, strides: ary.strides, ndim: ary.ndim, dtype: ary.dtype, byte_size: ary.byte_size, items: actual_items })
      end

      test("with order: :column_major") do
        ary1 = MemoryViewTestHelper::NDArray.try_convert(@items, dtype: :float64, order: :column_major)
        ary2 = MemoryViewTestHelper::NDArray.try_convert(@items, dtype: :float64, order: :row_major)
        actual_items = 0.upto(2).map {|i| [ary1[i, 0], ary1[i, 1]] }
        strides = [ary1.item_size, 3*ary1.item_size ]
        assert_equal({ equality: true,         strides: strides,      items: @items.map {|row| row.map(&:to_f) } },
                     { equality: ary1 == ary2, strides: ary1.strides, items: actual_items })
      end

      test("without order") do
        ary2 = MemoryViewTestHelper::NDArray.try_convert(@items, dtype: :float64)
        ary1 = MemoryViewTestHelper::NDArray.try_convert(@items, dtype: :float64, order: :row_major)
        assert_equal(ary1.strides, ary2.strides)
      end
    end

    sub_test_case("called with 3-D array and dtype") do
      test("with order: :row_major") do
        items = [
                  [
                    [1, 2, 3, 4],
                    [2, 3, 4, 5],
                  ],
                  [
                    [3, 4, 5, 6],
                    [6, 7, 8, 9],
                  ]
                ]
        ary = MemoryViewTestHelper::NDArray.try_convert(items, dtype: :float64, order: :row_major)
        actual_items = 0.upto(1).map {|i| 0.upto(1).map {|j| 0.upto(3).map {|k| ary[i, j, k] } } }
        assert_equal({ shape: [2, 2, 4], ndim: 3,        dtype: :float64,  byte_size: 128,           items: items.map {|d0| d0.map {|d1| d1.map(&:to_f) } } },
                     { shape: ary.shape, ndim: ary.ndim, dtype: ary.dtype, byte_size: ary.byte_size, items: actual_items })
      end

      test("with order: :column_major") do
        omit("TODO")
      end

      test("without order") do
        omit("TODO")
      end
    end

    test("called without dtype") do
      ary = MemoryViewTestHelper::NDArray.try_convert([[1, 2], [3, 4]])
      assert_equal(:int64, ary.dtype)

      ary = MemoryViewTestHelper::NDArray.try_convert([[1, 2], [3.0, 4]])
      assert_equal(:float64, ary.dtype)

      assert_raise(TypeError) do
        MemoryViewTestHelper::NDArray.try_convert([[1, 2], ["3", 4]])
      end
    end

    test("error for giving inhomogeneous dimension array") do
      assert_raise(ArgumentError) do
        MemoryViewTestHelper::NDArray.try_convert([[[1], 2], [3, 4]])
      end
      assert_raise(ArgumentError) do
        MemoryViewTestHelper::NDArray.try_convert([[1, [2]], [3, 4]])
      end
      assert_raise(ArgumentError) do
        MemoryViewTestHelper::NDArray.try_convert([[1, 2], [[3], 4]])
      end
      assert_raise(ArgumentError) do
        MemoryViewTestHelper::NDArray.try_convert([[1, 2], [3, [4]]])
      end
    end
  end

  sub_test_case("#==") do
    sub_test_case("same dimension") do
      sub_test_case("compatible shape") do
        sub_test_case("1D arrays") do
          data do
            ary1 = MemoryViewTestHelper::NDArray.try_convert([1, 2, 3, 4], dtype: :int32)
            ary2 = MemoryViewTestHelper::NDArray.try_convert([1, 2, 3, 4], dtype: :float32)
            ary3 = MemoryViewTestHelper::NDArray.try_convert([1, 2, 3, 5], dtype: :int32)
            {
              "same array"       => [ary1, ary1, true],
              "int32 == float32" => [ary1, ary2, true],
              "int32 != int32"   => [ary1, ary3, false],
              "float32 != int32" => [ary2, ary3, false],
            }
          end
          def test_eq(data)
            ary1, ary2, eq = data
            if eq
              assert_equal(ary1, ary2)
            else
              assert_not_equal(ary1, ary2)
            end
          end
        end

        sub_test_case("2D arrays") do
          data do
            ary1 = MemoryViewTestHelper::NDArray.try_convert([[1, 2], [3, 4]], dtype: :int32)
            ary2 = MemoryViewTestHelper::NDArray.try_convert([[1, 2], [3, 4]], dtype: :float32)
            ary3 = MemoryViewTestHelper::NDArray.try_convert([[1, 2], [3, 5]], dtype: :int32)
            {
              "same array"       => [ary1, ary1, true],
              "int32 == float32" => [ary1, ary2, true],
              "int32 != int32"   => [ary1, ary3, false],
              "float32 != int32" => [ary2, ary3, false],
            }
          end
          def test_eq(data)
            ary1, ary2, eq = data
            if eq
              assert_equal(ary1, ary2)
            else
              assert_not_equal(ary1, ary2)
            end
          end
        end

        sub_test_case("large dimension") do
          data do
            base1 = [1, 2, 3]
            base2 = [1, 2, 4]
            ary1 = MemoryViewTestHelper::NDArray.try_convert(99.times.inject(base1){|a, b| [a] }, dtype: :int32)
            ary2 = MemoryViewTestHelper::NDArray.try_convert(99.times.inject(base1){|a, b| [a] }, dtype: :float64)
            ary3 = MemoryViewTestHelper::NDArray.try_convert(99.times.inject(base2){|a, b| [a] }, dtype: :int32)
            {
              "same array"       => [ary1, ary1, true],
              "int32 == float64" => [ary1, ary2, true],
              "int32 != int32"   => [ary1, ary3, false],
              "float64 != int32" => [ary2, ary3, false],
            }
          end
          def test_eq(data)
            ary1, ary2, eq = data
            if eq
              assert_equal(ary1, ary2)
            else
              assert_not_equal(ary1, ary2)
            end
          end
        end
      end

      sub_test_case("incompatible shape") do
        test("1D arrays") do
          ary1 = MemoryViewTestHelper::NDArray.try_convert([1, 2, 3, 4])
          ary2 = MemoryViewTestHelper::NDArray.try_convert([1, 2, 3, 4, 5])
          assert_not_equal(ary1, ary2)
        end

        sub_test_case("2D arrays") do
          data do
            ary1 = MemoryViewTestHelper::NDArray.try_convert([[1, 2], [3, 4]])
            ary2 = MemoryViewTestHelper::NDArray.try_convert([[1, 2, 3], [4, 5, 6]])
            ary3 = MemoryViewTestHelper::NDArray.try_convert([[1, 2], [3, 4], [5, 6]])
            ary4 = MemoryViewTestHelper::NDArray.try_convert([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            {
              "(2x2) != (2x3)" => [ary1, ary2],
              "(2x3) != (3x2)" => [ary2, ary3],
              "(3x2) != (2x2)" => [ary3, ary1],
              "(2x2) != (3x3)" => [ary1, ary4],
            }
          end
          def test_eq(data)
            ary1, ary2 = data
            assert_not_equal(ary1, ary2)
          end
        end

        sub_test_case("3D arrays") do
          data do
            ary1 = MemoryViewTestHelper::NDArray.try_convert([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
            ary2 = MemoryViewTestHelper::NDArray.try_convert([[[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]])
            {
              "(2x2x2) != (2x2x3)" => [ary1, ary2],
            }
          end
          def test_eq(data)
            ary1, ary2 = data
            assert_not_equal(ary1, ary2)
          end
        end
      end
    end

    sub_test_case("differnt dimensions") do
      sub_test_case("1D and 2D") do
        data do
          ary1 = MemoryViewTestHelper::NDArray.try_convert([1, 2, 3, 4, 5, 6])
          {
            "1D != 2D" => [ary1, ary1.reshape([2, 3])],
            "2D != 1D" => [ary1.reshape([2, 3]), ary1]
          }
        end
        def test_eq(data)
          ary1, ary2 = data
          assert_not_equal(ary1, ary2)
        end
      end
    end
  end

  sub_test_case("#reshape") do
    sub_test_case("base array is row_major") do
      test("order: :row_major") do
        ary1 = MemoryViewTestHelper::NDArray.try_convert([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype: :float64)
        ary2 = ary1.reshape([3, 3], order: :row_major)
        ary2[1, 1] = 12
        ary2_items = 0.upto(2).map {|i| 0.upto(2).map {|j| ary2[i, j] } }
        assert_equal({ shape: [3, 3]    , changed_value: 12.0   , ary2_items: [[1.0, 2.0, 3.0], [4.0, 12.0, 6.0], [7.0, 8.0, 9.0]] },
                     { shape: ary2.shape, changed_value: ary1[4], ary2_items: ary2_items })
      end

      test("large dimension and order: :row_major") do
        ary1 = MemoryViewTestHelper::NDArray.try_convert([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype: :float64)
        new_shape = [*[1]*98, 3, 3]
        ary2 = ary1.reshape(new_shape, order: :row_major)
        preind = [0]*98
        ary2_items = 0.upto(2).map {|i| 0.upto(2).map {|j| ary2[*preind, i, j] } }
        assert_equal({ shape: new_shape,  ary2_items: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]] },
                     { shape: ary2.shape, ary2_items: ary2_items })
      end
    end
  end
end
