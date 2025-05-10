#pragma once
#include <cstdint>

#ifdef NTT_MATRIX_IMPLEMENTATION
#include <cstring>
#include <cstdlib>
#endif

namespace ntt
{

#if defined(NTT_MATRIX_I8)
    using value_type = int8_t;
#elif defined(NTT_MATRIX_I16)
    using value_type = int16_t;
    constexpr value_type default_value = 0;
#elif defined(NTT_MATRIX_I32)
    using value_type = int32_t;
    constexpr value_type default_value = 0;
#elif defined(NTT_MATRIX_I64)
    using value_type = int64_t;
    constexpr value_type default_value = 0;
#else
    using value_type = float;
#endif

    constexpr value_type default_value = 0;

    /**
     * The main object of the library, it will contains the whole information
     *      and data of a matrix, and will be used to perform the operations
     *      between the matrices. This library is designed specifically for the
     *      Neural Network library (header-only) project.
     */
    class Matrix
    {
    public:
        /**
         * Can only construct by firstly specifying the number of rows and columns. These
         *      values cannot be modified during the lifetime of the object.
         * @param rows: the number of rows of the matrix.
         * @param columns: the number of columns of the matrix.
         * @param defaultValue: the default value of the matrix.
         */
        Matrix(size_t rows, size_t columns, value_type defaultValue = default_value);

        /**
         * Copy constructor of the matrix.
         * @param other: the matrix to be copied.
         */
        Matrix(const Matrix &other);

        ~Matrix();

        inline size_t get_rows() const { return m_rows; }
        inline size_t get_columns() const { return m_columns; }
        inline const value_type get_element(size_t rowIndex, size_t columnIndex) const
        {
            return m_data[rowIndex * m_columns + columnIndex];
        }
        inline void set_element(size_t rowIndex, size_t columnIndex, value_type value)
        {
            m_data[rowIndex * m_columns + columnIndex] = value;
        }

    private:
        size_t m_rows;
        size_t m_columns;
        value_type *m_data;
    };

/**
 * The Macro which will be used for noticing the compiler which will create the
 *      implementation file. This pattern can be used without the use of the CMake
 *      to create the implementation file.
 * @todo: paraphase this comment again
 */
#ifdef NTT_MATRIX_IMPLEMENTATION
    Matrix::Matrix(size_t rows, size_t columns, value_type defaultValue)
        : m_rows(rows), m_columns(columns)
    {
        m_data = (value_type *)malloc(rows * columns * sizeof(value_type));

        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < columns; j++)
            {
                m_data[i * columns + j] = defaultValue;
            }
        }
    }

    Matrix::~Matrix()
    {
        free(m_data);
    }

    Matrix::Matrix(const Matrix &other)
        : m_rows(other.m_rows), m_columns(other.m_columns)
    {
        m_data = (value_type *)malloc(m_rows * m_columns * sizeof(value_type));
        memcpy(m_data, other.m_data, m_rows * m_columns * sizeof(value_type));
    }
#endif
} // namespace ntt
