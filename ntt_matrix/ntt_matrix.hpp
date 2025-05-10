#pragma once
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <cmath>
#include <limits>

#if defined(NTT_MATRIX_STATIC)
#define NTT_MATRIX_API static
#elif defined(NTT_MATRIX_EXTERN)
#define NTT_MATRIX_API extern
#else
#define NTT_MATRIX_API
#endif

#ifdef NTT_MATRIX_IMPLEMENTATION
#include <cstring>
#include <cstdlib>
#endif

#ifdef NTT_MATRIX_IMPLEMENTATION
namespace
{
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

    bool isEqual(float a, float b);
#endif
        constexpr NTT_MATRIX_API value_type default_value = 0;

        class Matrix;

        typedef void (*sliding_callback)(
            size_t startRow,
            size_t startColumn,
            size_t endRow,
            size_t endColumn,
            Matrix &matrix,
            void *data);

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

            /**
             * The dot product operation between two matrices, which is different from the
             *      cross product operation.
             * @param other: the matrix to be multiplied.
             * @return: the result of the dot product operation.
             */
            Matrix dot(const Matrix &other);

            /**
             * The transpose operation of the matrix, which will swap the rows and columns of the matrix.
             * @return: the transposed matrix.
             */
            Matrix transpose();

            Matrix add(const Matrix &other);
            Matrix negative();
            Matrix subtract(const Matrix &other);

            Matrix add_padding(size_t padding);

            void sliding(sliding_callback callback,
                         size_t window_col,
                         size_t window_row,
                         size_t stride_col,
                         size_t stride_row,
                         void *data = nullptr);

            /**
             * Compare two matrices
             * @param other: the matrix to be compared.
             * @return: true if the matrices are equal, false otherwise.
             */
            bool operator==(const Matrix &other) const;

            Matrix operator+(const Matrix &other);
            Matrix operator+(value_type value);
            Matrix operator-(const Matrix &other);
            Matrix operator-(value_type value);
            Matrix operator*(const Matrix &other);
            Matrix operator*(value_type value);
            Matrix operator/(value_type value);

            std::string to_string() const;

            /**
             * Create a matrix from a vector of vectors.
             * @param vector: the vector of vectors to be converted.
             * @return: the created matrix.
             */
            static Matrix create_from_vector_vector(const std::vector<std::vector<value_type>> &vector);

            static Matrix create_identity_matrix(size_t size);

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
        bool isEqual(float a, float b)
        {
            return std::fabs(a - b) < std::numeric_limits<float>::epsilon();
        }

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

        Matrix Matrix::dot(const Matrix &other)
        {
            if (m_columns != other.m_rows)
                throw std::invalid_argument("The number of columns of the first matrix must be equal to the number of rows of the second matrix");

            Matrix result(m_rows, other.m_columns);

            for (size_t i = 0; i < m_rows; i++)
            {
                for (size_t j = 0; j < other.m_columns; j++)
                {
                    for (size_t k = 0; k < m_columns; k++)
                    {
                        result.set_element(i, j, result.get_element(i, j) + get_element(i, k) * other.get_element(k, j));
                    }
                }
            }

            return result;
        }

        Matrix Matrix::transpose()
        {
            Matrix result(m_columns, m_rows);

            for (size_t i = 0; i < m_rows; i++)
            {
                for (size_t j = 0; j < m_columns; j++)
                {
                    result.set_element(j, i, get_element(i, j));
                }
            }

            return result;
        }

        Matrix Matrix::add(const Matrix &other)
        {
            if (m_rows != other.m_rows || m_columns != other.m_columns)
                throw std::invalid_argument("The matrices must have the same size");

            Matrix result(m_rows, m_columns);

            for (size_t i = 0; i < m_rows; i++)
            {
                for (size_t j = 0; j < m_columns; j++)
                {
                    result.set_element(i, j, get_element(i, j) + other.get_element(i, j));
                }
            }

            return result;
        }

        Matrix Matrix::negative()
        {
            Matrix result(m_rows, m_columns);

            for (size_t i = 0; i < m_rows; i++)
            {
                for (size_t j = 0; j < m_columns; j++)
                {
                    result.set_element(i, j, -get_element(i, j));
                }
            }

            return result;
        }

        Matrix Matrix::subtract(const Matrix &other)
        {
            if (m_rows != other.m_rows || m_columns != other.m_columns)
                throw std::invalid_argument("The matrices must have the same size");

            Matrix result(m_rows, m_columns);
            for (size_t i = 0; i < m_rows; i++)
            {
                for (size_t j = 0; j < m_columns; j++)
                {
                    result.set_element(i, j, get_element(i, j) - other.get_element(i, j));
                }
            }

            return result;
        }

        Matrix Matrix::add_padding(size_t padding)
        {
            Matrix result(m_rows + padding * 2, m_columns + padding * 2, default_value);

            for (size_t i = padding; i < m_rows + padding; i++)
            {
                for (size_t j = padding; j < m_columns + padding; j++)
                {
                    result.set_element(i, j, get_element(i - padding, j - padding));
                }
            }

            return result;
        }

        void Matrix::sliding(sliding_callback callback,
                             size_t window_col,
                             size_t window_row,
                             size_t stride_col,
                             size_t stride_row,
                             void *data)
        {
            for (size_t i = 0; i < m_rows; i += stride_row)
            {
                for (size_t j = 0; j < m_columns; j += stride_col)
                {
                    callback(i, j, i + window_row, j + window_col, *this, data);
                }
            }
        }

        Matrix Matrix::operator-(const Matrix &other)
        {
            return subtract(other);
        }

        Matrix Matrix::operator-(value_type value)
        {
            return subtract(Matrix(m_rows, m_columns, value));
        }

        Matrix Matrix::operator+(const Matrix &other)
        {
            return add(other);
        }

        Matrix Matrix::operator+(value_type value)
        {
            return add(Matrix(m_rows, m_columns, value));
        }

        Matrix Matrix::operator*(const Matrix &other)
        {
            return dot(other);
        }

        Matrix Matrix::operator*(value_type value)
        {
            Matrix result(m_rows, m_columns);
            for (size_t i = 0; i < m_rows; i++)
            {
                for (size_t j = 0; j < m_columns; j++)
                {
                    result.set_element(i, j, get_element(i, j) * value);
                }
            }

            return result;
        }

        Matrix Matrix::operator/(value_type value)
        {
            Matrix result(m_rows, m_columns);
            for (size_t i = 0; i < m_rows; i++)
            {
                for (size_t j = 0; j < m_columns; j++)
                {
                    result.set_element(i, j, get_element(i, j) / value);
                }
            }

            return result;
        }

        bool Matrix::operator==(const Matrix &other) const
        {
            if (m_rows != other.m_rows || m_columns != other.m_columns)
                return false;

#if defined(NTT_MATRIX_I8) || defined(NTT_MATRIX_I16) || defined(NTT_MATRIX_I32) || defined(NTT_MATRIX_I64)
            return memcmp(m_data, other.m_data, m_rows * m_columns * sizeof(value_type)) == 0;
#else
            for (size_t i = 0; i < m_rows; i++)
            {
                for (size_t j = 0; j < m_columns; j++)
                {
                    if (!isEqual(get_element(i, j), other.get_element(i, j)))
                        return false;
                }
            }

            return true;
#endif // NTT_MATRIX_I8 || NTT_MATRIX_I16 || NTT_MATRIX_I32 || NTT_MATRIX_I64
        }
#endif // NTT_MATRIX_IMPLEMENTATION

        Matrix Matrix::create_from_vector_vector(const std::vector<std::vector<value_type>> &vector)
        {
            Matrix matrix(vector.size(), vector[0].size());

            for (size_t i = 0; i < vector.size(); i++)
            {
                for (size_t j = 0; j < vector[i].size(); j++)
                {
                    matrix.set_element(i, j, vector[i][j]);
                }
            }

            return matrix;
        }

        Matrix Matrix::create_identity_matrix(size_t size)
        {
            Matrix matrix(size, size);

            for (size_t i = 0; i < size; i++)
            {
                matrix.set_element(i, i, 1);
            }

            return matrix;
        }

        std::string Matrix::to_string() const
        {
            std::stringstream ss;
            for (size_t i = 0; i < m_rows; i++)
            {
                for (size_t j = 0; j < m_columns; j++)
                {
                    ss << get_element(i, j) << " ";
                }
                ss << "\n";
            }
            return ss.str();
        }
    } // namespace ntt

#ifdef NTT_MATRIX_IMPLEMENTATION
} // namespace
#endif