#pragma once
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <cmath>
#include <exception>
#include <limits>

#if defined(NTT_MICRO_NN_STATIC)
#define NTT_MICRO_NN_API static
#elif defined(NTT_MICRO_NN_EXTERN)
#define NTT_MICRO_NN_API extern
#else
#define NTT_MICRO_NN_API
#endif

#ifdef NTT_MICRO_NN_IMPLEMENTATION
#include <cstring>
#include <cstdlib>
#endif

#ifdef NTT_MICRO_NN_IMPLEMENTATION
namespace
{
#endif
    namespace ntt
    {

#if defined(NTT_MICRO_NN_I8)
        using value_type = int8_t;
#elif defined(NTT_MICRO_NN_I16)
    using value_type = int16_t;
    constexpr value_type default_value = 0;
#elif defined(NTT_MICRO_NN_I32)
    using value_type = int32_t;
    constexpr value_type default_value = 0;
#elif defined(NTT_MICRO_NN_I64)
    using value_type = int64_t;
    constexpr value_type default_value = 0;
#else
    using value_type = float;

    bool isEqual(float a, float b);
#endif
        constexpr NTT_MICRO_NN_API value_type default_value = 0;

        class Tensor;

        typedef void (*sliding_callback)(
            size_t startRow,
            size_t startColumn,
            size_t endRow,
            size_t endColumn,
            Tensor &matrix,
            void *data);

        /**
         * The main object of the library, it will contains the whole information
         *      and data of a matrix, and will be used to perform the operations
         *      between the matrices. This library is designed specifically for the
         *      Neural Network library (header-only) project.
         */
        class Tensor
        {
        public:
            /**
             * Can only construct by firstly specifying the number of rows and columns. These
             *      values cannot be modified during the lifetime of the object.
             * @param rows: the number of rows of the matrix.
             * @param columns: the number of columns of the matrix.
             * @param defaultValue: the default value of the matrix.
             * @param name: the name of the matrix (for debugging purposes).
             */
            Tensor(size_t rows, size_t columns, value_type defaultValue = default_value);

            /**
             * Copy constructor of the matrix.
             * @param other: the matrix to be copied.
             */
            Tensor(const Tensor &other);

            ~Tensor();

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
            Tensor dot(const Tensor &other);

            /**
             * The transpose operation of the matrix, which will swap the rows and columns of the matrix.
             * @return: the transposed matrix.
             */
            Tensor transpose();

            void reshape(size_t rows, size_t columns);
            Tensor toShape(size_t rows, size_t columns);

            Tensor add(const Tensor &other);
            Tensor negative();
            Tensor subtract(const Tensor &other);

            Tensor add_padding(size_t padding);

            Tensor cross_correlation(const Tensor &other, size_t stride = 1);

            enum Axis
            {
                ROW = 0,
                COLUMN = 1,
                MATRIX = 2
            };

            /**
             * The max operation of the matrix, which will return the maximum value of the matrix.
             * @param axis: the axis to be used for the max operation. ROW for rows, COLUMN for columns and MATRIX for
             *      the whole matrix.
             * @return: the maximum value of the matrix.
             */
            Tensor max(Axis axis = Axis::MATRIX);

            size_t argmax();

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
            bool operator==(const Tensor &other) const;
            void operator=(const Tensor &other);

            Tensor operator+(const Tensor &other);
            Tensor operator+(value_type value);
            Tensor operator-(const Tensor &other);
            Tensor operator-(value_type value);
            Tensor operator*(const Tensor &other);
            Tensor operator*(value_type value);
            Tensor operator/(value_type value);

            std::string to_string() const;

            /**
             * Create a matrix from a vector of vectors.
             * @param vector: the vector of vectors to be converted.
             * @return: the created matrix.
             */
            static Tensor create_from_vector_vector(const std::vector<std::vector<value_type>> &vector);

            static Tensor create_from_vector(const std::vector<value_type> &vector);

            static Tensor create_identity_matrix(size_t size);

        private:
            size_t m_rows;
            size_t m_columns;
            value_type *m_data;
        };

        /**
         * Base class for all layers of the neural network.
         */
        class Layer
        {
        public:
            virtual Tensor forward(const Tensor &input) = 0;
        };

        class FullyConnectedLayer : public Layer
        {
        public:
            FullyConnectedLayer(size_t input_size, size_t output_size);
            FullyConnectedLayer(const Tensor &weights, const Tensor &biases);
            FullyConnectedLayer(const Tensor &weights, value_type biases);
            ~FullyConnectedLayer();

            virtual Tensor forward(const Tensor &input) override;

        private:
            Tensor m_weights;
            Tensor m_biases;
        };

        class ReLU : public Layer
        {
        public:
            virtual Tensor forward(const Tensor &input) override;
        };

        class Softmax : public Layer
        {
        public:
            virtual Tensor forward(const Tensor &input) override;
        };

        class ClipLayer : public Layer
        {
        public:
            ClipLayer(value_type min, value_type max);
            virtual Tensor forward(const Tensor &input) override;

        private:
            value_type m_min;
            value_type m_max;
        };

        class Sigmoid : public Layer
        {
        public:
            virtual Tensor forward(const Tensor &input) override;
        };

/**
 * The Macro which will be used for noticing the compiler which will create the
 *      implementation file. This pattern can be used without the use of the CMake
 *      to create the implementation file.
 * @todo: paraphase this comment again
 */
#ifdef NTT_MICRO_NN_IMPLEMENTATION
        bool isEqual(float a, float b)
        {
            return std::fabs(a - b) < std::numeric_limits<float>::epsilon();
        }

        Tensor::Tensor(size_t rows, size_t columns, value_type defaultValue)
            : m_rows(rows), m_columns(columns), m_data(nullptr)
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

        Tensor::~Tensor()
        {
            if (m_data != nullptr)
            {
                free(m_data);
                m_data = nullptr;
            }
        }

        Tensor::Tensor(const Tensor &other)
            : m_rows(other.m_rows), m_columns(other.m_columns)
        {
            m_data = (value_type *)malloc(m_rows * m_columns * sizeof(value_type));
            memcpy(m_data, other.m_data, m_rows * m_columns * sizeof(value_type));
        }

        Tensor Tensor::dot(const Tensor &other)
        {
            if (m_columns != other.m_rows)
            {
                char message[512];
                snprintf(
                    message, sizeof(message),
                    "The first matrix has size (%zu, %zu) which is not matching "
                    "with the second matrix with size (%zu, %zu)",
                    m_rows, m_columns, other.m_rows, other.m_columns);
                throw std::invalid_argument(message);
            }

            Tensor result(m_rows, other.m_columns);

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

        Tensor Tensor::transpose()
        {
            Tensor result(m_columns, m_rows);

            for (size_t i = 0; i < m_rows; i++)
            {
                for (size_t j = 0; j < m_columns; j++)
                {
                    result.set_element(j, i, get_element(i, j));
                }
            }

            return result;
        }

        void Tensor::reshape(size_t rows, size_t columns)
        {
            if (rows * columns != m_rows * m_columns)
            {
                char message[256];
                snprintf(
                    message, sizeof(message),
                    "The new size (%zu, %zu) does not match with the previous size (%zu, %zu)",
                    rows, columns, m_rows, m_columns);

                throw std::invalid_argument(message);
            }

            m_rows = rows;
            m_columns = columns;
        }

        Tensor Tensor::toShape(size_t rows, size_t columns)
        {
            Tensor result(*this);
            result.reshape(rows, columns);
            return result;
        }

        Tensor Tensor::add(const Tensor &other)
        {
            if (m_rows != other.m_rows || m_columns != other.m_columns)
            {
                char message[256];
                snprintf(
                    message, sizeof(message),
                    "The matrix with the size (%zu, %zu) cannot be added to the matrix with the size (%zu, %zu)",
                    m_rows, m_columns, other.m_rows, other.m_columns);
                throw std::invalid_argument(message);
            }

            Tensor result(m_rows, m_columns);

            for (size_t i = 0; i < m_rows; i++)
            {
                for (size_t j = 0; j < m_columns; j++)
                {
                    result.set_element(i, j, get_element(i, j) + other.get_element(i, j));
                }
            }

            return result;
        }

        Tensor Tensor::negative()
        {
            Tensor result(m_rows, m_columns);

            for (size_t i = 0; i < m_rows; i++)
            {
                for (size_t j = 0; j < m_columns; j++)
                {
                    result.set_element(i, j, -get_element(i, j));
                }
            }

            return result;
        }

        Tensor Tensor::subtract(const Tensor &other)
        {
            if (m_rows != other.m_rows || m_columns != other.m_columns)
            {
                char message[256];
                snprintf(
                    message, sizeof(message),
                    "The matrix with the size (%zu, %zu) cannot be subtracted from the matrix with the size (%zu, %zu)",
                    m_rows, m_columns, other.m_rows, other.m_columns);
                throw std::invalid_argument(message);
            }

            Tensor result(m_rows, m_columns);
            for (size_t i = 0; i < m_rows; i++)
            {
                for (size_t j = 0; j < m_columns; j++)
                {
                    result.set_element(i, j, get_element(i, j) - other.get_element(i, j));
                }
            }

            return result;
        }

        Tensor Tensor::add_padding(size_t padding)
        {
            Tensor result(m_rows + padding * 2, m_columns + padding * 2, default_value);

            for (size_t i = padding; i < m_rows + padding; i++)
            {
                for (size_t j = padding; j < m_columns + padding; j++)
                {
                    result.set_element(i, j, get_element(i - padding, j - padding));
                }
            }

            return result;
        }

        Tensor Tensor::cross_correlation(const Tensor &other, size_t stride)
        {
            size_t resultRows = (m_rows - other.m_rows) / stride + 1;
            size_t resultColumns = (m_columns - other.m_columns) / stride + 1;
            Tensor result(resultRows, resultColumns, default_value);

            for (size_t i = 0; i < resultRows; i++)
            {
                for (size_t j = 0; j < resultColumns; j++)
                {
                    size_t indexRow = i * stride;
                    size_t indexColumn = j * stride;

                    for (size_t k = 0; k < other.m_rows; k++)
                    {
                        for (size_t l = 0; l < other.m_columns; l++)
                        {
                            result.set_element(
                                i, j,
                                result.get_element(i, j) +
                                    get_element(indexRow + k, indexColumn + l) *
                                        other.get_element(k, l));
                        }
                    }
                }
            }

            return result;
        }

        void Tensor::sliding(sliding_callback callback,
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

        Tensor Tensor::max(Axis axis)
        {
            if (axis == Axis::MATRIX)
            {
                Tensor result(1, 1);
                for (size_t i = 0; i < m_rows; i++)
                {
                    for (size_t j = 0; j < m_columns; j++)
                    {
                        if (result.get_element(0, 0) < get_element(i, j))
                        {
                            result.set_element(0, 0, get_element(i, j));
                        }
                    }
                }
                return result;
            }

            if (axis == Axis::ROW)
            {
                Tensor result(m_rows, 1);
                for (size_t i = 0; i < m_rows; i++)
                {
                    for (size_t j = 0; j < m_columns; j++)
                    {
                        if (result.get_element(i, 0) < get_element(i, j))
                        {
                            result.set_element(i, 0, get_element(i, j));
                        }
                    }
                }

                return result;
            }

            if (axis == Axis::COLUMN)
            {
                Tensor result(1, m_columns);
                for (size_t i = 0; i < m_columns; i++)
                {
                    for (size_t j = 0; j < m_rows; j++)
                    {
                        if (result.get_element(0, i) < get_element(j, i))
                        {
                            result.set_element(0, i, get_element(j, i));
                        }
                    }
                }
                return result;
            }
            return Tensor(m_rows, m_columns);
        }

        size_t Tensor::argmax()
        {
            if (m_columns != 1 && m_rows != 1)
            {
                char message[256];
                snprintf(
                    message, sizeof(message),
                    "The matrix with the size (%zu, %zu) cannot be used for the argmax operation"
                    ", one of them must be 1",
                    m_rows, m_columns);
                throw std::invalid_argument(message);
            }

            size_t maxIndex = 0;

            if (m_rows == 1)
            {
                value_type maxValue = get_element(0, 0);

                for (size_t i = 0; i < m_columns; i++)
                {
                    if (get_element(0, i) > maxValue)
                    {
                        maxValue = get_element(0, i);
                        maxIndex = i;
                    }
                }
            }

            if (m_columns == 1)
            {
                value_type maxValue = get_element(0, 0);

                for (size_t i = 0; i < m_rows; i++)
                {
                    if (get_element(i, 0) > maxValue)
                    {
                        maxValue = get_element(i, 0);
                        maxIndex = i;
                    }
                }
            }

            return maxIndex;
        }

        Tensor Tensor::operator-(const Tensor &other)
        {
            return subtract(other);
        }

        Tensor Tensor::operator-(value_type value)
        {
            return subtract(Tensor(m_rows, m_columns, value));
        }

        Tensor Tensor::operator+(const Tensor &other)
        {
            return add(other);
        }

        Tensor Tensor::operator+(value_type value)
        {
            return add(Tensor(m_rows, m_columns, value));
        }

        Tensor Tensor::operator*(const Tensor &other)
        {
            return dot(other);
        }

        Tensor Tensor::operator*(value_type value)
        {
            Tensor result(m_rows, m_columns);
            for (size_t i = 0; i < m_rows; i++)
            {
                for (size_t j = 0; j < m_columns; j++)
                {
                    result.set_element(i, j, get_element(i, j) * value);
                }
            }

            return result;
        }

        Tensor Tensor::operator/(value_type value)
        {
            Tensor result(m_rows, m_columns);
            for (size_t i = 0; i < m_rows; i++)
            {
                for (size_t j = 0; j < m_columns; j++)
                {
                    result.set_element(i, j, get_element(i, j) / value);
                }
            }

            return result;
        }

        bool Tensor::operator==(const Tensor &other) const
        {
            if (m_rows != other.m_rows || m_columns != other.m_columns)
                return false;

#if defined(NTT_MICRO_NN_I8) || defined(NTT_MICRO_NN_I16) || defined(NTT_MICRO_NN_I32) || defined(NTT_MICRO_NN_I64)
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
#endif // NTT_MICRO_NN_I8 || NTT_MICRO_NN_I16 || NTT_MICRO_NN_I32 || NTT_MICRO_NN_I64
        }
#endif // NTT_MICRO_NN_IMPLEMENTATION

        void Tensor::operator=(const Tensor &other)
        {
            if (m_data != nullptr)
            {
                free(m_data);
                m_data = nullptr;
            }

            m_rows = other.m_rows;
            m_columns = other.m_columns;
            m_data = (value_type *)malloc(m_rows * m_columns * sizeof(value_type));
            memcpy(m_data, other.m_data, m_rows * m_columns * sizeof(value_type));
        }

        Tensor Tensor::create_from_vector_vector(const std::vector<std::vector<value_type>> &vector)
        {
            Tensor matrix(vector.size(), vector[0].size());

            for (size_t i = 0; i < vector.size(); i++)
            {
                for (size_t j = 0; j < vector[i].size(); j++)
                {
                    matrix.set_element(i, j, vector[i][j]);
                }
            }

            return matrix;
        }

        Tensor Tensor::create_from_vector(const std::vector<value_type> &vector)
        {
            Tensor matrix(1, vector.size());

            for (size_t i = 0; i < vector.size(); i++)
            {
                matrix.set_element(0, i, vector[i]);
            }

            return matrix;
        }

        Tensor Tensor::create_identity_matrix(size_t size)
        {
            Tensor matrix(size, size);

            for (size_t i = 0; i < size; i++)
            {
                matrix.set_element(i, i, 1);
            }

            return matrix;
        }

        std::string Tensor::to_string() const
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

        FullyConnectedLayer::FullyConnectedLayer(size_t input_size, size_t output_size)
            : m_weights(input_size, output_size, 1.0f), m_biases(output_size, 1, 0.0f)
        {
        }

        FullyConnectedLayer::FullyConnectedLayer(const Tensor &weights, value_type biases)
            : m_weights(weights), m_biases(weights.get_columns(), 1, biases)
        {
        }

        FullyConnectedLayer::FullyConnectedLayer(const Tensor &weights, const Tensor &biases)
            : m_weights(weights), m_biases(biases)
        {
        }

        FullyConnectedLayer::~FullyConnectedLayer()
        {
        }

        Tensor FullyConnectedLayer::forward(const Tensor &input)
        {
            return m_weights * input + m_biases;
        }

        Tensor ReLU::forward(const Tensor &input)
        {
            Tensor result(input.get_rows(), input.get_columns());
            for (size_t i = 0; i < input.get_rows(); i++)
            {
                for (size_t j = 0; j < input.get_columns(); j++)
                {
                    value_type newValue = input.get_element(i, j) > 0 ? input.get_element(i, j) : 0;
                    result.set_element(i, j, newValue);
                }
            }
            return result;
        }

        Tensor Sigmoid::forward(const Tensor &input)
        {
            Tensor result(input.get_rows(), input.get_columns());
            for (size_t i = 0; i < input.get_rows(); i++)
            {
                for (size_t j = 0; j < input.get_columns(); j++)
                {
                    result.set_element(i, j, 1.0f / (1.0f + std::exp(-input.get_element(i, j))));
                }
            }
            return result;
        }

        ClipLayer::ClipLayer(value_type min, value_type max)
            : m_min(min), m_max(max)
        {
        }

        Tensor ClipLayer::forward(const Tensor &input)
        {
            Tensor result(input.get_rows(), input.get_columns());
            for (size_t i = 0; i < input.get_rows(); i++)
            {
                for (size_t j = 0; j < input.get_columns(); j++)
                {
                    result.set_element(i, j, input.get_element(i, j) < m_min ? m_min : input.get_element(i, j) > m_max ? m_max
                                                                                                                       : input.get_element(i, j));
                }
            }
            return result;
        }

        Tensor Softmax::forward(const Tensor &input)
        {
            Tensor result(input.get_rows(), input.get_columns());
            value_type sum = 0.0f;
            for (size_t i = 0; i < input.get_rows(); i++)
            {
                for (size_t j = 0; j < input.get_columns(); j++)
                {
                    sum += std::exp(input.get_element(i, j));
                }
            }

            for (size_t i = 0; i < input.get_rows(); i++)
            {
                for (size_t j = 0; j < input.get_columns(); j++)
                {
                    result.set_element(i, j, std::exp(input.get_element(i, j)) / sum);
                }
            }

            return result;
        }
    }

#ifdef NTT_MICRO_NN_IMPLEMENTATION
} // namespace
#endif