#pragma once
#include <cstdint>
#include <stdexcept>
#include <vector>
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
#endif // NTT_MICRO_NN_IMPLEMENTATION

#define NTT_ERROR_MESSAGE_SIZE 1994
#define NTT_DEFAULT_VALUE 0.0f

#ifdef NTT_MICRO_NN_IMPLEMENTATION
namespace
{
#endif // NTT_MICRO_NN_IMPLEMENTATION
    namespace ntt
    {
        using shape_type = std::vector<size_t>;
        using stride_type = std::vector<size_t>;

        class Shape
        {
        public:
            Shape(const shape_type &shape);
            bool is_first_element() const;
            bool is_end() const;
            void next();
            void reset();
            inline shape_type &get_current_index() { return m_currentIndex; }
            size_t get_number_of_new_lines() const;

            std::string to_string() const;

        public:
            static std::string convert_shape_to_string(const shape_type &shape);
            static bool is_shape_equal(const shape_type &shape1, const shape_type &shape2);

        private:
            shape_type m_shape;
            shape_type m_currentIndex;
        };

        class Tensor
        {
        public:
            Tensor(const shape_type &shape, float defaultValue = NTT_DEFAULT_VALUE);
            Tensor(const Tensor &other);
            ~Tensor();

            inline shape_type get_shape() const { return m_shape; }
            float get_element(const shape_type &indexes) const;
            void set_element(const shape_type &indexes, float value);
            void reshape(const shape_type &newShape);
            Tensor reshape_clone(const shape_type &newShape);

            std::string to_string() const;
            std::string flatten() const;

        public:
            Tensor add(const Tensor &other) const;
            Tensor negative() const;
            Tensor multiply(const float &other) const;
            Tensor divide(const float &other) const;
            Tensor subtract(const Tensor &other) const;

        public:
            bool operator==(const Tensor &other) const;
            void operator=(const Tensor &other);
            Tensor operator+(const Tensor &other) const;
            Tensor operator+(const float &other) const;
            Tensor operator-(const Tensor &other) const;
            Tensor operator-(const float &other) const;
            Tensor operator*(const float &other) const;
            Tensor operator/(const float &other) const;

        public:
            static Tensor from_vector(const std::vector<float> &data);
            static Tensor from_vector(const std::vector<std::vector<float>> &data);
            static Tensor from_vector(const std::vector<std::vector<std::vector<float>>> &data);
            static Tensor from_vector(const std::vector<std::vector<std::vector<std::vector<float>>>> &data);

        private:
            static size_t reloadTotalElements(const shape_type &shape);
            inline const size_t getTotalElements() const { return m_totalElements; }
            bool is_index_in_range(const shape_type &indexes) const;
            void reload_new_strides();

        private:
            shape_type m_shape;
            stride_type m_strides;
            size_t m_totalElements;
            float *m_data;
        };

#ifdef NTT_MICRO_NN_IMPLEMENTATION
        Shape::Shape(const shape_type &shape) : m_shape(shape)
        {
            m_currentIndex.clear();
            for (size_t i = 0; i < shape.size(); i++)
            {
                m_currentIndex.push_back(0);
            }
        }

        bool Shape::is_first_element() const
        {
            for (size_t i = 0; i < m_currentIndex.size(); i++)
            {
                if (m_currentIndex[i] != 0)
                {
                    return false;
                }
            }

            return true;
        }

        bool Shape::is_end() const
        {
            for (size_t i = 0; i < m_currentIndex.size(); i++)
            {
                if (m_currentIndex[i] >= m_shape[i])
                {
                    return true;
                }
            }

            return false;
        }

        void Shape::next()
        {
            if (is_end())
            {
                return;
            }

            for (size_t i = m_currentIndex.size() - 1; i >= 0; i--)
            {
                size_t matchedShape = m_shape[i];

                if (i == 0)
                {
                    m_currentIndex[i]++;
                    break;
                }

                if (m_currentIndex[i] < matchedShape - 1)
                {
                    m_currentIndex[i]++;
                    break;
                }
                else
                {
                    m_currentIndex[i] = 0;
                }
            }
        }

        void Shape::reset()
        {
            for (size_t i = 0; i < m_currentIndex.size(); i++)
            {
                m_currentIndex[i] = 0;
            }
        }

        size_t Shape::get_number_of_new_lines() const
        {
            size_t result = 0;
            for (size_t i = m_shape.size() - 1; i >= 0; i--)
            {
                if (m_currentIndex[i] == 0)
                {
                    result++;
                }
                else
                {
                    break;
                }
            }

            return result;
        }

        std::string Shape::to_string() const
        {
            char buffer[NTT_ERROR_MESSAGE_SIZE];

            std::string shapeString = "";

            shapeString += "[";
            for (size_t i = 0; i < m_shape.size(); i++)
            {
                shapeString += std::to_string(m_shape[i]);

                if (i != m_shape.size() - 1)
                {
                    shapeString += ", ";
                }
            }
            shapeString += "]";

            std::string indexString = "";
            indexString += "[";
            for (size_t i = 0; i < m_currentIndex.size(); i++)
            {
                indexString += std::to_string(m_currentIndex[i]);

                if (i != m_currentIndex.size() - 1)
                {
                    indexString += ", ";
                }
            }
            indexString += "]";

            snprintf(
                buffer, sizeof(buffer),
                "Shape: %s - Index: %s",
                shapeString.c_str(), indexString.c_str());

            return std::string(buffer);
        }

        std::string Shape::convert_shape_to_string(const shape_type &shape)
        {
            std::string result = "[";
            for (size_t i = 0; i < shape.size(); i++)
            {
                result += std::to_string(shape[i]);

                if (i != shape.size() - 1)
                {
                    result += ", ";
                }
            }
            result += "]";
            return result;
        }

        bool Shape::is_shape_equal(const shape_type &shape1, const shape_type &shape2)
        {
            if (shape1.size() != shape2.size())
            {
                return false;
            }

            for (size_t i = 0; i < shape1.size(); i++)
            {
                if (shape1[i] != shape2[i])
                {
                    return false;
                }
            }

            return true;
        }

        std::string Tensor::flatten() const
        {
            std::string result = "[";
            for (size_t i = 0; i < getTotalElements(); i++)
            {
                result += std::to_string(m_data[i]) + ", ";
            }
            result += "]";

            return result;
        }

        Tensor Tensor::add(const Tensor &other) const
        {
            if (!Shape::is_shape_equal(m_shape, other.m_shape))
            {
                char buffer[NTT_ERROR_MESSAGE_SIZE];
                snprintf(buffer, sizeof(buffer),
                         "Shape mismatch: %s != %s",
                         Shape::convert_shape_to_string(m_shape).c_str(),
                         Shape::convert_shape_to_string(other.m_shape).c_str());
                throw std::invalid_argument(buffer);
            }

            Tensor result(m_shape, 0.0f);

            for (size_t i = 0; i < getTotalElements(); i++)
            {
                result.m_data[i] = m_data[i] + other.m_data[i];
            }

            return result;
        }

        Tensor Tensor::multiply(const float &other) const
        {
            Tensor result(m_shape, 0.0f);

            for (size_t i = 0; i < getTotalElements(); i++)
            {
                result.m_data[i] = m_data[i] * other;
            }

            return result;
        }

        Tensor Tensor::divide(const float &other) const
        {
            Tensor result(m_shape, 0.0f);

            for (size_t i = 0; i < getTotalElements(); i++)
            {
                result.m_data[i] = m_data[i] / other;
            }

            return result;
        }

        Tensor Tensor::negative() const
        {
            Tensor result(m_shape, 0.0f);

            for (size_t i = 0; i < getTotalElements(); i++)
            {
                result.m_data[i] = -m_data[i];
            }

            return result;
        }

        Tensor Tensor::subtract(const Tensor &other) const
        {
            return add(other.negative());
        }

        Tensor Tensor::operator+(const Tensor &other) const
        {
            return add(other);
        }

        Tensor Tensor::operator+(const float &other) const
        {
            Tensor result(m_shape, other);
            return add(result);
        }

        Tensor Tensor::operator-(const Tensor &other) const
        {
            return subtract(other);
        }

        Tensor Tensor::operator-(const float &other) const
        {
            Tensor result(m_shape, other);
            return subtract(result);
        }

        void Tensor::operator=(const Tensor &other)
        {
            if (m_data != nullptr)
            {
                free(m_data);
                m_data = nullptr;
            }

            m_shape = other.m_shape;
            m_strides = other.m_strides;
            m_totalElements = other.m_totalElements;
            m_data = (float *)malloc(sizeof(float) * m_totalElements);

            for (size_t i = 0; i < m_totalElements; i++)
            {
                m_data[i] = other.m_data[i];
            }
        }

        bool Tensor::operator==(const Tensor &other) const
        {
            if (m_shape != other.m_shape)
            {
                return false;
            }

            for (size_t i = 0; i < m_totalElements; i++)
            {
                if (std::fabs(m_data[i] - other.m_data[i]) > std::numeric_limits<float>::epsilon())
                {
                    return false;
                }
            }

            return true;
        }

        Tensor Tensor::operator*(const float &other) const
        {
            return multiply(other);
        }

        Tensor Tensor::operator/(const float &other) const
        {
            return divide(other);
        }

        Tensor Tensor::from_vector(const std::vector<float> &data)
        {
            Tensor tensor({data.size()}, 0.0f);
            for (size_t i = 0; i < data.size(); i++)
            {
                tensor.set_element({i}, data[i]);
            }

            return tensor;
        }

        Tensor Tensor::from_vector(const std::vector<std::vector<float>> &data)
        {
            Tensor tensor({data.size(), data[0].size()}, 0.0f);
            for (size_t i = 0; i < data.size(); i++)
            {
                for (size_t j = 0; j < data[i].size(); j++)
                {
                    tensor.set_element({i, j}, data[i][j]);
                }
            }

            return tensor;
        }

        Tensor Tensor::from_vector(const std::vector<std::vector<std::vector<float>>> &data)
        {
            Tensor tensor({data.size(), data[0].size(), data[0][0].size()}, 0.0f);
            for (size_t i = 0; i < data.size(); i++)
            {
                for (size_t j = 0; j < data[i].size(); j++)
                {
                    for (size_t k = 0; k < data[i][j].size(); k++)
                    {
                        tensor.set_element({i, j, k}, data[i][j][k]);
                    }
                }
            }
            return tensor;
        }

        Tensor Tensor::from_vector(const std::vector<std::vector<std::vector<std::vector<float>>>> &data)
        {
            Tensor tensor({data.size(), data[0].size(), data[0][0].size(), data[0][0][0].size()}, 0.0f);
            for (size_t i = 0; i < data.size(); i++)
            {
                for (size_t j = 0; j < data[i].size(); j++)
                {
                    for (size_t k = 0; k < data[i][j].size(); k++)
                    {
                        for (size_t l = 0; l < data[i][j][k].size(); l++)
                        {
                            tensor.set_element({i, j, k, l}, data[i][j][k][l]);
                        }
                    }
                }
            }
            return tensor;
        }

        size_t Tensor::reloadTotalElements(const shape_type &shape)
        {
            size_t result = 1;
            for (size_t i = 0; i < shape.size(); i++)
            {
                result *= shape[i];
            }
            return result;
        }

        bool Tensor::is_index_in_range(const shape_type &indexes) const
        {
            for (size_t i = 0; i < indexes.size(); i++)
            {
                if (indexes[i] >= m_shape[i])
                {
                    return false;
                }
            }
            return true;
        }

        Tensor::Tensor(const shape_type &shape, float defaultValue)
            : m_shape(shape)
        {
            m_totalElements = reloadTotalElements(m_shape);

            m_data = (float *)malloc(sizeof(float) * getTotalElements());

            for (size_t i = 0; i < getTotalElements(); i++)
            {
                m_data[i] = defaultValue;
            }

            reload_new_strides();
        }

        Tensor::Tensor(const Tensor &other)
        {
            m_shape = other.m_shape;
            m_strides = other.m_strides;
            m_totalElements = other.m_totalElements;
            m_data = (float *)malloc(sizeof(float) * m_totalElements);

            for (size_t i = 0; i < m_totalElements; i++)
            {
                m_data[i] = other.m_data[i];
            }
        }

        void Tensor::reload_new_strides()
        {
            size_t currentStride = 1;
            for (size_t i = 1; i < m_shape.size(); i++)
            {
                currentStride *= m_shape[i];
            }

            m_strides.clear();
            for (size_t i = 1; i < m_shape.size(); i++)
            {
                m_strides.push_back(currentStride);
                currentStride /= m_shape[i];
            }

            m_strides.push_back(1);
        }

        Tensor::~Tensor()
        {
            if (m_data != nullptr)
            {
                free(m_data);
                m_data = nullptr;
            }
        }

        float Tensor::get_element(const shape_type &indexes) const
        {
            if (!is_index_in_range(indexes))
            {
                char buffer[NTT_ERROR_MESSAGE_SIZE];
                snprintf(buffer, sizeof(buffer),
                         "Index is out of range: %s not in range of %s",
                         Shape::convert_shape_to_string(indexes).c_str(),
                         Shape::convert_shape_to_string(m_shape).c_str());
                throw std::out_of_range(buffer);
            }

            size_t index = 0;
            for (size_t i = 0; i < indexes.size(); i++)
            {
                index += indexes[i] * m_strides[i];
            }

            return m_data[index];
        }

        void Tensor::set_element(const shape_type &indexes, float value)
        {
            if (!is_index_in_range(indexes))
            {
                char buffer[NTT_ERROR_MESSAGE_SIZE];
                snprintf(buffer, sizeof(buffer),
                         "Index is out of range: %s not in range of %s",
                         Shape::convert_shape_to_string(indexes).c_str(),
                         Shape::convert_shape_to_string(m_shape).c_str());
                throw std::out_of_range(buffer);
            }

            size_t index = 0;
            for (size_t i = 0; i < indexes.size(); i++)
            {
                index += indexes[i] * m_strides[i];
            }

            m_data[index] = value;
        }

        void Tensor::reshape(const shape_type &newShape)
        {
            if (reloadTotalElements(newShape) != getTotalElements())
            {
                char buffer[NTT_ERROR_MESSAGE_SIZE];
                snprintf(buffer, sizeof(buffer),
                         "Total elements mismatch: %zu (%s) != %zu (%s)",
                         reloadTotalElements(newShape),
                         Shape::convert_shape_to_string(newShape).c_str(),
                         getTotalElements(),
                         Shape::convert_shape_to_string(m_shape).c_str());
                throw std::invalid_argument(buffer);
            }

            m_shape = newShape;
            reload_new_strides();
        }

        Tensor Tensor::reshape_clone(const shape_type &newShape)
        {
            Tensor newTensor(*this);
            newTensor.reshape(newShape);
            return newTensor;
        }

        std::string Tensor::to_string() const
        {
            std::string result = "[\n";

            size_t currentTabNumber = 0;

            for (size_t i = 0; i < m_shape.size(); i++)
            {
                currentTabNumber++;
                for (size_t j = 0; j < currentTabNumber; j++)
                {
                    result += "\t";
                }
                result += "[\n";
            }

            currentTabNumber++;

            Shape shape(m_shape);

            while (!shape.is_end())
            {
                if (shape.is_first_element())
                {
                    for (size_t i = 0; i < currentTabNumber; i++)
                    {
                        result += "\t";
                    }

                    result += std::to_string(get_element(shape.get_current_index())) + ", \n";

                    shape.next();
                    continue;
                }

                size_t numberOfNewLines = shape.get_number_of_new_lines();
                for (int i = numberOfNewLines - 1; i >= 0; i--)
                {
                    currentTabNumber--;
                    for (size_t j = 0; j < currentTabNumber; j++)
                    {
                        result += "\t";
                    }
                    result += "],\n";
                }

                for (int i = 0; i < numberOfNewLines; i++)
                {
                    for (size_t j = 0; j < currentTabNumber; j++)
                    {
                        result += "\t";
                    }
                    result += "[\n";
                    currentTabNumber++;
                }

                for (size_t i = 0; i < currentTabNumber; i++)
                {
                    result += "\t";
                }

                result += std::to_string(get_element(shape.get_current_index())) + ", \n";

                shape.next();
            }

            while (currentTabNumber > 0)
            {
                currentTabNumber--;
                for (size_t j = 0; j < currentTabNumber; j++)
                {
                    result += "\t";
                }
                result += "]\n";
            }

            return result;
        }
#endif // NTT_MICRO_NN_IMPLEMENTATION
    }

#ifdef NTT_MICRO_NN_IMPLEMENTATION
} // namespace
#endif // NTT_MICRO_NN_IMPLEMENTATION
