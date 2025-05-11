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

        private:
            shape_type m_shape;
            shape_type m_currentIndex;
        };

        class Tensor
        {
        public:
            Tensor(const shape_type &shape, float defaultValue = NTT_DEFAULT_VALUE);

            inline shape_type get_shape() const { return m_shape; }
            float get_element(const shape_type &indexes) const;
            void set_element(const shape_type &indexes, float value);

            std::string to_string() const;
            std::string flatten() const;

        private:
            const size_t reloadTotalElements();
            inline const size_t getTotalElements() const { return m_totalElements; }

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

        const size_t Tensor::reloadTotalElements()
        {
            m_totalElements = 1;
            for (size_t i = 0; i < m_shape.size(); i++)
            {
                m_totalElements *= m_shape[i];
            }
            return m_totalElements;
        }

        Tensor::Tensor(const shape_type &shape, float defaultValue)
            : m_shape(shape)
        {
            reloadTotalElements();

            m_data = (float *)malloc(sizeof(float) * getTotalElements());

            for (size_t i = 0; i < getTotalElements(); i++)
            {
                m_data[i] = defaultValue;
            }

            size_t currentStride = 1;
            for (size_t i = 1; i < shape.size(); i++)
            {
                currentStride *= shape[i];
            }

            for (size_t i = 1; i < shape.size(); i++)
            {
                m_strides.push_back(currentStride);
                currentStride /= shape[i];
            }
            m_strides.push_back(1);
        }

        float Tensor::get_element(const shape_type &indexes) const
        {
            size_t index = 0;
            for (size_t i = 0; i < indexes.size(); i++)
            {
                index += indexes[i] * m_strides[i];
            }

            return m_data[index];
        }

        void Tensor::set_element(const shape_type &indexes, float value)
        {
            size_t index = 0;
            for (size_t i = 0; i < indexes.size(); i++)
            {
                index += indexes[i] * m_strides[i];
            }

            m_data[index] = value;
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
