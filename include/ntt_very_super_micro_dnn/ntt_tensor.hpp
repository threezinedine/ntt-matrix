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
#include <string>
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

        using vec = std::vector<float>;
        using tensor2d = std::vector<vec>;
        using tensor3d = std::vector<tensor2d>;
        using tensor4d = std::vector<tensor3d>;

        class Layer;

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
            inline const size_t getTotalElements() const { return m_totalElements; }
            float get_element(const shape_type &indexes) const;
            void set_element(const shape_type &indexes, float value);
            void reshape(const shape_type &newShape);
            Tensor reshape_clone(const shape_type &newShape);
            Tensor transpose(const size_t &axis1, const size_t &axis2) const;

            std::string to_string() const;
            std::string flatten() const;

            float max() const;
            Tensor max(const size_t &axis) const;
            shape_type argmax() const;
            Tensor argmax(const size_t &axis) const;

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
            static Tensor from_vector(const vec &data);
            static Tensor from_vector(const tensor2d &data);
            static Tensor from_vector(const tensor3d &data);
            static Tensor from_vector(const tensor4d &data);

        private:
            static size_t reloadTotalElements(const shape_type &shape);
            bool is_index_in_range(const shape_type &indexes) const;
            void reload_new_strides();

        private:
            shape_type m_shape;
            stride_type m_strides;
            size_t m_totalElements;
            float *m_data;
        };

        class Layer
        {
        public:
            virtual Tensor forward(const Tensor &input) = 0;
        };

        class ReLULayer : public Layer
        {
        public:
            Tensor forward(const Tensor &input) override;
        };

        class Clip2DLayer : public Layer
        {
        public:
            Clip2DLayer(const float &min, const float &max);
            Tensor forward(const Tensor &input) override;

        private:
            float m_min;
            float m_max;
        };

        class FullyConnectedLayer : public Layer
        {
        public:
            FullyConnectedLayer(const Tensor &weights, const Tensor &bias);
            Tensor forward(const Tensor &input) override;

        private:
            Tensor m_weights;
            Tensor m_bias;
        };

        class SoftmaxLayer : public Layer
        {
        public:
            Tensor forward(const Tensor &input) override;
        };

        class Conv2DLayer : public Layer
        {
        public:
            Conv2DLayer(const Tensor &weights, const Tensor &bias,
                        const size_t &stride, bool same_padding = true);
            Tensor forward(const Tensor &input) override;

        private:
            Tensor m_weights;
            Tensor m_bias;
            size_t m_stride;
            bool m_same_padding;
        };

#ifdef NTT_MICRO_NN_IMPLEMENTATION
        static float getMax(const float &a, const float &b)
        {
            return a > b ? a : b;
        }

        static float getMin(const float &a, const float &b)
        {
            return a < b ? a : b;
        }

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

        float Tensor::max() const
        {
            float max = m_data[0];
            for (size_t i = 0; i < getTotalElements(); i++)
            {
                if (m_data[i] > max)
                {
                    max = m_data[i];
                }
            }

            return max;
        }

        Tensor Tensor::max(const size_t &axis) const
        {
            shape_type newShape = m_shape;
            newShape[axis] = 1;
            Tensor result(newShape, 0.0f);
            Shape newShapeIndex(newShape);

            size_t maxAxisSize = m_shape[axis];

            while (!newShapeIndex.is_end())
            {
                shape_type targetIndexes = newShapeIndex.get_current_index();
                size_t maxValue = get_element(targetIndexes);

                for (size_t i = 0; i < maxAxisSize; i++)
                {
                    shape_type tempIndexes = targetIndexes;
                    tempIndexes[axis] = i;

                    if (get_element(tempIndexes) > maxValue)
                    {
                        maxValue = get_element(tempIndexes);
                    }
                }

                result.set_element(targetIndexes, maxValue);
                newShapeIndex.next();
            }

            return result;
        }

        shape_type Tensor::argmax() const
        {
            shape_type result(m_shape.size(), 0);
            Shape argmaxIndex(m_shape);

            float maxValue = get_element(argmaxIndex.get_current_index());
            argmaxIndex.next();

            while (!argmaxIndex.is_end())
            {
                if (get_element(argmaxIndex.get_current_index()) > maxValue)
                {
                    maxValue = get_element(argmaxIndex.get_current_index());
                    for (size_t i = 0; i < result.size(); i++)
                    {
                        result[i] = argmaxIndex.get_current_index()[i];
                    }
                }

                argmaxIndex.next();
            }

            return result;
        }

        Tensor Tensor::argmax(const size_t &axis) const
        {
            shape_type newShape = m_shape;
            newShape[axis] = 1;
            Tensor result(newShape, 0.0f);
            Shape newShapeIndex(newShape);

            size_t maxAxisSize = m_shape[axis];

            while (!newShapeIndex.is_end())
            {
                shape_type targetIndexes = newShapeIndex.get_current_index();
                size_t maxValue = get_element(targetIndexes);
                size_t maxValueIndex = 0;

                for (size_t i = 0; i < maxAxisSize; i++)
                {
                    shape_type tempIndexes = targetIndexes;
                    tempIndexes[axis] = i;

                    if (get_element(tempIndexes) > maxValue)
                    {
                        maxValue = get_element(tempIndexes);
                        maxValueIndex = i;
                    }
                }

                result.set_element(targetIndexes, maxValueIndex);
                newShapeIndex.next();
            }

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

        Tensor Tensor::from_vector(const vec &data)
        {
            Tensor tensor({data.size()}, 0.0f);
            for (size_t i = 0; i < data.size(); i++)
            {
                tensor.set_element({i}, data[i]);
            }

            return tensor;
        }

        Tensor Tensor::from_vector(const tensor2d &data)
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

        Tensor Tensor::from_vector(const tensor3d &data)
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

        Tensor Tensor::from_vector(const tensor4d &data)
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
            if (indexes.size() != m_shape.size())
            {
                char buffer[NTT_ERROR_MESSAGE_SIZE];
                snprintf(buffer, sizeof(buffer),
                         "Shape mismatch: %s != %s",
                         Shape::convert_shape_to_string(indexes).c_str(),
                         Shape::convert_shape_to_string(m_shape).c_str());
                throw std::invalid_argument(buffer);
            }

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

        Tensor Tensor::transpose(const size_t &axis1, const size_t &axis2) const
        {
            if (axis1 >= m_shape.size() || axis2 >= m_shape.size())
            {
                char buffer[NTT_ERROR_MESSAGE_SIZE];
                snprintf(buffer, sizeof(buffer),
                         "Invalid axis: %zu or %zu not in range of %zu",
                         axis1, axis2, m_shape.size());
                throw std::invalid_argument(buffer);
            }

            shape_type newShape = m_shape;
            newShape[axis1] = m_shape[axis2];
            newShape[axis2] = m_shape[axis1];

            Tensor result(newShape, 0.0f);
            Shape shape(m_shape);

            while (!shape.is_end())
            {
                shape_type targetIndexes = shape.get_current_index();
                size_t temp = targetIndexes[axis1];
                targetIndexes[axis1] = targetIndexes[axis2];
                targetIndexes[axis2] = temp;

                result.set_element(targetIndexes, get_element(shape.get_current_index()));
                shape.next();
            }

            return result;
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

        Tensor ReLULayer::forward(const Tensor &input)
        {
            Tensor result(input.get_shape(), 0.0f);
            Shape shape(input.get_shape());

            while (!shape.is_end())
            {
                float value = input.get_element(shape.get_current_index());
                result.set_element(shape.get_current_index(), std::max(0.0f, value));
                shape.next();
            }

            return result;
        }

        Clip2DLayer::Clip2DLayer(const float &min, const float &max)
            : m_min(min), m_max(max)
        {
        }

        Tensor Clip2DLayer::forward(const Tensor &input)
        {
            Tensor result(input.get_shape(), 0.0f);
            Shape shape(input.get_shape());

            while (!shape.is_end())
            {
                float value = input.get_element(shape.get_current_index());

                result.set_element(shape.get_current_index(), getMax(m_min, getMin(value, m_max)));
                shape.next();
            }

            return result;
        }

        FullyConnectedLayer::FullyConnectedLayer(const Tensor &weights, const Tensor &bias)
            : m_weights(weights), m_bias(bias)
        {
            if (m_weights.get_shape().size() != 2)
            {
                char buffer[NTT_ERROR_MESSAGE_SIZE];
                snprintf(buffer, sizeof(buffer),
                         "Weights must be a 2D tensor: %s",
                         Shape::convert_shape_to_string(m_weights.get_shape()).c_str());
                throw std::invalid_argument(buffer);
            }

            if (m_bias.get_shape().size() != 2)
            {
                char buffer[NTT_ERROR_MESSAGE_SIZE];
                snprintf(buffer, sizeof(buffer),
                         "Bias must be a 2D tensor: %s",
                         Shape::convert_shape_to_string(m_bias.get_shape()).c_str());
                throw std::invalid_argument(buffer);
            }

            if (m_weights.get_shape()[0] != m_bias.get_shape()[0])
            {
                char buffer[NTT_ERROR_MESSAGE_SIZE];
                snprintf(buffer, sizeof(buffer),
                         "Weights and bias dimensions mismatch: %s != %s",
                         Shape::convert_shape_to_string(m_weights.get_shape()).c_str(),
                         Shape::convert_shape_to_string(m_bias.get_shape()).c_str());
                throw std::invalid_argument(buffer);
            }
        }

        Tensor FullyConnectedLayer::forward(const Tensor &input)
        {
            // assert the matrix has valid size
            if (input.get_shape().size() != 2)
            {
                char buffer[NTT_ERROR_MESSAGE_SIZE];
                snprintf(buffer, sizeof(buffer),
                         "Input must be a 2D tensor: %s",
                         Shape::convert_shape_to_string(input.get_shape()).c_str());
                throw std::invalid_argument(buffer);
            }

            if (m_weights.get_shape()[1] != input.get_shape()[0])
            {
                char buffer[NTT_ERROR_MESSAGE_SIZE];
                snprintf(buffer, sizeof(buffer),
                         "Weights and input dimensions mismatch: %s != %s",
                         Shape::convert_shape_to_string(m_weights.get_shape()).c_str(),
                         Shape::convert_shape_to_string(input.get_shape()).c_str());
                throw std::invalid_argument(buffer);
            }

            Tensor result({m_weights.get_shape()[0], input.get_shape()[1]}, 0.0f);

            // dot product
            for (size_t i = 0; i < m_weights.get_shape()[0]; i++)
            {
                for (size_t j = 0; j < input.get_shape()[1]; j++)
                {
                    float value = 0.0f;
                    for (size_t k = 0; k < m_weights.get_shape()[1]; k++)
                    {
                        value += m_weights.get_element({i, k}) * input.get_element({k, j});
                    }
                    result.set_element({i, j}, value + m_bias.get_element({i, 0}));
                }
            }

            return result;
        }

        Tensor SoftmaxLayer::forward(const Tensor &input)
        {
            Tensor result(input.get_shape(), 0.0f);
            Shape shape(input.get_shape());

            float sum = 0.0f;

            while (!shape.is_end())
            {
                float value = input.get_element(shape.get_current_index());
                sum += std::exp(value);
                shape.next();
            }

            shape.reset();

            while (!shape.is_end())
            {
                float value = input.get_element(shape.get_current_index());
                result.set_element(shape.get_current_index(), std::exp(value) / sum);
                shape.next();
            }

            return result;
        }

        Conv2DLayer::Conv2DLayer(const Tensor &weights, const Tensor &bias,
                                 const size_t &stride, bool same_padding)
            : m_weights(weights), m_bias(bias), m_stride(stride), m_same_padding(same_padding)
        {
            if (m_weights.get_shape().size() != 4)
            {
                char buffer[NTT_ERROR_MESSAGE_SIZE];
                snprintf(buffer, sizeof(buffer),
                         "Weights must be a 4D tensor: %s",
                         Shape::convert_shape_to_string(m_weights.get_shape()).c_str());
                throw std::invalid_argument(buffer);
            }

            if (m_bias.get_shape().size() != 2)
            {
                char buffer[NTT_ERROR_MESSAGE_SIZE];
                snprintf(buffer, sizeof(buffer),
                         "Bias must be a 2D tensor: %s",
                         Shape::convert_shape_to_string(m_bias.get_shape()).c_str());
            }

            if (m_weights.get_shape()[0] != m_bias.get_shape()[0])
            {
                char buffer[NTT_ERROR_MESSAGE_SIZE];
                snprintf(buffer, sizeof(buffer),
                         "Weights and bias dimensions mismatch: %s != %s",
                         Shape::convert_shape_to_string(m_weights.get_shape()).c_str(),
                         Shape::convert_shape_to_string(m_bias.get_shape()).c_str());
                throw std::invalid_argument(buffer);
            }
        }

        Tensor Conv2DLayer::forward(const Tensor &input)
        {
            if (input.get_shape().size() != 4)
            {
                char buffer[NTT_ERROR_MESSAGE_SIZE];
                snprintf(buffer, sizeof(buffer),
                         "Input must be a 4D tensor: %s",
                         Shape::convert_shape_to_string(input.get_shape()).c_str());
                throw std::invalid_argument(buffer);
            }

            if (m_weights.get_shape()[1] != input.get_shape()[0])
            {
                char buffer[NTT_ERROR_MESSAGE_SIZE];
                snprintf(buffer, sizeof(buffer),
                         "Weights and input dimensions mismatch: %s != %s",
                         Shape::convert_shape_to_string(m_weights.get_shape()).c_str(),
                         Shape::convert_shape_to_string(input.get_shape()).c_str());
                throw std::invalid_argument(buffer);
            }

            shape_type outputShape = {m_bias.get_shape()[0],
                                      m_bias.get_shape()[1],
                                      0,  // height
                                      0}; // width

            if (m_same_padding)
            {
                outputShape[2] = input.get_shape()[2];
                outputShape[3] = input.get_shape()[3];
            }
            else
            {
                outputShape[2] = (input.get_shape()[2] - m_weights.get_shape()[2]) / m_stride + 1;
                outputShape[3] = (input.get_shape()[3] - m_weights.get_shape()[3]) / m_stride + 1;
            }

            Tensor result(outputShape, 0.0f);

            for (size_t i = 0; i < m_bias.get_shape()[0]; i++)
            {
                for (size_t j = 0; j < input.get_shape()[1]; j++)
                {
                    float bias_value = m_bias.get_element({i, j});

                    for (size_t k = 0; k < m_weights.get_shape()[1]; k++)
                    {
                        // TODO: implement the cross-correlation code
                        for (size_t l = 0; l < result.get_shape()[2]; l++)
                        {
                            for (size_t m = 0; m < result.get_shape()[3]; m++)
                            {
                                size_t input_x = l * m_stride;
                                size_t input_y = m * m_stride;

                                float value = 0.0f;

                                for (size_t n = 0; n < m_weights.get_shape()[2]; n++)
                                {
                                    for (size_t o = 0; o < m_weights.get_shape()[3]; o++)
                                    {
                                        value += m_weights.get_element({i, k, n, o}) *
                                                 input.get_element({k, j, input_x + n, input_y + o});
                                    }
                                }

                                result.set_element({i, j, l, m},
                                                   result.get_element({i, j, l, m}) + value +
                                                       bias_value / m_weights.get_shape()[1]);
                            }
                        }
                    }
                }
            }

            return result;
        }

#endif // NTT_MICRO_NN_IMPLEMENTATION
    }

#ifdef NTT_MICRO_NN_IMPLEMENTATION
} // namespace
#endif // NTT_MICRO_NN_IMPLEMENTATION
