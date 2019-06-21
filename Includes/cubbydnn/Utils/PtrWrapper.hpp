//
// Created by jwkim98 on 4/21/19.
//

#ifndef CUBBYDNN_PTRWRAPPER_HPP
#define CUBBYDNN_PTRWRAPPER_HPP

#include <cubbydnn/Tensors/Decl/TensorPlug.hpp>
#include <cubbydnn/Tensors/Decl/TensorSocket.hpp>

#include <atomic>
#include <memory>

namespace CubbyDNN
{
template <typename T, typename = void>
class Ptr
{
};
template <typename T>
class Ptr<TensorPlug<T>>
{
 public:
    Ptr() = default;

    Ptr(Ptr<TensorPlug<T>>&& tensorObjectPtr) noexcept
    {
    }

    Ptr<TensorPlug<T>>& operator=(Ptr<TensorPlug<T>>&& ptrWrapper) noexcept
    {
    }

    template <typename... Ts>
    Ptr<TensorPlug<T>> Make(Ts... args)
    {
        auto ptr = Ptr<TensorPlug<T>>();
        ptr.m_tensorObjectPtr = std::make_unique<TensorPlug<T>>(args...);
        return std::move(ptr);
    }

 private:
    std::unique_ptr<TensorPlug<T>> m_tensorObjectPtr = nullptr;
};

template <typename T>
class Ptr<TensorSocket<T>>
{
 public:
    Ptr() = default;

    Ptr(Ptr<TensorSocket<T>>&& ptrWrapper) noexcept
    {
        ptrWrapper.m_tensorSocketPtr = nullptr;
    }

    Ptr(const Ptr<TensorSocket<T>>& ptrWrapper)
        : m_tensorSocketPtr(ptrWrapper.m_tensorSocketPtr),
          m_reference_count(ptrWrapper.m_reference_count + 1)
    {
    }

    template <typename... Ts>
    static Ptr<TensorSocket<T>> Make(Ts... args)
    {
        auto ptrWrapper = Ptr<TensorSocket<T>>();
        ptrWrapper.m_tensorSocketPtr = new TensorSocket<T>(args...);
        ptrWrapper.m_reference_count = 0;
        return std::move(ptrWrapper);
    }

    Ptr<TensorSocket<T>>& operator=(Ptr<TensorSocket<T>>&& ptrWrapper) noexcept
    {
        m_tensorSocketPtr = ptrWrapper.m_tensorSocketPtr;
        m_reference_count = ptrWrapper.m_reference_count;
        ptrWrapper.m_tensorSocketPtr = nullptr;
    }

    Ptr<TensorSocket<T>>& operator=(const Ptr<TensorSocket<T>>& ptrWrapper)
    {
        m_tensorSocketPtr = ptrWrapper.m_tensorSocketPtr;
        m_reference_count = ptrWrapper.m_reference_count + 1;
    }

    TensorSocket<T>& operator->()
    {
        return *m_tensorSocketPtr;
    }

 private:
    TensorSocket<T>* m_tensorSocketPtr = nullptr;
    std::atomic_int m_reference_count = 0;
};

enum class SharedPtrState
{
    valid,
    dirty,
    invalid,
};

template <typename T>
class SharedPtr
{
 private:
    /**
     * Shared object stores the actual object with atomic reference counter
     */
    struct SharedObject
    {
        SharedObject(T&& object, const int maxRefCount)
            : Object(std::move(object)),
              RefCount(1),
              MaxRefCount(maxRefCount){};

        T Object;
        std::atomic<int> RefCount;
        /// Maximum reference count that it can reach
        const int MaxRefCount;
    };

    SharedObject* m_sharedObjectPtr;

    SharedPtrState m_ptrState;

    /**
     * private constructor for constructing the object for the first time
     * @param objectPtr : objectPtr that has been created
     * @param state : state of the sharedObject
     */
    explicit SharedPtr(SharedObject* objectPtr, SharedPtrState state);

    /**
     * Makes copy of the sharedPtr
     * @return
     */
    SharedPtr<T> tryMakeCopy();

 public:
    /**
     * Builds new SharedPtr object with no parameters
     * @return : SharedPtr
     */
    static SharedPtr<T> Make();

    /**
     * Builds new SharedPtr object with parameters
     * @tparam Ts : template parameter pack
     * @param maxReferenceCount : maximum reference count of this object
     * @param args : arguments to build new object
     * @return : SharedPtr
     */
    template <typename... Ts>
    static SharedPtr<T> Make(int maxReferenceCount, Ts&... args);

    /**
     * Copy constructor is explicitly deleted
     * @param sharedPtr
     */
    SharedPtr(const SharedPtr<T>& sharedPtr) = delete;

    /**
     * Copy assign operator is explicitly deleted
     * @param sharedPtr
     * @return
     */
    SharedPtr<T>& operator=(const SharedPtr<T>& sharedPtr) = delete;

    /**
     * Move constructor
     * This will make given parameter (sharedPtr) invalid
     * @param sharedPtr : SharedPtr<T> to move from
     */
    SharedPtr(SharedPtr<T>&& sharedPtr) noexcept;

    /**
     * Move assign operator
     * This will make given parameter (sharedPtr) invalid
     * @param sharedPtr : SharedPtr<T> to move from
     * @return : SharedPtr<T>
     */
    SharedPtr<T>& operator=(SharedPtr<T>&& sharedPtr) noexcept;

    /**
     * Makes copy of this SharedPtr
     * Increments reference count of the object
     * @return
     */
    SharedPtr<T> MakeCopy();

    /**
     * Returns state of this SharedPtr
     * This is used to determine if SharePtr is in valid state
     * @return : state of this SharedPtr
     */
    SharedPtrState GetState()
    {
        return m_ptrState;
    }
};

}  // namespace CubbyDNN

#endif  // CUBBYDNN_PTRWRAPPER_HPP