#pragma once

#include <queue>
#include <deque>
#include <mutex>

template<typename T>
class FrameQueue {
    public:
        void push(std::unique_ptr<T> frame) {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(frame));
            cv_.notify_one();
        }
        std::unique_ptr<T> pop() {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !queue_.empty() || stopped_; });
            if (queue_.empty() && stopped_) return nullptr;
            std::unique_ptr<T> frame = std::move(queue_.front());
            queue_.pop();
            return frame;
        }
        void stop() {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
            cv_.notify_all();
        }
        size_t size() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.size();
        }
    private:
        std::queue<std::unique_ptr<T>> queue_;
        mutable std::mutex mutex_;
        std::condition_variable cv_;
        bool stopped_ = false;
};

template<typename T, typename = void>
struct has_clear_method : std::false_type {};

template<typename T>
struct has_clear_method<T, std::void_t<decltype(std::declval<T>().clear())>> : std::true_type {};

template<typename T>
inline constexpr bool has_clear_v = has_clear_method<T>::value;

template <typename T>
class ObjectPool {
public:

    explicit ObjectPool(size_t initial_size = 0) {
        Initialize(initial_size);
    }

    ObjectPool(const ObjectPool&) = delete;
    ObjectPool& operator=(const ObjectPool&) = delete;
    ObjectPool(ObjectPool&&) = delete;
    ObjectPool& operator=(ObjectPool&&) = delete;

    void Initialize(size_t pool_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.clear(); // Clear existing pooled objects
        for (size_t i = 0; i < pool_size; ++i) {
            pool_.push_back(std::make_unique<T>());
        }
    }

    std::unique_ptr<T> Get() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pool_.empty()) {
            std::unique_ptr<T> ptr = std::move(pool_.front());
            pool_.pop_front();
            return ptr;
        }
        return std::make_unique<T>();
    }

    void Return(std::unique_ptr<T> ptr) {
        if (!ptr) {
            return; 
        }
        if constexpr (has_clear_v<T>) {
            // If it does, call it to reset the object's state.
            ptr->clear();
        }

        std::lock_guard<std::mutex> lock(mutex_);
        pool_.push_back(std::move(ptr));
    }

    size_t GetAvailableCount() const {
         std::lock_guard<std::mutex> lock(mutex_);
         return pool_.size();
    }

private:
    std::deque<std::unique_ptr<T>> pool_;
    mutable std::mutex mutex_;
};