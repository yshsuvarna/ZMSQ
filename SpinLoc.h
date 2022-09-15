// Custom spin lock implementation
// CS 309

#include <thread>
#include <mutex>
#include <atomic>

struct SpinLock
{
    std::atomic<bool> lock_ = {0};

    void lock() noexcept
    {
        for (;;)
        {
            // Optimistically assume the lock is free on the first try
            if (!lock_.exchange(true, std::memory_order_acquire))
            {
                return;
            }
            // Wait for lock to be released without generating cache misses
            while (lock_.load(std::memory_order_relaxed))
            {
                // To reduce the contention between hyper threads
                __builtin_ia32_pause();
            }
        }
    }

    bool try_lock() noexcept
    {
        // Function to check if the lock is free inorder to prevent cache misses
        return !lock_.load(std::memory_order_relaxed) &&
               !lock_.exchange(true, std::memory_order_acquire);
    }

    // unlock the lock
    void unlock() noexcept
    {
        lock_.store(false, std::memory_order_release);
    }
};