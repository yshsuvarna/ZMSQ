// Relaxed Priority Queue
// CS 309

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include "SpinLoc.h"

using namespace std;

template <
    typename T,
    size_t PopBatch = 16,
    size_t ListTargetSize = 25,
    typename Mutex = SpinLock,
    template <typename> class Atom = std::atomic>
class RelaxedPriorityQueue
{
    // Max height of the tree
    static constexpr uint32_t MAX_LEVELS = 32;

    // The default minimum value
    static constexpr T MIN_VALUE = std::numeric_limits<T>::min();

    // Align size for the shared buffer node
    static constexpr size_t Align = 1u << 7;
    static constexpr int LevelForForceInsert = 3;
    static constexpr int LevelForTraverseParent = 7;

    static_assert(PopBatch <= 256, "PopBatch must be <= 256");
    static_assert(
        ListTargetSize >= 1 && ListTargetSize <= 256,
        "TargetSize must be in the range [1, 256]");

    // The maximal length for the list
    static constexpr size_t PruningSize = ListTargetSize * 2;

    // Used to check if the child should be merged with the parent
    static constexpr size_t MergingSize = ListTargetSize;

    // List Node structure
    struct Node
    {
        Node *next;
        T val;
    };

    // Mound Element (Tree node), head points to a linked list
    struct MoundElement
    {
        // Atomic data type to read write data without locks
        Atom<Node *> head;
        Atom<size_t> size;
        Mutex lock;
        MoundElement()
        {
            head.store(nullptr, std::memory_order_relaxed);
            size.store(0, std::memory_order_relaxed);
        }
    };

    // The level and index of any node in the mound
    struct Position
    {
        uint32_t level;
        uint32_t index;
    };

    // Shared Buffer Node element
    struct BufferNode
    {
        Atom<Node *> pnode;
    };

    // The mound data structre to represent the tree in the form of a 2D array
    MoundElement *levels_[MAX_LEVELS];

    // The last level in the tree
    Atom<uint32_t> bottom_;

    // Required while expanding the tree
    Atom<uint32_t> guard_;

    // The consumer threads will access the shared buffer
    std::unique_ptr<BufferNode[]> shared_buffer_;
    Atom<int> top_loc_;

    // The count of producer and consumer threads
    Atom<size_t> producerCount;
    Atom<size_t> consumerCount;

public:
    // Constructor
    RelaxedPriorityQueue() : producerCount(0), consumerCount(0)
    {
        // If the shared buffer is not null reset it to null
        if (PopBatch > 0)
        {
            top_loc_ = -1;
            shared_buffer_.reset(new BufferNode[PopBatch]);
            for (size_t i = 0; i < PopBatch; i++)
            {
                shared_buffer_[i].pnode = nullptr;
            }
        }

        // The bottom and guard are initialized to 0
        bottom_.store(0, std::memory_order_relaxed);
        guard_.store(0, std::memory_order_relaxed);

        // Create the 2D array and initialize its levels to null
        levels_[0] = new MoundElement[1];
        for (uint32_t i = 1; i < MAX_LEVELS; i++)
        {
            levels_[i] = nullptr;
        }
    }
    // Destructor
    ~RelaxedPriorityQueue()
    {
        // Delete the shared buffer
        if (PopBatch > 0)
        {
            deleteSharedBuffer();
        }
        Position pos;
        pos.level = pos.index = 0;
        deleteAllNodes(pos);

        // Delete the mound tree
        for (int i = getBottomLevel(); i >= 0; i--)
        {
            delete[] levels_[i];
        }
    }

    // The Push function for the Relaxed priority queue
    void push(const T &val)
    {
        moundPush(val);
        producerCount.fetch_add(1, std::memory_order_relaxed);
    }

    // The pop function for the Relaxed priority queue
    void pop(T &val)
    {
        moundPop(val);
        consumerCount.fetch_add(1, std::memory_order_relaxed);
    }

    // An estimate size of the queue
    size_t size()
    {
        size_t p = producerCount.load(std::memory_order_acquire);
        size_t c = consumerCount.load(std::memory_order_acquire);
        return (p > c) ? p - c : 0;
    }

    // Returns true only if the queue was empty during the call.
    bool empty()
    {
        return isEmpty();
    }

private:
    // Function to get the bottom level of the mound tree
    uint32_t getBottomLevel()
    {
        return bottom_.load(std::memory_order_acquire);
    }

    // Function to delete the shared buffer
    void deleteSharedBuffer()
    {
        if (PopBatch > 0)
        {
            // delete nodes in the buffer
            int loc = top_loc_.load(std::memory_order_relaxed);
            while (loc >= 0)
            {
                Node *n = shared_buffer_[loc--].pnode.load(std::memory_order_relaxed);
                delete n;
            }
            // delete buffer
            shared_buffer_.reset();
        }
    }

    // Function to delete the nodes of the mound tree
    void deleteAllNodes(const Position &pos)
    {
        // The current list is empty
        if (getElementSize(pos) == 0)
        {
            return;
        }

        // Recusively go through the child nodes and erase them from bottom up
        Node *curList = getList(pos);
        setTreeNode(pos, nullptr);
        while (curList != nullptr)
        {
            Node *n = curList;
            curList = curList->next;
            delete n;
        }

        if (!isLeaf(pos))
        {
            deleteAllNodes(leftOf(pos));
            deleteAllNodes(rightOf(pos));
        }
    }

    // Check the first node in TreeElement keeps the heap structure.
    bool isHeap(const Position &pos)
    {
        if (isLeaf(pos))
        {
            return true;
        }
        Position lchild = leftOf(pos);
        Position rchild = rightOf(pos);
        return isHeap(lchild) && isHeap(rchild) &&
               readValue(pos) >= readValue(lchild) &&
               readValue(pos) >= readValue(rchild);
    }

    // Check if a node is a leaf node
    bool isLeaf(const Position &pos)
    {
        return pos.level == getBottomLevel();
    }

    // Check if the current node is the root node
    bool isRoot(const Position &pos)
    {
        return pos.level == 0;
    }

    // Locate the parent node
    Position parentOf(const Position &pos)
    {
        Position res;
        res.level = pos.level - 1;
        res.index = pos.index / 2;
        return res;
    }

    // Locate the left child
    Position leftOf(const Position &pos)
    {
        Position res;
        res.level = pos.level + 1;
        res.index = pos.index * 2;
        return res;
    }

    // Locate the right child
    Position rightOf(const Position &pos)
    {
        Position res;
        res.level = pos.level + 1;
        res.index = pos.index * 2 + 1;
        return res;
    }

    // Returns the size of the list of the current mound element list
    size_t getElementSize(const Position &pos)
    {
        return levels_[pos.level][pos.index].size.load(std::memory_order_relaxed);
    }

    // Updates the size of current MoundElement
    void setElementSize(
        const Position &pos, const uint32_t &v)
    {
        levels_[pos.level][pos.index].size.store(v, std::memory_order_relaxed);
    }

    // Increase the tree height
    void grow(uint32_t btm)
    {
        while (true)
        {

            if (guard_.fetch_add(1, std::memory_order_acq_rel) == 0)
            {
                break;
            }

            // Check if concurrently the tree was increased by some other producer
            if (btm != getBottomLevel())
            {
                return;
            }
            std::this_thread::yield();
        }
        // Check the bottom has not changed yet
        if (btm != getBottomLevel())
        {
            guard_.store(0, std::memory_order_release);
            return;
        }

        // Create and initialize the new level
        uint32_t tmp_btm = getBottomLevel();
        uint32_t size = 1 << (tmp_btm + 1);

        MoundElement *new_level = new MoundElement[size];
        levels_[tmp_btm + 1] = new_level;

        // Update the bottom and guard after adding the new level
        bottom_.store(tmp_btm + 1, std::memory_order_release);
        guard_.store(0, std::memory_order_release);
    }

    // It selects a position to insert the node,
    // 1. It returns a position with head node has lower priority than the target.
    // Thus it could be potentially used as the starting element to do the binary search to find the fit position.  (slow path)
    // 2. It returns a position, which is not the best fit. But it also prevents aggressively grow the Mound. (fast path)
    Position selectPosition(const T &val, bool &path, uint32_t &seed)
    {
        while (true)
        {
            uint32_t b = getBottomLevel();
            int bound = 1 << b;
            int steps = 1 + b * b;
            ++seed;
            uint32_t index = seed % bound;

            for (int i = 0; i < steps; i++)
            {
                int loc = (index + i) % bound;
                Position pos;
                pos.level = b;
                pos.index = loc;

                // A quick check
                if (optimisticReadValue(pos) <= val)
                {
                    path = false;
                    seed = ++loc;
                    return pos;
                }
                else if (
                    b > LevelForForceInsert && getElementSize(pos) < ListTargetSize)
                {
                    // It makes sure every tree element should have more than the given number of nodes.
                    seed = ++loc;
                    path = true;
                    return pos;
                }
                if (b != getBottomLevel())
                {
                    break;
                }
            }
            // Failed too many times grow
            if (b == getBottomLevel())
            {
                grow(b);
            }
        }
    }

    // Swap two Tree Elements (head, size)
    void swapList(const Position &a, const Position &b)
    {
        Node *tmp = getList(a);
        setTreeNode(a, getList(b));
        setTreeNode(b, tmp);

        // need to swap the tree node meta-data
        uint32_t sa = getElementSize(a);
        uint32_t sb = getElementSize(b);
        setElementSize(a, sb);
        setElementSize(b, sa);
    }

    // Utility Function to lock a particular node
    void lockNode(const Position &pos)
    {
        levels_[pos.level][pos.index].lock.lock();
    }

    // Utility Function to unlock a locked node
    void unlockNode(const Position &pos)
    {
        levels_[pos.level][pos.index].lock.unlock();
    }

    // Utility Function to try a lock
    bool trylockNode(const Position &pos)
    {
        return levels_[pos.level][pos.index].lock.try_lock();
    }

    T optimisticReadValue(const Position &pos)
    {
        Node *tmp = levels_[pos.level][pos.index].head;
        return (tmp == nullptr) ? MIN_VALUE : tmp->val;
    }

    // Get the value from the head of the list as the elementvalue
    T readValue(const Position &pos)
    {
        Node *tmp = getList(pos);
        return (tmp == nullptr) ? MIN_VALUE : tmp->val;
    }

    // Utility function to return the list of elements in a node
    Node *getList(const Position &pos)
    {
        return levels_[pos.level][pos.index].head.load(std::memory_order_acquire);
    }

    // Utility function to set a particular node
    void setTreeNode(const Position &pos, Node *t)
    {
        levels_[pos.level][pos.index].head.store(t, std::memory_order_release);
    }

    // Merge two sorted lists
    Node *mergeList(Node *base, Node *source)
    {
        if (base == nullptr)
        {
            return source;
        }
        else if (source == nullptr)
        {
            return base;
        }

        Node *res, *p;
        // choose the head node
        if (base->val >= source->val)
        {
            res = base;
            base = base->next;
            p = res;
        }
        else
        {
            res = source;
            source = source->next;
            p = res;
        }

        while (base != nullptr && source != nullptr)
        {
            if (base->val >= source->val)
            {
                p->next = base;
                base = base->next;
            }
            else
            {
                p->next = source;
                source = source->next;
            }
            p = p->next;
        }
        if (base == nullptr)
        {
            p->next = source;
        }
        else
        {
            p->next = base;
        }
        return res;
    }

    // Merge list t to the Element Position
    void mergeListTo(const Position &pos, Node *t, const size_t &list_length)
    {
        Node *head = getList(pos);
        setTreeNode(pos, mergeList(head, t));
        uint32_t ns = getElementSize(pos) + list_length;
        setElementSize(pos, ns);
    }

    //Function to prune a leaf ie remove the node and merge the elements with the parent

    bool pruningLeaf(const Position &pos)
    {
        if (getElementSize(pos) <= PruningSize)
        {
            unlockNode(pos);
            return true;
        }

        int b = getBottomLevel();
        int leaves = 1 << b;
        int cnodes = 0;
        for (int i = 0; i < leaves; i++)
        {
            Position tmp;
            tmp.level = b;
            tmp.index = i;
            if (getElementSize(tmp) != 0)
            {
                cnodes++;
            }
            if (cnodes > leaves * 2 / 3)
            {
                break;
            }
        }

        if (cnodes <= leaves * 2 / 3)
        {
            unlockNode(pos);
            return true;
        }
        return false;
    }
    // Split the current list into two lists,
    // then split the tail list and merge to two children.
    void startPruning(const Position &pos)
    {
        if (isLeaf(pos) && pruningLeaf(pos))
        {
            return;
        }

        // split the list, record the tail
        Node *pruning_head = getList(pos);
        int steps = ListTargetSize; // keep in the original list
        for (int i = 0; i < steps - 1; i++)
        {
            pruning_head = pruning_head->next;
        }
        Node *t = pruning_head;
        pruning_head = pruning_head->next;
        t->next = nullptr;
        int tail_length = getElementSize(pos) - steps;
        setElementSize(pos, steps);

        // split the tail list into two lists
        // evenly merge to two children
        if (pos.level != getBottomLevel())
        {
            // split the rest into two lists
            int left_length = (tail_length + 1) / 2;
            int right_length = tail_length - left_length;
            Node *to_right, *to_left = pruning_head;
            for (int i = 0; i < left_length - 1; i++)
            {
                pruning_head = pruning_head->next;
            }
            to_right = pruning_head->next;
            pruning_head->next = nullptr;

            Position lchild = leftOf(pos);
            Position rchild = rightOf(pos);
            if (left_length != 0)
            {
                lockNode(lchild);
                mergeListTo(lchild, to_left, left_length);
            }
            if (right_length != 0)
            {
                lockNode(rchild);
                mergeListTo(rchild, to_right, right_length);
            }
            unlockNode(pos);
            if (left_length != 0 && getElementSize(lchild) > PruningSize)
            {
                startPruning(lchild);
            }
            else if (left_length != 0)
            {
                unlockNode(lchild);
            }
            if (right_length != 0 && getElementSize(rchild) > PruningSize)
            {
                startPruning(rchild);
            }
            else if (right_length != 0)
            {
                unlockNode(rchild);
            }
        }
        else
        { // time to grow the Mound
            grow(pos.level);
            // randomly choose a child to insert
            if (steps % 2 == 1)
            {
                Position rchild = rightOf(pos);
                lockNode(rchild);
                mergeListTo(rchild, pruning_head, tail_length);
                unlockNode(pos);
                unlockNode(rchild);
            }
            else
            {
                Position lchild = leftOf(pos);
                lockNode(lchild);
                mergeListTo(lchild, pruning_head, tail_length);
                unlockNode(pos);
                unlockNode(lchild);
            }
        }
    }
    // This function insert the new node (always) at the head of the current list. It needs to lock the parent & current node.
    // This function may cause the list becoming too long, so we provide pruning algorithm.
    bool regularInsert(const Position &pos, const T &val, Node *newNode)
    {
        // insert to the root node
        if (isRoot(pos))
        {
            lockNode(pos);
            T nv = readValue(pos);
            if (nv <= val)
            {
                newNode->next = getList(pos);
                setTreeNode(pos, newNode);
                uint32_t sz = getElementSize(pos);
                setElementSize(pos, sz + 1);
                if (sz > PruningSize)
                {
                    startPruning(pos);
                }
                else
                {
                    unlockNode(pos);
                }
                return true;
            }
            unlockNode(pos);
            return false;
        }

        // insert to an inner node
        Position parent = parentOf(pos);
        if (!trylockNode(parent))
        {
            return false;
        }
        if (!trylockNode(pos))
        {
            unlockNode(parent);
            return false;
        }
        T pv = readValue(parent);
        T nv = readValue(pos);
        if (pv > val && nv <= val)
        {

            uint32_t sz = getElementSize(pos);
            if (pos.level >= LevelForTraverseParent)
            {
                Node *start = getList(parent);
                while (start->next != nullptr && start->next->val >= val)
                {
                    start = start->next;
                }
                if (start->next != nullptr)
                {
                    newNode->next = start->next;
                    start->next = newNode;
                    while (start->next->next != nullptr)
                    {
                        start = start->next;
                    }
                    newNode = start->next;
                    start->next = nullptr;
                }
                unlockNode(parent);

                Node *curList = getList(pos);
                if (curList == nullptr)
                {
                    newNode->next = nullptr;
                    setTreeNode(pos, newNode);
                }
                else
                {
                    Node *p = curList;
                    if (p->val <= newNode->val)
                    {
                        newNode->next = curList;
                        setTreeNode(pos, newNode);
                    }
                    else
                    {
                        while (p->next != nullptr && p->next->val >= newNode->val)
                        {
                            p = p->next;
                        }
                        newNode->next = p->next;
                        p->next = newNode;
                    }
                }
                setElementSize(pos, sz + 1);
            }
            else
            {
                unlockNode(parent);
                newNode->next = getList(pos);
                setTreeNode(pos, newNode);
                setElementSize(pos, sz + 1);
            }
            if (sz > PruningSize)
            {
                startPruning(pos);
            }
            else
            {
                unlockNode(pos);
            }
            return true;
        }
        unlockNode(parent);
        unlockNode(pos);
        return false;
    }
    bool forceInsertToRoot(Node *newNode)
    {
        Position pos;
        pos.level = pos.index = 0;
        std::unique_lock<Mutex> lck(
            levels_[pos.level][pos.index].lock, std::try_to_lock);
        if (!lck.owns_lock())
        {
            return false;
        }
        uint32_t sz = getElementSize(pos);
        if (sz >= ListTargetSize)
        {
            return false;
        }

        Node *curList = getList(pos);
        if (curList == nullptr)
        {
            newNode->next = nullptr;
            setTreeNode(pos, newNode);
        }
        else
        {
            Node *p = curList;
            if (p->val <= newNode->val)
            {
                newNode->next = curList;
                setTreeNode(pos, newNode);
            }
            else
            {
                while (p->next != nullptr && p->next->val >= newNode->val)
                {
                    p = p->next;
                }
                newNode->next = p->next;
                p->next = newNode;
            }
        }
        setElementSize(pos, sz + 1);
        return true;
    }
    bool forceInsert(const Position &pos, const T &val, Node *newNode)
    {
        if (isRoot(pos))
        {
            return forceInsertToRoot(newNode);
        }

        while (true)
        {
            std::unique_lock<Mutex> lck(
                levels_[pos.level][pos.index].lock, std::try_to_lock);
            if (!lck.owns_lock())
            {
                if (getElementSize(pos) < ListTargetSize && readValue(pos) >= val)
                {
                    continue;
                }
                else
                {
                    return false;
                }
            }
            T nv = readValue(pos);
            uint32_t sz = getElementSize(pos);
            // do not allow the new node to be the first one
            // do not allow the list size tooooo big
            if (nv < val || sz >= ListTargetSize)
            {
                return false;
            }

            Node *p = getList(pos);
            // find a place to insert the node
            while (p->next != nullptr && p->next->val > val)
            {
                p = p->next;
            }
            newNode->next = p->next;
            p->next = newNode;
            // do not forget to change the metadata
            setElementSize(pos, sz + 1);
            return true;
        }
    }
    void binarySearchPosition(
        Position &cur, const T &val)
    {
        Position parent, mid;
        if (cur.level == 0)
        {
            return;
        }
        // start from the root
        parent.level = parent.index = 0;

        while (true)
        { // binary search
            mid.level = (cur.level + parent.level) / 2;
            mid.index = cur.index >> (cur.level - mid.level);

            T mv = optimisticReadValue(mid);
            if (val < mv)
            {
                parent = mid;
            }
            else
            {
                cur = mid;
            }

            if (mid.level == 0 || // the root
                ((parent.level + 1 == cur.level) && parent.level != 0))
            {
                return;
            }
        }
    }
    // The push keeps the length of each element stable
    void moundPush(const T &val)
    {
        Position cur;
        Node *newNode = new Node;
        newNode->val = val;
        uint32_t seed = rand() % (1 << 21);

        while (true)
        {
            // Shall we go the fast path?
            bool go_fast_path = false;
            // Choose the right node to start
            cur = selectPosition(val, go_fast_path, seed);
            if (go_fast_path)
            {
                if (forceInsert(cur, val, newNode))
                {

                    return;
                }
                else
                {
                    continue;
                }
            }

            binarySearchPosition(cur, val);
            if (regularInsert(cur, val, newNode))
            {
                return;
            }
        }
    }
    // Function to pop to the shared buffer
    int popToSharedBuffer(const uint32_t rsize, Node *head)
    {
        Position pos;
        pos.level = pos.index = 0;

        int num = std::min(rsize, (uint32_t)PopBatch);
        for (int i = num - 1; i >= 0; i--)
        {
            // wait until this block is empty
            while (shared_buffer_[i].pnode.load(std::memory_order_relaxed) != nullptr)
                ;
            shared_buffer_[i].pnode.store(head, std::memory_order_relaxed);
            head = head->next;
        }
        if (num > 0)
        {
            top_loc_.store(num - 1, std::memory_order_release);
        }
        setTreeNode(pos, head);
        return rsize - num;
    }

    // If the child nodes have relatively lesser number of elements we will merge them into the parent node
    void mergeDown(const Position &pos)
    {
        if (isLeaf(pos))
        {
            unlockNode(pos);
            return;
        }

        // acquire locks for L and R and compare
        Position lchild = leftOf(pos);
        Position rchild = rightOf(pos);
        lockNode(lchild);
        lockNode(rchild);
        // read values
        T nv = readValue(pos);
        T lv = readValue(lchild);
        T rv = readValue(rchild);
        if (nv >= lv && nv >= rv)
        {
            unlockNode(pos);
            unlockNode(lchild);
            unlockNode(rchild);
            return;
        }

        size_t sum =
            getElementSize(rchild) + getElementSize(lchild) + getElementSize(pos);
        if (sum <= MergingSize)
        {
            Node *l1 = mergeList(getList(rchild), getList(lchild));
            setTreeNode(pos, mergeList(l1, getList(pos)));
            setElementSize(pos, sum);
            setTreeNode(lchild, nullptr);
            setElementSize(lchild, 0);
            setTreeNode(rchild, nullptr);
            setElementSize(rchild, 0);
            unlockNode(pos);
            mergeDown(lchild);
            mergeDown(rchild);
            return;
        }
        // pull from right
        if (rv >= lv && rv > nv)
        {
            swapList(rchild, pos);
            unlockNode(pos);
            unlockNode(lchild);
            mergeDown(rchild);
        }
        else if (lv >= rv && lv > nv)
        {
            // pull from left
            swapList(lchild, pos);
            unlockNode(pos);
            unlockNode(rchild);
            mergeDown(lchild);
        }
    }
    bool deferSettingRootSize(Position &pos)
    {
        if (isLeaf(pos))
        {
            setElementSize(pos, 0);
            unlockNode(pos);
            return true;
        }

        // acquire locks for L and R and compare
        Position lchild = leftOf(pos);
        Position rchild = rightOf(pos);
        lockNode(lchild);
        lockNode(rchild);
        if (getElementSize(lchild) == 0 && getElementSize(rchild) == 0)
        {
            setElementSize(pos, 0);
            unlockNode(pos);
            unlockNode(lchild);
            unlockNode(rchild);
            return true;
        }
        else
        {
            // read values
            T lv = readValue(lchild);
            T rv = readValue(rchild);
            if (lv >= rv)
            {
                swapList(lchild, pos);
                setElementSize(lchild, 0);
                unlockNode(pos);
                unlockNode(rchild);
                pos = lchild;
            }
            else
            {
                swapList(rchild, pos);
                setElementSize(rchild, 0);
                unlockNode(pos);
                unlockNode(lchild);
                pos = rchild;
            }
            return false;
        }
    }
    bool moundPopMany(T &val)
    {
        // pop from the root
        Position pos;
        pos.level = pos.index = 0;
        // the root is nullptr, return false
        Node *head = getList(pos);
        if (head == nullptr)
        {
            unlockNode(pos);
            return false;
        }

        // shared buffer already filled by other threads
        if (PopBatch > 0 && top_loc_.load(std::memory_order_acquire) >= 0)
        {
            unlockNode(pos);
            return false;
        }

        uint32_t sz = getElementSize(pos);
        // get the one node first
        val = head->val;
        Node *p = head;
        head = head->next;
        sz--;

        if (PopBatch > 0)
        {
            sz = popToSharedBuffer(sz, head);
        }
        else
        {
            setTreeNode(pos, head);
        }

        bool done = false;
        if (sz == 0)
        {
            done = deferSettingRootSize(pos);
        }
        else
        {
            setElementSize(pos, sz);
        }

        if (!done)
        {
            mergeDown(pos);
        }

        // p->retire();
        return true;
    }
    // This could guarentee the Mound is empty
    bool isMoundEmpty()
    {
        Position pos;
        pos.level = pos.index = 0;
        return getElementSize(pos) == 0;
    }

    // Return true if the shared buffer is empty
    bool isSharedBufferEmpty()
    {
        return top_loc_.load(std::memory_order_acquire) < 0;
    }
    // Utility function to check if the data structure is empty
    bool isEmpty()
    {
        if (PopBatch > 0)
        {
            return isMoundEmpty() && isSharedBufferEmpty();
        }
        return isMoundEmpty();
    }
    // Function to check if we can pop from the mound
    bool tryPopFromMound(T &val)
    {
        if (isMoundEmpty())
        {
            return false;
        }
        Position pos;
        pos.level = pos.index = 0;

        // lock the root
        if (trylockNode(pos))
        {
            return moundPopMany(val);
        }
        return false;
    }

    // Function to check if we can pop from the shared buffer
    bool tryPopFromSharedBuffer(T &val)
    {
        int get_or = -1;
        if (!isSharedBufferEmpty())
        {
            get_or = top_loc_.fetch_sub(1, std::memory_order_acq_rel);
            if (get_or >= 0)
            {
                Node *c = shared_buffer_[get_or].pnode.load(std::memory_order_relaxed);
                shared_buffer_[get_or].pnode.store(nullptr, std::memory_order_release);
                val = c->val;
                return true;
            }
        }
        return false;
    }
    // Function to pop from the mound
    void moundPop(T &val)
    {

        if (PopBatch > 0)
        {
            if (tryPopFromSharedBuffer(val))
            {
                return;
            }
        }

        while (true)
        {
            if (tryPopFromMound(val))
            {
                return;
            }
            if (PopBatch > 0 && tryPopFromSharedBuffer(val))
            {
                return;
            }
        }
    }
};
