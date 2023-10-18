#include <future>
#include <memory>
#include <queue>
#include <vector>
#include <iostream>

/// @brief Consumer and producer model.
namespace tinycv
{
    template <typename result_ty, typename input_ty, typename model_ty>
    class Cpm
    {
    private:    
        struct Item
        {
            input_ty input;
            std::shared_ptr<std::promise<result_ty>> pro;
        };

    public:
        /// @brief 
        /// @tparam load_ty 
        /// @param load 
        /// @param max_items 
        /// @param stream 
        /// @return 
        template <typename load_ty>
        bool start(const load_ty &load, int max_items = 1, void *stream = nullptr)
        {
            stop();
            stream_ = stream;
            max_items_ = max_items;
            std::promise<bool> status;
            // launch worker thread.
            worker_ = std::make_shared<std::thread> (&Cpm::worker<load_ty>, this, std::ref(load), std::ref(status));
            return status.get_future().get();
        }

        /// @brief @see stop()
        ~Cpm() { stop(); }

        /// @brief Stop the cpm model
        void stop()
        {
            run_ = false;
            cond_.notify_one();
            {
                std::unique_lock<std::mutex> lock(mtx_);
                while (!input_queue_.empty())  // if tasks left.
                {
                    Item &item = input_queue_.front();
                    if (item.pro)
                        item.pro->set_value(result_ty());
                    input_queue_.pop();
                }
            }
            if (worker_ && worker_->joinable())
            {
                worker_->join();
                worker_.reset();
            }
        }

        /// @brief Produce one input for cpm model, convert from "input_ty" to "Item" 
        /// this function should be called by producer.
        /// @param input (const input_ty &) 
        /// @return std::share_future<result_ty>
        std::shared_future<result_ty> commit(const  input_ty &input)
        {
            Item item;
            item.input = input;
            item.pro.reset(new std::promise<result_ty>());
            {
                std::unique_lock<std::mutex> lock(mtx_);
                input_queue_.push(item);
            }
            // notify get_item_and_wait() function to fetch item from input_queue_
            cond_.notify_one();
            return item.pro->get_future();
        }
        
        /// @brief Produce some inputs for cpm model. 
        /// this function should be called by producer.
        /// @note This function will be called on main thread, so we need to add lock.
        /// @param inputs 
        /// @return std::vector<std::shared_future<result_ty>>
        std::vector<std::shared_future<result_ty>> commit(const std::vector<input_ty> &inputs)
        {
            std::vector<std::shared_future<result_ty>> ret;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                for (const input_ty &input: inputs)
                {
                    Item item;
                    item.input = input;
                    item.pro = std::make_shared<std::promise<result_ty>>();
                    ret.emplace_back(item.pro->get_future());
                    input_queue_.push(item);
                }
            }
            cond_.notify_one();
            return ret;
        }

    private:
        /// @brief Fetch one item from input queue.
        /// this function should be called by consumer.
        /// @param fetch_item (Item &)
        /// @return true on success, false on failure.
        bool get_item_and_wait(Item &fetch_item)
        {
            std::unique_lock<std::mutex> lock(mtx_);
            // wait for commit to produce task.
            // quit when run_ == false or input_que_ is not empty. 
            cond_.wait(lock, [&]() -> bool { return !run_ || !input_queue_.empty(); });
            if (!run_)
                return false;
            
            fetch_item = std::move(input_queue_.front());
            input_queue_.pop();
            return true;
        }

        /// @brief Fetch items from input queue.
        /// this function should be called by consumer.
        /// @param fetch_items (std::vector<Item> &fetch_items)
        /// @param max_size (int)
        /// @return true on success, false on failure.
        bool get_item_and_wait(std::vector<Item> &fetch_items, int max_size)
        {
            std::unique_lock<std::mutex> lock(mtx_);
            // wait for commit to produce task.
            // quit when run_ == false or input_que_ is not empty. 
            cond_.wait(lock, [&]() -> bool { return !run_ || !input_queue_.empty(); });
            if (!run_)
                return false;
            fetch_items.clear();
            for (int i = 0; i < max_size && !input_queue_.empty(); ++i)
            {
                fetch_items.emplace_back(std::move(input_queue_.front()));
                input_queue_.pop();
            }
            return true;
        }

        /// @brief 
        /// @tparam load_ty 
        /// @param load 
        /// @param status 
        template <typename load_ty>
        void worker(const load_ty &load, std::promise<bool> &status)
        {
            // load engine.
            std::shared_ptr<model_ty> model = load();
            if (!model)
            {
                status.set_value(false);
                return;
            }
            run_ = true;
            status.set_value(true);
            std::vector<Item> fetch_items;
            std::vector<input_ty> inputs;
            while (get_item_and_wait(fetch_items, max_items_))
            {
                inputs.resize(fetch_items.size());
                std::transform(fetch_items.begin(), fetch_items.end(), inputs.begin(), [](Item &item) { return item.input; });
                auto ret = model->forward(inputs, stream_);
                for (int i = 0; i < fetch_items.size(); ++i)
                {
                    if (i < ret.size())
                        fetch_items[i].pro->set_value(ret[i]);
                    else 
                        fetch_items[i].pro->set_value(result_ty());
                }
                inputs.clear();
                fetch_items.clear();
            }
            model.reset();
            run_ = false;
        }

    private:
        std::condition_variable cond_;
        std::queue<Item> input_queue_;
        std::mutex mtx_;
        std::shared_ptr<std::thread> worker_;
        volatile bool run_ = false;
        volatile int max_items_ = 0;
        void *stream_ = nullptr;
    };
};