//
// Created by Ran Xian on 8/5/16.
//

#ifndef TX1DNN_DATA_BUFFER_H
#define TX1DNN_DATA_BUFFER_H

#include <glog/logging.h>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>

/**
 * @brief The buffer wrapper for a bunch of data.
 */
class DataBuffer {
 public:
  DataBuffer() : size_(0), buffer_(nullptr) {}
  /**
   * @brief Constructor.
   * @param buf The data hold by the buffer.
   * @param size The size of the buffer.
   */
  DataBuffer(size_t size) : size_(size) {
    buffer_ =
        std::shared_ptr<char>(new char[size], std::default_delete<char[]>());
    buffer_ptr_ = buffer_.get();
  }

  /**
   * @brief Construct a data buffer from existing data pointer.
   * @param data The pointer to data.
   * @param size Size of the buffer
   */
  DataBuffer(void *data, size_t size) : size_(size), buffer_ptr_(data){};

  /**
   * @brief Construct a data buffer from content of a file.
   * @param filename
   * @return
   */
  DataBuffer(const std::string &filename) {
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    CHECK(ifs) << "Can't open the file. Please check " << filename;

    ifs.seekg(0, std::ios::end);
    size_ = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    LOG(INFO) << "Read from file into buffer: " << filename.c_str() << " ... "
              << size_ << " bytes";

    AllocateBuffer(size_);
    ifs.read((char *)buffer_ptr_, size_);
    ifs.close();
  }
  /**
   * @brief The pointer to the buffer.
   */
  inline void *GetBuffer() { return buffer_ptr_; }

  /**
   * @brief Get a const pointer to the buffer.
   */
  inline const void *GetBuffer() const { return buffer_ptr_; }
  /**
   * @brief Get size of the buffer.
   */
  inline size_t GetSize() const { return size_; }
  /**
   * @brief Clone another data buffer, deeply copy bytes.
   */
  void Clone(const DataBuffer &data_buffer) {
    CHECK(size_ == data_buffer.GetSize())
        << "Can't clone buffer of a different size";
    memcpy(buffer_ptr_, data_buffer.GetBuffer(), size_);
  }

  void Clone(const void *src, size_t size) {
    CHECK(size <= size_) << "Size exceeds the size of the data buffer";
    memcpy(buffer_ptr_, src, size);
  }

 private:
  /**
   * @brief Allocate a buffer with size.
   * @param size The size of the buffer to be allocated
   */
  inline void AllocateBuffer(size_t size) {
    buffer_ =
        std::shared_ptr<char>(new char[size], std::default_delete<char[]>());
    buffer_ptr_ = buffer_.get();
  }

 private:
  size_t size_;
  std::shared_ptr<char> buffer_;
  void *buffer_ptr_;
};

#endif  // TX1DNN_DATA_BUFFER_H