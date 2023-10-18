# CUDA_Relation

A Simple case for learning trt and quantization
===============================================

## configure

```shell
mkdir build
cd build
cmake ..
make -j16
```

## run 

```shell
cd workspace
./infer_image
```

## demo

* `Generate int8 quantization model`

  ```shell
  cd quant/yolov5
  python yolov5_qat.py
  ```

* [`infer single image`](./src/infer_image.cpp)

  ```cpp
  std::string image_path = "./train.jpg";
  cv::Mat image = cv::imread(image_path);
      
  std::string engine_path = "../../quant/yolov5/engine_files/yolov5n-ptq-percentile-99.99-1024.engine";
  auto infer = tinycv::yolo::load(engine_path);
  warm_up(infer);
  
  tinycv::MixMat in_mat({image.rows, image.cols, image.channels()}, tinycv::DataType::UINT8);
  int height = image.rows, width = image.cols, n_channels = image.channels();
  size_t n_bytes = static_cast<size_t>(height) * width *n_channels; 
  in_mat.ref_data({image.rows, image.cols, image.channels()}, image.data, n_bytes, nullptr, 0);
  INFO("Start inference...");
  auto boxes = infer->forward(in_mat);
  return 0;
  ```

* [`infer_video_with_CUVID`](./src/infer_hardware.cpp)

  ```cpp
  const std::string url = "video=HP Wide Vision HD Camera"; 
  const std::string input_fmt_name = "dshow";  
  const std::string input_fmt_name = "";  
  tinycv::NVVideoCap nv_cap(url, input_fmt_name, 0, true, false);
  // initialize cpm
  using cpm_ty = tinycv::Cpm<tinycv::yolo::BoxArray, tinycv::MixMat, tinycv::yolo::YoloInfer>;
  const int n = 3;
  std::vector<cpm_ty> cpms(n);
  std::vector<cudaStream_t> streams(n, nullptr);
  bool status = true;
  const std::string engine_path = "../../quant/yolov5/quant_yolov5n_replace_to_quantization.engine"; 
  for (int i = 0; i < n; ++i)
  {
      CHECK_CUDA_RUNTIME(cudaStreamCreate(&streams[i]));
      status &= cpms[i].start(
          [&engine_path]() { return tinycv::yolo::load(engine_path); },
          1,
          streams[i]
      );
  }
  if (!status)
      INFO_ERROR("CPM launch failed!\n");
  
  int height = nv_cap.get_height();
  int width = nv_cap.get_width();
  tinycv::MixMat in_mat({height, width, 3}, tinycv::DataType::UINT8);
  cv::Mat image;
  bool done = false;
  cudaEvent_t begin, end;
  CHECK_CUDA_RUNTIME(cudaEventCreate(&begin));
  CHECK_CUDA_RUNTIME(cudaEventCreate(&end));
  float cost = 0.;
  CHECK_CUDA_RUNTIME(cudaEventRecord(begin, streams[0]));
  std::vector<std::shared_future<tinycv::yolo::BoxArray>> futs(n);
  while (!done)
  {
      in_mat = nv_cap.read_mix_mat(streams[0]);
      for (int i = 0; i < n; ++i)
      {
          in_mat.set_stream(streams[i]);
          if (in_mat.empty())
          {
              done = true;
              break;
          }
          futs[i] = cpms[i].commit(in_mat);
      }
      for (int i = 0; i < n && !done; ++i)
      {
          image = cv::Mat(height, width, CV_8UC3, in_mat.host());
          auto boxes = futs[i].get();
      }
  }
  ```

  

