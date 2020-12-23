if [ ! -d ./build ]; then
  mkdir build
fi

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )


# nvcc src/grid_sampling.cu -o build/grid_sampling.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 src/grid_sampling.cpp build/grid_sampling.cu.o -o build/grid_sampling.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
#
# nvcc src/voxel_sampling.cu -o build/voxel_sampling.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 src/voxel_sampling.cpp build/voxel_sampling.cu.o -o build/voxel_sampling.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include

nvcc src/roi_logits_to_attrs.cu -o build/roi_logits_to_attrs.cu.o -c ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 src/roi_logits_to_attrs.cpp build/roi_logits_to_attrs.cu.o -o build/roi_logits_to_attrs.so -shared ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]} -L /usr/local/cuda/lib64/ -I /usr/local/cuda/include
