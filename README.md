This repository is to implement mosaic calculation in single thread, OpenMP and CUDA
which is one of my assignmnet on my master course
The initial intention is that few mosaic calculation in CUDA on github, this is an example.
It is command line program, Usages: 

mosaic C M -i input_file -o output_file [options]
where:
        C              Is the mosaic cell size which should be any positive
                       power of 2 number
        M              Is the mode with a value of either CPU, OPENMP, CUDA or
                       ALL. The mode specifies which version of the simulation
                       code should execute. ALL should execute each mode in
                       turn.
        -i input_file  Specifies an input image file
        -o output_file Specifies an output image file which will be used
                       to write the mosaic image
[options]:
        -f ppm_format  PPM image output format either PPM_BINARY (default) or
                       PPM_PLAIN_TEXT

mosaic.c calculates in single thread and OpenMP. It only run on gcc compiler, e.g. gcc mosaic.c -o mosaic -O2 -fopenmp
Makefile provides automatically mosaic.c compile on gcc, using make command, e.g. make -j4
mosaic.cu calculates in single thread, OpenMP and CUDA. It only run on Visual Stdio
(PS: The code has bugs, but can run. This may not be updated)


这个是用来做马赛克的，会用单线程，OpenMP和CUDA 库来实现。是我上硕士的小作业之一。
github上用CUDA写马赛克的比较少， 这个可以作为例子吧
这个程序只能用命令行来运行， 参见：

mosaic C M -i input_file -o output_file [options]
where:
        C              Is the mosaic cell size which should be any positive
                       power of 2 number
        M              Is the mode with a value of either CPU, OPENMP, CUDA or
                       ALL. The mode specifies which version of the simulation
                       code should execute. ALL should execute each mode in
                       turn.
        -i input_file  Specifies an input image file
        -o output_file Specifies an output image file which will be used
                       to write the mosaic image
[options]:
        -f ppm_format  PPM image output format either PPM_BINARY (default) or
                       PPM_PLAIN_TEXT

mosaic.c 只有单线程计算和OpenMP加速，只能用gcc编译, e.g. gcc mosaic.c -o mosaic -O2 -fopenmp
Makefile 自动编译mosaic.c文件, 用make命令, e.g. make -j4
mosaic.cu 都包含，但是只能用Visual Stdio运行
(PS: 这个程序肯定有很多bug，很大可能不会再更新)