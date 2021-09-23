#!/bin/bash

function bench_run()
{
	echo "Running $1"
	$1
}

function bench_select()
{
	if [ "$1" = "intel" ]; then
		bench_run ./saxpy_icc_novec
		bench_run ./saxpy_icc1
		bench_run ./saxpy_icc2
		bench_run ./saxpy_sse
		bench_run ./saxpy_avx
		bench_run ./saxpy_avx_noopt
	elif [ "$1" = "gcc" ]; then
		bench_run ./saxpy_gcc_novec
		bench_run ./saxpy_gcc1
		bench_run ./saxpy_gcc2
		bench_run ./saxpy_gcc3
		bench_run ./saxpy_gcc4
	fi
}

for type in ${@}; do
	echo "Benchmarks type: $type."
	bench_select $type
done
