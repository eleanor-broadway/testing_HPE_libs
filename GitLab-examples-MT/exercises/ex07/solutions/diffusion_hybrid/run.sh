#!/bin/sh
if [ $MV2_COMM_WORLD_LOCAL_SIZE -eq 24 ]
then
case "$MV2_COMM_WORLD_LOCAL_RANK" in
0) LIST=0
;;
1) LIST=1
;;
2) LIST=2
;;
3) LIST=3
;;
4) LIST=4
;;
5) LIST=5
;;
6) LIST=6
;;
7) LIST=7
;;
8) LIST=8
;;
9) LIST=9
;;
10) LIST=10
;;
11) LIST=11
;;
12) LIST=12
;;
13) LIST=13
;;
14) LIST=14
;;
15) LIST=15
;;
16) LIST=16
;;
17) LIST=17
;;
18) LIST=18
;;
19) LIST=19
;;
20) LIST=20
;;
21) LIST=21
;;
22) LIST=22
;;
23) LIST=23
;;
esac
fi
if [ $MV2_COMM_WORLD_LOCAL_SIZE -eq 12 ]
then
case "$MV2_COMM_WORLD_LOCAL_RANK" in
0) LIST=0,1
;;
1) LIST=2,3
;;
2) LIST=4,5
;;
3) LIST=6,7
;;
4) LIST=8,9
;;
5) LIST=10,11
;;
6) LIST=12,13
;;
7) LIST=14,15
;;
8) LIST=16,17
;;
9) LIST=18,19
;;
10) LIST=20,21
;;
11) LIST=22,23
;;
esac
fi
if [ $MV2_COMM_WORLD_LOCAL_SIZE -eq 8 ]
then
case "$MV2_COMM_WORLD_LOCAL_RANK" in
0) LIST=0,1,2
;;
1) LIST=3,4,5
;;
2) LIST=6,7,8
;;
3) LIST=9,10,11
;;
4) LIST=12,13,14
;;
5) LIST=15,16,17
;;
6) LIST=18,19,20
;;
7) LIST=21,22,23
;;
esac
fi
if [ $MV2_COMM_WORLD_LOCAL_SIZE -eq 4 ]
then
case "$MV2_COMM_WORLD_LOCAL_RANK" in
0) LIST=0,1,2,3,4,5
;;
1) LIST=6,7,8,9,10,11
;;
2) LIST=12,13,14,15,16,17
;;
3) LIST=18,19,20,21,22,23
;;
esac
fi
if [ $MV2_COMM_WORLD_LOCAL_SIZE -eq 2 ]
then
case "$MV2_COMM_WORLD_LOCAL_RANK" in
0) LIST=0,1,2,3,4,5,6,7,8,9,10,11
;;
1) LIST=12,13,14,15,16,17,18,19,20,21,22,23
;;
esac
fi
if [ $MV2_COMM_WORLD_LOCAL_SIZE -eq 1 ]
then
case "$MV2_COMM_WORLD_LOCAL_RANK" in
0) LIST=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
;;
esac
fi
# running
numactl --physcpubind=$LIST "$@"
