# vela_gulp_continues_acquire_data_27_04_2024.sh
# USAGE
# ./vela_gulp_continues_acquire_data_27_04_2024.sh <file_prefix> <Ethernet_port:eth0> 


file_prefix=$1     # File prefix
net_port=$2   #eth0 #enxd03745fcff37

ACQ_DIR=$3

fname=$file_prefix #"_"$(date +%d_%m_%Y_%H_%m_%s)

cd $ACQ_DIR

echo Present directory: $PWD

echo gulp -i $net_port -r 1000 -C 2 -W 3 -n $file_prefix -o ./.

sudo gulp -i $net_port -r 1000 -C 2 -W 3 -n $file_prefix -o ./.
