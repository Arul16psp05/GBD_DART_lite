sudo ip link set dev eno1 down       # eno1 is the default name in system
sudo ip link set dev eno1 name eth0
sudo ip link set dev eth0 up
sudo ifconfig eth0 10.0.0.1 netmask 255.0.0.0 broadcast 0.0.0.0 

sudo arp -s 10.0.1.1 00:0a:35:01:8e:31
