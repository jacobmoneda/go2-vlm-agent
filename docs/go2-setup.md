# Step-by-step guide to connecting your PC to the Go2 via ethernet.

1.  Turn on Go2 (battery and controller)
2.  Connect ethernet cable to dog and PC
3.  Open a terminal
4.  `ping 192.168.123.18` to test if connected
    - If not connected, windows button > search "view network connections" > click on ethernet device
    - Properties > IPv4 > Properties
    - IP address: 192.168.123.51
    - Subnet Mask: 255.255.255.0
5.  in terminal, `ssh unitree@192.168.123.18`
6.  default password is 123

## Opening a window in VS Code

1.  Click bottom left of VSCode window
2.  Press "connect to Host (SSH)"
