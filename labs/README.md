# MT Marathon 2019

Teaching materials from the 14th Machine Translation Marathon 2019.


## Instructions for labs

### 1. Which machine can I use for tutorials?

The most convenient way will be to use your own machine if it has a GPU. Both a remote server and laptop will be fine. If you have an account on Valhalla, please use it for all tutorials. See each tutorial for what software needs to be installed.

### 2. How to log in into a PC in labs? 

Machines in labs have local accounts.

    Login: <LOGIN>
    Password: “<PASSWORD>XYZ”

<LOGIN> and <PASSWORD> will be written on a whiteboard, and XYZ are last 3 characters (after the dash ‘-’) of the computer name displayed on the screen or written on a stick on top of your monitor. These are Windows machines, but they have an SSH client called *mobaXterm*, or you can download the portable PuTTy from:

    https://the.earth.li/~sgtatham/putty/latest/w64/putty.exe

### 3. How to connect to the virtual machine provided by organisers?

We have prepared a number of virtual machines with Google Cloud. Please note that the machines will be started/stopped a few hours before/after a tutorial every day. Logging details, i.e. IP address and password (<IP_ADDRESS> and <PASSWORD> in the example below) will be provided to each student individually at the beginning of each lab.

Connect to a running VM as user "mtm" using an SSH client of your choice, for example:

    ssh mtm@<IP_ADDRESSS>

and type <PASSWORD>. The user has sudo. 

OpenKiwi and Tensor2Tensor are installed within `python3` environment, Nematus is cloned into `~/nematus`, and marian-dev is compiled in `~/marian-dev/build`.
