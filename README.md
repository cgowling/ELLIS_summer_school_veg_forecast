# 

## Login to Raven cluster 
Raven documentation 
https://docs.mpcdf.mpg.de/doc/computing/raven-user-guide.html

If not on an internal network, depending on your setup this should require your password and OTP 
```
$ ssh <user>@gate.mpcdf.mpg.de
```

From within network or after the above login, if on external network 
```
$ ssh <user>@raven.mpcdf.mpg.de
```

## Set up ssh Key 
Once you are logged into Raven you need to setup an ssh key for connecting with git.

Below are some very brief instructions if you need a memory jog, if you want more detailed instructions see here: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

Check for exisitng keys 
``` shell
$ ls -al ~/.ssh
```

generate ssh key 
```shell
$ ssh-keygen -t rsa -b 4096 -C "<your_email>@example.com"
```

Create a config file  (unless you already have one)

```shell
emacs ~/.ssh/config
```

Add this to your config file, ensuring the file specified matches the public key you just created. 
```text
Host github.com
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
```

Copy contents of your public key and add a new key to  your github account

## Clone ELLIS_summer_school_veg_forecast repository to your Raven account
Once you've logged in your should be somewhere like `/u/<userid>`. From here clone the repository 
```
$ git clone git@github.com:cgowling/ELLIS_summer_school_veg_forecast.git
```



## Create conda environment 
Navigate into the repository you have just cloned 

```
$ module load anaconda/3/2023.03
```

```
$ conda create --name myenv --file spec-file.txt
```

## Test run Batch script 

Now that  the Initial setup is complete we can test the setup by running a small batch run 



### Data 


The data for this challenge can be found at the shared directory 

```
/ptmp/mp002/ellis/veg_forecast
```
