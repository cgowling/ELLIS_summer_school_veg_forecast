# 

## Login to Raven cluster 
Raven documentation 
https://docs.mpcdf.mpg.de/doc/computing/raven-user-guide.html

## Login to Raven 
 (if not on a network that allows direct login)
```
$ ssh <user>@gate.mpcdf.mpg.de
```


Requires password and OTP 
From within network and after the above login, if on external network 
```
$ssh <user>@raven.mpcdf.mpg.de
```


## Set up ssh Key 


setting up git  https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent 

Check for exisitng keys 
``` shell
$ ls -al ~/.ssh
```

generate ssh key 
```shell
$ ssh-keygen -t rsa -b 4096 -C "<your_email>@example.com"
```

Add to ssh config file 

```shell
emacs ~/.ssh/config
```
Add this to your config file, ensuring file 
```text
Host github.com
  AddKeysToAgent yes
  UseKeychain yes
  IdentityFile ~/.ssh/id_ed25519
```

copy contents of  your public key and add a new key to  your github account

## Clone repository 

```
$ git clone git@github.com:cgowling/ELLIS_summer_school_veg_forecast.git
```


## create conda environment 

```
$ module load anaconda/3/2023.03
```

```
$ conda env create -f environment.yaml
```

## test run Batch script 
