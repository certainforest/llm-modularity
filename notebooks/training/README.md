## About
This contains instructions for connecting to the GPU cloud instance. It has a Nvidia RTX A6000 GPU with 48 GB of RAM. Total disk space is 100 GB.

## Setup
1. Connect to Jupyter Lab.
2. Create a PAT in Github with full read/write permissions.
3. In a terminal in Jupyter lab, `cd` to your user folder. Then, clone your repo with `git clone https://{PAT}@github.com/{USERNAME}/{REPO-NAME}.git`.
4. Add credentials to this git repo.
    ```
    git config user.email "{some_email_connected_to_your_github@email.com}"
    git config user.name {your_name}
    ```
5. Now after changes you should be able to push normally, `git push origin master`.

**Important:** Push VERY REGULARLY to the remote Git server. Make work as state-independent as possible, as everything not saved in Git will be destroyed every time the cloud server is turned off.
*Note*: Hidden files aren't shown in the Jupyter lab file sidebar (you'll have to use `nano`), so be careful managing `.env` - make sure you don't commit secrets to your Git repo! If you create a secrets file, download it to your local computer when you're finished with your work, as it'll be destroyed since it's outside Git.

## Installing packages
Don't setup a new virtual env. By default, you'll always be using the Python3.10 env that comes with this Jupyter lab.

To install packages, just do `!pip install {package}` from a `.ipynb` file. Make sure you save the `!pip install {package}` code you use, as you may have to reinstall them whenever the server is rebooted.

## Monitoring
You can use the following function to monitor GPU memory:
```
import torch

def check_memory():
    print("Allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("Reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("Total: %fGB"%(torch.cuda.get_device_properties(0).total_memory/1024/1024/1024))

check_memory()
```

Disk space can be monitered with the command `du -hs $HOME /workspace/*`. We have 100GB available in total.

# Misc/Admin
- To install `nano`, run `apt update & apt upgrade & apt install nano`
- If there's an error, open a web terminal and run `pip install jupyterlab ipywidgets jupyterlab-widgets --upgrade`