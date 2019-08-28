These exercises are designed to get you familiar with OpenKiwi's code by doing some small changes to its models.

# Installing OpenKiwi for development

Requirement: A python version >= 3.6

First, make sure you have `virtualenv` and `virtualenvwrapper` installed. 

```
# virtualenv install
pip3 install virtualenv

# virtualenvwrapper install
pip3 install virtualenvwrapper
printf 'export WORKON_HOME=$HOME/.virtualenvs\nexport PROJECT_HOME=$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh' | cat - ~/.profile > $$tmp && mv $$tmp ~/.profile && source ~/.profile
```

Then, create a new virtualenv for this session

```
mkvirtualenv --python PATH_TO_PYTHON>=3.6 kiwi
```

Now we can clone `kiwi` and install it for local development using poetry

```
git clone https://github.com/Unbabel/OpenKiwi.git
pip install poetry
poetry install
```

# Messing around with Model internals

You have previously trained a NuQE model. Let's change some of its architecture and observe what happens.

By default, NuQE applies linear layers with ReLU nonlinearitries and Gated Recurrent Units.

Try replacing the nonlinearities with some other nonlinearity of your choice, and train again to see if the performance changes.
You can do the same with the Gated Recurrent Units: Use LSTMs for instance

```
kiwi train --config ~/KiwiCutter/exercises/train_nuqe.yaml
```
# Going Deeper

We all know that deep learning is the key to success. Try changing the architecture of NuQE by e.g. adding more blocks of the same type, or adding a convolution as an output layer.