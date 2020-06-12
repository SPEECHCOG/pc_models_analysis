# Python module

This module contains the APC and CPC models. 

The package models have ModelBase abstract class from which
all the models should inherit so that we can train and predict
using the same two methods.

To execute from command line a JSON configuration 
file is needed, please use config_base.json as the
template. By default train.py and predict.py will use
config.json as the configuration file, however you can
provide a different path using:

```bash
$ python train.py --config path_to_json_file
$ python predict.py --config path_to_json_file
```

