## Pretrained models
Asteroid provides pretrained models through [Hugging Face's Model Hub](https://huggingface.co/models?filter=asteroid).
Have a look at this page to choose which model you want to use.

Enjoy having pretrained models? **Please share your models** if you train some :pray:
It's really simple with the Hub, check the next sections.

### Using them
Loading a pretrained model is super simple!
```python
from asteroid.models import ConvTasNet
model = ConvTasNet.from_pretrained('mpariente/ConvTasNet_WHAM!_sepclean')
```

You can also load it with Hub
```python
from torch import hub
model = hub.load('mpariente/asteroid', 'conv_tasnet', 'mpariente/ConvTasNet_WHAM!_sepclean')
```

### Model caching
When using a `from_pretrained` method, the model is downloaded and cached.
The cache directory is either the value in the `$ASTEROID_CACHE` environment variable,
or `~/.cache/torch/asteroid`.

### Share your models
At the end of each sharing-enabled recipe, all the necessary infos are gathered into a file,
the only thing that's left to is to add it to the Model Hub.
After creating an account ([here](https://huggingface.co/join)), you can
- Add a new model [here](https://huggingface.co/new).
  with a name like `{model_name}_{dataset_name}_{task}_{sampling_rate}`.
- Clone the repo (`git clone the_URL_youre_at`), cd into it.
- Copy the `model_card_template.md` and fill in the missing information.
- Move the pretrained model in the folder, rename it `pytorch.bin`.
- Register files and commit `git add . && git commit -m "Model release: v1"`.
- And push :tada: `git push` :tada:
- Thank you! :pray:

You can have a look at [the docs](https://huggingface.co/docs) for more details!

### Note about licenses
All Asteroid's pretrained models are shared under the
[Attribution-ShareAlike 3.0 (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/)
license. This means that models are released under the same license as the original
training data. **If any non-commercial data is used during training (wsj0, WHAM's noises etc..), the
models are non-commercial use only.**
This is indicated in the bottom of the model page
(ex: [here on the bottom](https://huggingface.co/mpariente/ConvTasNet_WHAM_sepclean)).
