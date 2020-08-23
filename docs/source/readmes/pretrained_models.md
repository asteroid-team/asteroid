## Pretrained models
Asteroid provides pretrained models through the
[Asteroid community](https://zenodo.org/communities/asteroid-models) in Zenodo.
Have a look at the Zenodo page to choose which model you want to use.

Enjoy having pretrained models? **Please share your models** if you train some,
we made it simple with the `asteroid-upload` CLI, check the next sections.

### Using them
Loading a pretrained model is super simple!
```python
from asteroid.models import ConvTasNet
model = ConvTasNet.from_pretrained('mpariente/ConvTasNet_WHAM!_sepclean')
```
Use the [search page](https://zenodo.org/communities/asteroid-models/search)
if you want to narrow your search.

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
At the end of each sharing-enabled recipe, all the necessary infos are gathered into a file, the only thing
that's left to do is to run
```bash
asteroid-upload exp/your_exp_dir/publish_dir --uploader "Name Here"
```
Ok, not really. First you need to register to [Zenodo](https://zenodo.org/) (Sign in with GitHub: ok),
[create a token](https://zenodo.org/account/settings/applications/tokens/new/) and use it with
the `--token` option of the CLI, or by setting the `ACCESS_TOKEN` environment variable.
If you plan to upload more models (and you should :innocent:), you can fill in your infos in
`uploader_info.yml` at the root, like this.
```yaml
uploader: Manuel Pariente
affiliation: INRIA
git_username: mpariente
token: TOKEN_HERE
```

### Note about licenses
All Asteroid's pretrained models are shared under the
[Attribution-ShareAlike 3.0 (CC BY-SA 3.0)](https://creativecommons.org/licenses/by-sa/3.0/)
license. This means that models are released under the same license as the original
training data. **If any non-commercial data is used during training (wsj0, WHAM's noises etc..), the
models are non-commercial use only.**
This is indicated in the bottom of the corresponding Zenodo page (ex: [here](https://zenodo.org/record/3903795#collapseTwo)).
