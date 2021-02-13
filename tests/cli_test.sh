# Save a model (tmp.th) and two wavfiles (tmp.wav, tmp2.wav)
python -m pip install -e . --quiet
python tests/cli_setup.py

# asteroid-register-sr`
coverage run -a `which asteroid-register-sr` tmp.th 8000

# asteroid-infer
coverage run -a `which asteroid-infer` tmp.th --files tmp.wav
coverage run -a `which asteroid-infer` tmp.th --files tmp.wav tmp2.wav --force-overwrite
coverage run -a `which asteroid-infer` tmp.th --files tmp.wav --ola-window 1000 --force-overwrite
coverage run -a `which asteroid-infer` tmp.th --files tmp.wav --ola-window 1000 --ola-no-reorder --force-overwrite

# asteroid-upload
echo "n" | coverage run -a `which asteroid-upload` publish_dir --uploader "Manuel Pariente" --affiliation "Loria" --use_sandbox --token $ACCESS_TOKEN

# asteroid-version
coverage run -a `which asteroid-versions`
