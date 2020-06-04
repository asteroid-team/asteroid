#!/bin/bash
wsj0_dir=
wsj1_dir=
num_jobs=$(nproc --all)
sms_wsj_dir=
python_path=python
json_dir=${sms_wsj_dir}

. utils/parse_options.sh || exit 1;

wsj_8k_zeromean=${sms_wsj_dir}/wsj_8k_zeromean
rir_dir=${sms_wsj_dir}/rirs

echo using ${num_jobs} parallel jobs

if [[ ! -d $wsj_8k_zeromean ]]; then
	echo creating ${wsj_8k_zeromean}
	mpiexec -np ${num_jobs} $python_path -m sms_wsj.database.wsj.write_wav \
		with dst_dir=${wsj_8k_zeromean} wsj0_root=${wsj0_dir} wsj1_root=${wsj1_dir} sample_rate=8000
fi

if [[ ! -d $json_dir ]]; then
	mkdir -p $json_dir
fi

if [[ ! -f $json_dir/wsj_8k_zeromean.json ]]; then
	echo creating $json_dir/wsj_8k_zeromean.json
	$python_path -m sms_wsj.database.wsj.create_json \
		with json_path=$json_dir/wsj_8k_zeromean.json database_dir=$wsj_8k_zeromean as_wav=True
fi

if [[ ! -d $rir_dir ]]; then
	echo "RIR directory does not exist, starting download."
	mkdir -p ${rir_dir}
	wget -qO- https://zenodo.org/record/3517889/files/sms_wsj.tar.gz.parta{a,b,c,d,e} \
		| tar -C ${rir_dir}/ -zx --checkpoint=10000 --checkpoint-action=echo="%u/5530000 %c"
fi

if [[ ! -f $json_dir/sms_wsj.json ]]; then
	echo creating $json_dir/sms_wsj.json
	$python_path -m sms_wsj.database.create_json \
		with json_path=$json_dir/sms_wsj.json rir_dir=$rir_dir \
		wsj_json_path=$json_dir/wsj_8k_zeromean.json

fi


echo creating $sms_wsj_dir files
echo This amends the sms_wsj.json with the new paths.
mpiexec -np ${num_jobs} $python_path -m sms_wsj.database.write_files \
	with dst_dir=${sms_wsj_dir} json_path=$json_dir/sms_wsj.json \
	write_all=True new_json_path=$json_dir/sms_wsj.json