#!/usr/bin/env bash

DATASET_LAYOUT=()

DATASET_MINIMIZE=false
while getopts ":s:md:" arguments; do
    case "${arguments}" in
        s) DATASET_SPLIT=${OPTARG};;
		m) DATASET_MINIMIZE=true;;
		d) DATASET_LAYOUT+=("$OPTARG");;
    esac
done

if [ -z "${DATASET_SPLIT}" ]; then
    echo "You must specify the download split (train|val|test) using the '-s' flag."
    exit
fi

readarray -t DataScenesArray < scripts/preprocessing/${DATASET_SPLIT}.txt
readarray -t DATASET_LAYOUT < scripts/preprocessing/directory_layout.txt



for folder in ${DATASET_LAYOUT[@]}; do
	echo $folder
	mkdir -p MVPd/${DATASET_SPLIT}/${folder}
done


if [ "$DATASET_MINIMIZE" = true ]; then
	python3 -c "import json; json.dump({'videos':[], 'annotations': [], 'categories': []}, open('MVPd/${DATASET_SPLIT}/panoptic_${DATASET_SPLIT}.json', 'w'))"
else
	python3 -c "import json; json.dump({'videos':[], 'annotations': [], 'categories': [], 'instances': []}, open('MVPd/${DATASET_SPLIT}/panoptic_${DATASET_SPLIT}.json', 'w'))"
fi



for scene in ${DataScenesArray[@]}; do

	if [ -e "MVPd/${DATASET_SPLIT}/${scene}.tar.gz" ]; then
	    for folder in ${DATASET_LAYOUT[@]}; do
			tar -C MVPd/${DATASET_SPLIT}/${folder}/ -xf MVPd/${DATASET_SPLIT}/${scene}.tar.gz ${scene}/${folder}/ --strip-components=2
		done
	 	
		tar -C MVPd/${DATASET_SPLIT}/ -xf MVPd/${DATASET_SPLIT}/${scene}.tar.gz ${scene}/panoptic.json --strip-components=1

		if [ "$DATASET_MINIMIZE" = true ]; then
			python3 scripts/preprocessing/panoptic_merge.py MVPd/${DATASET_SPLIT}/panoptic.json MVPd/${DATASET_SPLIT}/panoptic_${DATASET_SPLIT}.json --minimize
		else
			python3 scripts/preprocessing/panoptic_merge.py MVPd/${DATASET_SPLIT}/panoptic.json MVPd/${DATASET_SPLIT}/panoptic_${DATASET_SPLIT}.json
		fi

		rm MVPd/${DATASET_SPLIT}/panoptic.json
		rm MVPd/${DATASET_SPLIT}/${scene}.tar.gz
	else 
	    echo "Expected file (MVPd/${DATASET_SPLIT}/${scene}.tar.gz) does not exist"
	fi 

	

done

