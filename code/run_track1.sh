
################################################################
#TD-SV-FreeText system: Speaker-sentence labels + Speaker labels
#Steps
#   1. First the classification model is trained in text-independent manner using VoxCeleb 1 & 2 data sets. This classification code can be found in "run_ti.sh"
#   2. This text-independent model is finetuned using tdsv2024 challenge track1 training data.
#   3. Once the classification network is trained, speaker embeddings can be extracted for enroll and test data. This speaker embeddings are compared using cosine similarity score
#   4. Speaker embedding extraction and scoring codes can be found track1_evaluate.py
################################################################


ti_model_location=exp/ti_model/model/model000000015.model
save_name=exp/Track1_TD-SV-FreeText
yaml_name=Track1_TD-SV-FreeText.yaml

mkdir -p $save_name/model
cp $ti_model_location $save_name/model/model000000001.model
name=$yaml_name
python trainSpeakerNet_track1_td-sv-freetext.py --config yaml/$name --distributed >> log/$name.log




################################################################
#TD-SV system: Speaker-sentence labels
#Steps
#   1. First the classification model is trained in text-independent manner using VoxCeleb 1 & 2 data sets. This classification code can be found in "run_ti.sh"
#   2. This text-independent model is finetuned using tdsv2024 challenge track1 training data.
#   3. Once the classification network is trained, speaker embeddings can be extracted for enroll and test data. This speaker embeddings are compared using cosine similarity score
#   4. Speaker embedding extraction and scoring codes can be found track1_evaluate.py
################################################################

ti_model_location=exp/ti_model/model/model000000015.model
save_name=exp/Track1_TD-SV
yaml_name=Track1_TD-SV.yaml

mkdir -p $save_name/model
cp $ti_model_location $save_name/model/model000000001.model
name=$yaml_name
python trainSpeakerNet_track1_td-sv.py --config yaml/$name --distributed >> log/$name.log

