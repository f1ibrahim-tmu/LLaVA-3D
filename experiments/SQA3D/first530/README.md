Applying LLaVa-3D to the first 530 of the 650 SQA3D (since I wanted to ensure that I get results before the meeting)

this run was from sqa3d; the json is the predictions, the json has the predictions from LLaVa, the txt shows the performance of those predictions.
./llava-3d-7b-sqa3d_test_answer-first530.json
./llava-3d-7b-sqa3d_test_answer-first530.txt

these runs were from sqa3d_distributed; it is the same as ./llava-3d-7b-sqa3d_test_answer-first530.json, but split in two (although both run on the same GPU simultaneously)
./llava-3d-7b-sqa3d_test_answer-first530-chunk0of2.json
./llava-3d-7b-sqa3d_test_answer-first530-chunk1of2.json 

